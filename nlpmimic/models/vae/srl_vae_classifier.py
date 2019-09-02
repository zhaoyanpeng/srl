"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""
from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Linear, Dropout

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

from nlpmimic.nn.util import gumbel_softmax, gumbel_sinkhorn
from nlpmimic.training.metrics import FeatureBasedF1Measure 
from nlpmimic.training.metrics import ClusteringBasedF1Measure 
from nlpmimic.training.metrics import DependencyBasedF1Measure

@Model.register("srl_vae_classifier")
class SrlVaeClassifier(Model):
    """ A srl classifier make predictions of p(y|x).
    """
    def __init__(self, vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder = None,
                 lemma_embedder: TextFieldEmbedder = None,
                 label_embedder: Embedding = None,
                 predt_embedder: Embedding = None, # predt: predicate
                 ctx_encoder: Seq2SeqEncoder = None,
                 seq_encoder: Seq2SeqEncoder = None,
                 psign_dim: int = None, # psign: predicate sign bit (0/1) 
                 tau: float = None,
                 tunable_tau: bool = False,
                 suppress_nonarg: bool = False,
                 seq_projection_dim: int = None, 
                 token_dropout: float = 0.,
                 lemma_dropout: float = 0.,
                 label_dropout: float = 0.,
                 predt_dropout: float = 0.,
                 label_smoothing: float = None,
                 embed_lemma_ctx: bool = False,
                 calc_seq_vector: bool = False,
                 metric_type: str = 'dependency',
                 ignore_span_metric: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlVaeClassifier, self).__init__(vocab, regularizer)
        self.signature = 'classifier'
        
        self.seq_encoder = seq_encoder
        self.token_embedder = token_embedder
        self.lemma_embedder = lemma_embedder
        self.label_embedder = label_embedder
        self.predt_embedder = predt_embedder
        if psign_dim is not None:
            self.psign_embedder = Embedding(2, psign_dim)
        # to be campatible with old models
        if ctx_encoder is not None:
            self.ctx_encoder = ctx_encoder

        self.token_dropout = Dropout(p=token_dropout)
        self.lemma_dropout = Dropout(p=lemma_dropout)
        self.label_dropout = Dropout(p=label_dropout)
        self.predt_dropout = Dropout(p=predt_dropout)

        self.tau = None
        self.minimum_tau = 1e-5 
        if tau is not None:
            self.tau = torch.nn.Parameter(torch.tensor(tau), requires_grad=tunable_tau) 
        # another representation of lemmas
        if embed_lemma_ctx:
            embedder = getattr(self.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
            nlemma, dim = embedder.weight.size() 
            # similar to in/out embeddings
            self.lectx_embedder = torch.nn.Embedding(nlemma, dim) 
            torch.nn.init.xavier_normal_(self.lectx_embedder.weight)
        
        # feature space to label space
        self.suppress_nonarg = suppress_nonarg
        if not self.suppress_nonarg:
            self.nclass = self.vocab.get_vocab_size("srl_tags")
        else: # FIXME: assumming non-argument label has id 0, should ensure this in the configuration file
            self.nclass = self.vocab.get_vocab_size("srl_tags") - 1

        self.label_projection_layer = None
        if self.seq_encoder is not None:
            self.label_projection_layer = TimeDistributed(
                Linear(self.seq_encoder.get_output_dim(), self.nclass))
        
        # feature space transformation
        self.calc_seq_vector = calc_seq_vector
        self.seq_projection_dim = seq_projection_dim
        self.seq_projection_layer = None
        if self.seq_projection_dim is not None:
            if ctx_encoder is not None:
                dim = self.ctx_encoder.get_output_dim()
            else:
                dim = self.seq_encoder.get_output_dim()
            self.seq_projection_layer = Linear(dim, self.seq_projection_dim)
        
        self.ignore_span_metric = ignore_span_metric

        params = {'unlabeled_vals': self.suppress_nonarg, 'tag_namespace': "srl_tags"}
        if metric_type == 'dependency':
            self.span_metric = DependencyBasedF1Measure(vocab, **params) 
        elif metric_type == 'clustering':
            self.span_metric = ClusteringBasedF1Measure(vocab, **params) 
        else:
            params['per_predicate'] = True 
            params['is_a_sentence'] = True 
            self.span_metric = FeatureBasedF1Measure(vocab, **params) 

        self._label_smoothing = label_smoothing

    def forward(self, 
                tokens: Dict[str, torch.LongTensor],
                predicate_sign: torch.LongTensor, # 1 (is) or 0 (not) predicate position
                lemma_embedder: TextFieldEmbedder = None,
                lemmas: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        
        embedded_tokens = self.token_embedder(tokens)
        embedded_tokens = self.token_dropout(embedded_tokens)

        batch_size = embedded_tokens.size(0) 
        mask = get_text_field_mask(tokens)

        embedded_psigns = self.psign_embedder(predicate_sign)
        embedded_tokens_and_psigns = torch.cat([embedded_tokens, embedded_psigns], -1)
        seq_length = embedded_tokens_and_psigns.size(1) # (batch_size, length, dim)
        
        if not self.calc_seq_vector:
            encoded_token = self.seq_encoder(embedded_tokens_and_psigns, mask)
            embedded_seqs = None
        else: # the 'seq_encoder' can also produce a sequence embedding
            encoded_token, embedded_seqs = self.seq_encoder(embedded_tokens_and_psigns, mask)
            embedded_seqs = self.seq_projection_layer(embedded_seqs)

        # making predictions
        logits = self.label_projection_layer(encoded_token)
        logits_reshape = logits.view(-1, self.nclass)
        logits_softmax = F.softmax(logits_reshape, dim=-1).view(
                         [batch_size, seq_length, self.nclass])

        # We need to retain the mask in the output dictionary so that we can crop the 
        # sequences to remove padding when we do viterbi inference in self.decode.
        output_dict = {
            'mask': mask,
            'logits': logits, 
            'logits_softmax': logits_softmax,
            'embedded_seqs': embedded_seqs}
        return output_dict  

    def encode_skeletons(self,
                        tokens: Dict[str, torch.LongTensor],
                        predicate_sign: torch.LongTensor, # 1 (is) or 0 (not) predicate position
                        arg_mask: torch.Tensor,
                        arg_indices: torch.LongTensor,
                        max_pooling: bool = False,        # max or mean pooling for feature extraction 
                        rm_argument: bool = False,        # remove encoded arguments
                        lemma_embedder: TextFieldEmbedder = None,
                        lemmas: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = predicate_sign.size(0) 
        token_mask = get_text_field_mask(tokens)
        seq_lengths = get_lengths_from_binary_sequence_mask(token_mask).data.tolist()

        arg_token_idx = self.vocab.get_token_index("NULL_TOKEN", namespace="tokens")
        arg_lengths = get_lengths_from_binary_sequence_mask(arg_mask).data.tolist()
        
        # creat mask tensor to keep desired items
        mask = token_mask.clone().detach().fill_(1) # (bsize, len)
        mask.scatter_(1, arg_indices, 0)   # remove arguments 
        # check if there is a 0-indexed argument
        arg_indx = arg_indices + (arg_mask == 0).long()
        arg_indx = (arg_indx == 0).sum(-1) # check index 0
        pad_mask = arg_indx == 0
        mask[:, 0].masked_fill_(pad_mask, 1)
        mask = (mask == 0).unsqueeze(-1)
        
        # mask arguments
        if 'tokens' in tokens:
            all_tokens = tokens['tokens']
            tokens = all_tokens.masked_fill(mask.squeeze(-1), arg_token_idx) 
            tokens = {'tokens': tokens}
        elif 'elmo' in tokens:
            all_tokens = tokens['elmo']
            tokens = all_tokens.masked_fill(mask, 1) 
            tokens = {'elmo': tokens}

        # encoding skeletons    
        embedded_tokens = self.token_embedder(tokens)
        embedded_tokens = self.token_dropout(embedded_tokens)

        embedded_psigns = self.psign_embedder(predicate_sign)
        embedded_tokens_and_psigns = torch.cat([embedded_tokens, embedded_psigns], -1)
        seq_length = embedded_tokens_and_psigns.size(1) # (batch_size, length, dim)

        ctx_encoder = getattr(self, 'ctx_encoder', None)
        if ctx_encoder is not None:
            encoded_tokens = self.ctx_encoder(embedded_tokens_and_psigns, token_mask)
            token_dim = self.ctx_encoder.get_output_dim()
        else:
            encoded_tokens = self.seq_encoder(embedded_tokens_and_psigns, token_mask)
            token_dim = self.seq_encoder.get_output_dim()

        # contextualized arguments
        p_idx = predicate_sign.max(-1, keepdim=True)[1] # (bsize, 1)
        p_ctx = torch.gather(encoded_tokens, 1, p_idx.unsqueeze(-1).expand(-1, -1, token_dim)) 
        a_idx = arg_indices.unsqueeze(-1).expand(-1, -1, token_dim)
        a_ctx = torch.gather(encoded_tokens, 1, a_idx) 

        p_ctx = self.seq_projection_layer(p_ctx)
        a_ctx = self.seq_projection_layer(a_ctx)
        
        embedded_seqs = [] 
        dummy = -1e15 if max_pooling else 0
        if rm_argument: # and padding items
            mask = mask | (token_mask.unsqueeze(-1) == 0)
            encoded_tokens = encoded_tokens.masked_fill(mask, dummy) 
        if max_pooling:
            embedded_seqs = encoded_tokens.max(1)[0]
        else:
            divider = token_mask.sum(-1, keepdim=True).float() # (bsize, 1)
            embedded_seqs = encoded_tokens.sum(1) / divider 
        z_ctx = self.seq_projection_layer(embedded_seqs)
        return z_ctx, a_ctx, p_ctx 

    def encode_patterns(self,
                        tokens: Dict[str, torch.LongTensor],
                        predicate_sign: torch.LongTensor, # 1 (is) or 0 (not) predicate position
                        arg_mask: torch.Tensor,
                        arg_indices: torch.LongTensor,
                        lemma_embedder: TextFieldEmbedder = None,
                        lemmas: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        arg_token_idx = self.vocab.get_token_index("NULL_TOKEN", namespace="tokens")
        arg_lengths = get_lengths_from_binary_sequence_mask(arg_mask).data.tolist()

        if 'tokens' in tokens:
            all_tokens = tokens['tokens']
            for i, length in enumerate(arg_lengths):
                all_tokens[i].scatter_(0, arg_indices[i][:length], arg_token_idx)
            #all_tokens = all_tokens.scatter(1, arg_indices, arg_token_idx)
            tokens['tokens'] = all_tokens
        elif 'elmo' in tokens:
            all_tokens = tokens['elmo']
            batch_size, _, dim = all_tokens.size()
            for i, length in enumerate(arg_lengths):
                this_arg_indices = arg_indices[i][:length].unsqueeze(-1).expand(-1, dim)
                all_tokens[i].scatter_(0, this_arg_indices, 1)
            tokens['elmo'] = all_tokens

        output_dict = self.forward(tokens, predicate_sign, lemma_embedder, lemmas)
        return output_dict
    
    def gumbel_relax(self, 
                     mask: torch.Tensor,
                     logits: torch.Tensor,
                     method: str='softmax'):
        seq_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()
        if logits.dim() < 3: # fake batch size
            logits.unsqueeze(0)
        batch_size, seq_length, _ = logits.size()

        self.tau.data.clamp_(min = self.minimum_tau)
        #gumbel_hard, gumbel_soft, gumbel_soft_log = gumbel_softmax(
        #    F.log_softmax(logits.view(-1, self.nclass), dim=-1), tau=self.tau)
        if method == 'softmax':
            gumbel_hard, gumbel_soft, gumbel_soft_log = gumbel_softmax(
                logits.view(-1, self.nclass), tau=self.tau)
        elif method == 'sinkhorn':
            gumbel_hard, gumbel_soft, gumbel_soft_log = gumbel_sinkhorn(
                logits.view(-1, self.nclass), tau=self.tau)
            
        gumbel_hard = gumbel_hard.view([batch_size, seq_length, self.nclass]) 
        gumbel_soft = gumbel_soft.view([batch_size, seq_length, self.nclass]) 
        gumbel_soft_log = gumbel_soft_log.view([batch_size, seq_length, self.nclass]) 
        
        _, labels = torch.max(gumbel_hard, -1)
        for i, length in enumerate(seq_lengths):
            # 0 is in fact the 1st argument label when we suppress the non-argument label
            labels[i, length:] = 0 # FIXME: be careful of this ...
        return gumbel_hard, gumbel_soft, gumbel_soft_log, labels 

    def embed_labels(self,
                     labels: torch.Tensor,
                     labels_add_one: bool = False,
                     labels_relaxed: torch.Tensor = None):
        """ assumming there can be an additional non-argument label
            labels_add_one: increase label indices by 1
        """
        if not self.suppress_nonarg:
            label_embeddings = self.label_embedder.weight
        else: # discard the embedding of the non-argument label
            label_embeddings = self.label_embedder.weight[1:, :]

        if labels_relaxed is not None:
            encoded_labels = torch.matmul(labels_relaxed, label_embeddings) 
        elif labels is not None:  
            if labels_add_one:
                labels = labels + 1
            encoded_labels = self.label_embedder(labels)
        else:
            encoded_labels = label_embeddings
        encoded_labels = self.label_dropout(encoded_labels)
        return encoded_labels

    def encode_global_predt(self, token: str = '_', device=None):
        predt_idx = self.vocab.get_token_index("_", namespace="predicates")
        predt_idx = torch.tensor(predt_idx, device=device)
        embedded_predt = self.predt_embedder(predt_idx)
        return embedded_predt

    def encode_predt(self, predicates: torch.Tensor, predicate_sign: torch.LongTensor):
        embedded_predicates = self.predt_embedder(predicates)
        # (batch_size, length, dim) -> (batch_size, dim, length)
        embedded_predicates = torch.transpose(embedded_predicates, 1, 2)
        # (batch_size, length, 1)
        psigns = torch.unsqueeze(predicate_sign.float(), -1) 
        # (batch_size, dim, 1); select the predicate embedding
        embedded_predicates = torch.bmm(embedded_predicates, psigns)
        embedded_predicates = embedded_predicates.transpose(1, 2)
        embedded_predicates = self.predt_dropout(embedded_predicates)
        return embedded_predicates

    def encode_lemma(self, lemmas: Dict[str, torch.LongTensor], arg_indices: torch.LongTensor):
        embedded_lemmas = self.lemma_embedder(lemmas)

        if arg_indices is None:
            embedded_lemmas = self.lemma_dropout(embedded_lemmas)  
            return embedded_lemmas

        lemma_dim = embedded_lemmas.size()[-1]
        arg_indices = arg_indices.unsqueeze(-1).expand(-1, -1, lemma_dim)
        
        embedded_arg_lemmas = torch.gather(embedded_lemmas, 1, arg_indices)
        embedded_arg_lemmas = self.lemma_dropout(embedded_arg_lemmas)  
        return embedded_arg_lemmas

    def encode_lemma_ctx(self, lemmas: torch.LongTensor):
        embedded_lemmas = self.lectx_embedder(lemmas)
        embedded_lemmas = self.lemma_dropout(embedded_lemmas)  
        return embedded_lemmas

    def encode_args(self,
                    lemmas: Dict[str, torch.LongTensor],
                    predicates: torch.Tensor,
                    predicate_sign: torch.LongTensor,
                    arg_indices: torch.LongTensor,
                    embedded_seqs: torch.Tensor = None):
        embedded_predicates = self.predt_embedder(predicates)
        # (batch_size, length, dim) -> (batch_size, dim, length)
        embedded_predicates = torch.transpose(embedded_predicates, 1, 2)
        # (batch_size, length, 1)
        psigns = torch.unsqueeze(predicate_sign.float(), -1) 
        # (batch_size, dim, 1); select the predicate embedding
        embedded_predicates = torch.bmm(embedded_predicates, psigns)
         
        embedded_lemmas = self.lemma_embedder(lemmas)

        lemma_dim = embedded_lemmas.size()[-1]
        arg_indices = arg_indices.unsqueeze(-1).expand(-1, -1, lemma_dim)
        
        embedded_arg_lemmas = torch.gather(embedded_lemmas, 1, arg_indices)
        embedded_arg_lemmas = self.lemma_dropout(embedded_arg_lemmas)  

        embedded_predicates = embedded_predicates.transpose(1, 2)
        embedded_predicates = self.predt_dropout(embedded_predicates)

        embedded_nodes = torch.cat([embedded_predicates, embedded_arg_lemmas], 1)

        if embedded_seqs is not None:
            narg = arg_indices.size(1)
            embedded_seqs = embedded_seqs.unsqueeze(1).expand(-1, narg + 1, -1)
            embedded_nodes = torch.cat([embedded_nodes, embedded_seqs], -1)

        return embedded_nodes
    
    def select_args(self,
                    logits: torch.Tensor,
                    labels: torch.LongTensor,
                    lemmas: torch.LongTensor,
                    arg_indices: torch.LongTensor):
        # (batch_size, length) -> (batch_size, num_arg) 
        if labels is not None:
            labels = torch.gather(labels, 1, arg_indices)
        # -> (batch_size, num_arg)
        if lemmas is not None:
            lemmas = torch.gather(lemmas, 1, arg_indices)
        # (batch_size, num_arg, num_class) 
        if logits is not None:
            logits_indices = arg_indices.unsqueeze(-1).expand(-1, -1, self.nclass)
            logits = torch.gather(logits, 1, logits_indices) 

        if labels is not None and self.suppress_nonarg: 
            labels = labels - 1 
            labels[labels < 0] = 0 # will be masked out
        
        return logits, labels, lemmas

    def labeled_loss(self,
                     mask: torch.Tensor,
                     logits: torch.Tensor,
                     labels: torch.LongTensor,
                     average: str = 'batch'):
        
        loss_ce = sequence_cross_entropy_with_logits(
            logits, labels, mask, average=average, label_smoothing=self._label_smoothing)
        return loss_ce 

    def add_outputs(self, 
                    pivot: int,
                    mask: torch.Tensor,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    output_dict: Dict[str, torch.Tensor], 
                    arg_mask: torch.Tensor = None,
                    arg_indices: torch.Tensor = None,
                    predicates: torch.Tensor = None, 
                    metadata: List[Dict[str, Any]] = None) -> None:
        if labels is None: 
            raise ConfigurationError("Prediction loss required but gold labels `labels` is None.")
        
        if not self.ignore_span_metric:
            if predicates is not None and not isinstance(self.span_metric, DependencyBasedF1Measure):
                self.span_metric(logits[pivot:], labels[pivot:], mask=mask[pivot:], predicates=predicates[pivot:])
            else:
                self.span_metric(logits[pivot:], labels[pivot:], mask[pivot:])
            
        # if self.suppress_nonarg: # for decoding
        if arg_mask is not None and arg_indices is not None:
            output_dict['arg_masks'] = arg_mask[pivot:]
            output_dict['arg_idxes'] = arg_indices[pivot:] 

        output_dict["gold_srl"] = labels[pivot:] # decoded in `decode` for the purpose of debugging
        if metadata is not None: 
            list_lemmas, list_tokens, list_pos_tags, list_head_ids, list_predicates, list_predicate_indexes = \
                                zip(*[(x["lemmas"], x["tokens"], x["pos_tags"], x["head_ids"], \
                                    x["predicate"], x["predicate_index"], ) for x in metadata])
            output_dict["tokens"] = list(list_tokens)[pivot:]
            output_dict["lemmas"] = list(list_lemmas)[pivot:]
            output_dict["pos_tags"] = list(list_pos_tags)[pivot:]
            output_dict["head_ids"] = list(list_head_ids)[pivot:]
            output_dict["predicate"] = list(list_predicates)[pivot:]
            output_dict["predicate_index"] = list(list_predicate_indexes)[pivot:]
        return None

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_predictions = output_dict['logits_softmax']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
        # discard useless stuff in the output dict 
        # print(output_dict.keys())
        returned_dict = {"tokens": output_dict["tokens"], 
                         "lemmas": output_dict["lemmas"],
                         "pos_tags": output_dict["pos_tags"],
                         "head_ids": output_dict["head_ids"],
                         "predicate": output_dict["predicate"],
                         "predicate_index": output_dict["predicate_index"]}
        batch_size = all_predictions.size(0)
        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(batch_size)]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        
        # if self.suppress_nonarg:
        if "arg_masks" in output_dict and "arg_idxes" in output_dict: 
            arg_masks = output_dict["arg_masks"]
            arg_idxes = output_dict["arg_idxes"]
            
            arg_masks = [arg_masks[i].detach().cpu().tolist() for i in range(batch_size)]
            arg_idxes = [arg_idxes[i].detach().cpu().tolist() for i in range(batch_size)]
            
            constraints = []
            for isent, masks in enumerate(arg_masks):
                valid = []
                for iword, mask in enumerate(masks):
                    if mask == 1:
                        valid.append(arg_idxes[isent][iword])    
                    else:
                        break
                constraints.append(valid)
        else:
            constraints = [[] for _ in range(batch_size)]     
        returned_dict["arg_idxes"] = constraints
            #print(constraints)
        
        #print(self.vocab._token_to_index['srl_tags'])
        isent = 0
        for predictions, length in zip(predictions_list, sequence_lengths):
            scores, max_likelihood_sequence = torch.max(predictions[:length], 1) 
            max_likelihood_sequence = max_likelihood_sequence.tolist() 
            # FIXME pay attention to this ... 
            if not self.suppress_nonarg:
                tags = [self.vocab.get_token_from_index(x, namespace="srl_tags") for x in max_likelihood_sequence]
            else:
                tags = []
                for idx, x in enumerate(max_likelihood_sequence):
                    ilabel = 0 # non-argument label
                    if idx in constraints[isent]:
                        ilabel = x + 1 
                    if isinstance(self.span_metric, FeatureBasedF1Measure):
                        tag = str(ilabel) 
                    else:
                        tag = self.vocab.get_token_from_index(ilabel, namespace="srl_tags")
                    tags.append(tag)
            isent += 1

            all_tags.append(tags)
        returned_dict["srl_tags"] = all_tags

        # gold srl labels
        gold_srl = []
        srl_frames = output_dict["gold_srl"]
        srl_frames = [srl_frames[i].detach().cpu() for i in range(srl_frames.size(0))]
        for srls in srl_frames:
            # FIXME pay attention to this ... do not need to touch gold labels
            tags = [self.vocab.get_token_from_index(x, namespace="srl_tags") for x in srls.tolist()]

            gold_srl.append(tags)
        returned_dict["gold_srl"] = gold_srl 
        return returned_dict

    def add_argument_outputs(self, 
                    pivot: int,
                    mask: torch.Tensor,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    output_dict: Dict[str, torch.Tensor], 
                    arg_mask: torch.Tensor = None,
                    all_labels: torch.Tensor = None,
                    arg_indices: torch.Tensor = None,
                    metadata: List[Dict[str, Any]] = None) -> None:
        if labels is None: 
            raise ConfigurationError("Prediction loss required but gold labels `labels` is None.")
        
        if not self.ignore_span_metric:
            self.span_metric(logits[pivot:], labels[pivot:], mask[pivot:])
            
        if self.suppress_nonarg: # for decoding
            output_dict['arg_masks'] = arg_mask[pivot:]
            output_dict['arg_idxes'] = arg_indices[pivot:] 

        output_dict["gold_srl"] = all_labels[pivot:] # decoded in `decode` for the purpose of debugging
        if metadata is not None: 
            list_lemmas, list_tokens, list_pos_tags, list_head_ids, list_predicates, list_predicate_indexes = \
                                zip(*[(x["lemmas"], x["tokens"], x["pos_tags"], x["head_ids"], \
                                    x["predicate"], x["predicate_index"], ) for x in metadata])
            output_dict["tokens"] = list(list_tokens)[pivot:]
            output_dict["lemmas"] = list(list_lemmas)[pivot:]
            output_dict["pos_tags"] = list(list_pos_tags)[pivot:]
            output_dict["head_ids"] = list(list_head_ids)[pivot:]
            output_dict["predicate"] = list(list_predicates)[pivot:]
            output_dict["predicate_index"] = list(list_predicate_indexes)[pivot:]
        return None

    def decode_arguments(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'logits' in output_dict:
            all_predictions = output_dict['logits']
        elif 'logits_softmax' in output_dict:
            all_predictions = output_dict['logits_softmax']
        else:
            raise ConfigurationError("unavailable logits for decoding predictions")
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
        # discard useless stuff in the output dict 
        returned_dict = {"tokens": output_dict["tokens"], 
                         "lemmas": output_dict["lemmas"],
                         "pos_tags": output_dict["pos_tags"],
                         "head_ids": output_dict["head_ids"],
                         "predicate": output_dict["predicate"],
                         "predicate_index": output_dict["predicate_index"]}
        batch_size = all_predictions.size(0)
        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(batch_size)]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        
        if self.suppress_nonarg:
            arg_masks = output_dict["arg_masks"]
            arg_idxes = output_dict["arg_idxes"]
            
            arg_masks = [arg_masks[i].detach().cpu().tolist() for i in range(batch_size)]
            arg_idxes = [arg_idxes[i].detach().cpu().tolist() for i in range(batch_size)]
            
            constraints = []
            for isent, masks in enumerate(arg_masks):
                valid = []
                for iword, mask in enumerate(masks):
                    if mask == 1:
                        valid.append(arg_idxes[isent][iword])    
                    else:
                        break
                constraints.append(valid)
        else:
            constraints = [[] for _ in range(batch_size)]     
            #print(constraints)
        returned_dict["arg_idxes"] = constraints
        
        #print(self.vocab._token_to_index['srl_tags'])
        isent = 0
        for predictions, length in zip(predictions_list, sequence_lengths):
            scores, max_likelihood_sequence = torch.max(predictions[:length], 1) 
            max_likelihood_sequence = max_likelihood_sequence.tolist() 
            # FIXME pay attention to this ... 
            if not self.suppress_nonarg:
                tags = [self.vocab.get_token_from_index(x, namespace="srl_tags") for x in max_likelihood_sequence]
            else:
                ntoken = len(output_dict["tokens"][isent])
                tags = [self.vocab.get_token_from_index(0, namespace="srl_tags") for _ in range(ntoken)]
                for iarg, idx in enumerate(constraints[isent]):
                    ilabel = max_likelihood_sequence[iarg] + 1
                    tag = self.vocab.get_token_from_index(ilabel, namespace="srl_tags")
                    tags[idx] = tag
            isent += 1

            all_tags.append(tags)
        returned_dict["srl_tags"] = all_tags

        # gold srl labels
        gold_srl = []
        srl_frames = output_dict["gold_srl"]
        srl_frames = [srl_frames[i].detach().cpu() for i in range(srl_frames.size(0))]
        for srls in srl_frames:
            # FIXME pay attention to this ... do not need to touch gold labels
            tags = [self.vocab.get_token_from_index(x, namespace="srl_tags") for x in srls.tolist()]

            gold_srl.append(tags)
        returned_dict["gold_srl"] = gold_srl 
        return returned_dict

    def add_arg_outputs(self, 
                    pivot: int,
                    mask: torch.Tensor,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    output_dict: Dict[str, torch.Tensor], 
                    arg_mask: torch.Tensor = None,
                    all_labels: torch.Tensor = None,
                    arg_indices: torch.Tensor = None,
                    metadata: List[Dict[str, Any]] = None) -> None:
        if labels is None: 
            raise ConfigurationError("Prediction loss required but gold labels `labels` is None.")
        
        if not self.ignore_span_metric:
            if not hasattr(self, 'arg_span_metric'):
                self.arg_span_metric = DependencyBasedF1Measure(self.vocab, unlabeled_vals=True, tag_namespace="lemmas")
            self.arg_span_metric.arg_metric(logits[pivot:], labels[pivot:], mask[pivot:])
            
        if self.suppress_nonarg: # for decoding
            output_dict['arg_masks'] = arg_mask[pivot:]
            output_dict['arg_idxes'] = arg_indices[pivot:] 

        output_dict["gold_srl"] = all_labels[pivot:] # decoded in `decode` for the purpose of debugging
        if metadata is not None: 
            list_lemmas, list_tokens, list_pos_tags, list_head_ids, list_predicates, list_predicate_indexes = \
                                zip(*[(x["lemmas"], x["tokens"], x["pos_tags"], x["head_ids"], \
                                    x["predicate"], x["predicate_index"], ) for x in metadata])
            output_dict["tokens"] = list(list_tokens)[pivot:]
            output_dict["lemmas"] = list(list_lemmas)[pivot:]
            output_dict["pos_tags"] = list(list_pos_tags)[pivot:]
            output_dict["head_ids"] = list(list_head_ids)[pivot:]
            output_dict["predicate"] = list(list_predicates)[pivot:]
            output_dict["predicate_index"] = list(list_predicate_indexes)[pivot:]
        return None

    def decode_args(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        namespace = 'lemmas'
        if 'logits' in output_dict:
            all_predictions = output_dict['logits']
        elif 'logits_softmax' in output_dict:
            all_predictions = output_dict['logits_softmax']
        else:
            raise ConfigurationError("unavailable logits for decoding predictions")
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
        # discard useless stuff in the output dict 
        returned_dict = {"lemmas": output_dict["tokens"], 
                         "tokens": output_dict["lemmas"],
                         "pos_tags": output_dict["pos_tags"],
                         "head_ids": output_dict["head_ids"],
                         "predicate": output_dict["predicate"],
                         "predicate_index": output_dict["predicate_index"]}
        batch_size = all_predictions.size(0)
        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(batch_size)]
        else:
            predictions_list = [all_predictions]
        all_tags = []

        if self.suppress_nonarg:
            arg_masks = output_dict["arg_masks"]
            arg_idxes = output_dict["arg_idxes"]
            
            arg_masks = [arg_masks[i].detach().cpu().tolist() for i in range(batch_size)]
            arg_idxes = [arg_idxes[i].detach().cpu().tolist() for i in range(batch_size)]
            
            constraints = []
            for isent, masks in enumerate(arg_masks):
                valid = []
                for iword, mask in enumerate(masks):
                    if mask == 1:
                        valid.append(arg_idxes[isent][iword])    
                    else:
                        break
                constraints.append(valid)
        else:
            constraints = [[] for _ in range(batch_size)]     
            #print(constraints)
        returned_dict["arg_idxes"] = constraints

        #print(self.vocab._token_to_index['srl_tags'])
        isent = 0
        for predictions, length in zip(predictions_list, sequence_lengths):
            scores, max_likelihood_sequence = torch.max(predictions[:length], 1) 
            max_likelihood_sequence = max_likelihood_sequence.tolist() 
            # FIXME pay attention to this ... 
            if not self.suppress_nonarg:
                tags = [self.vocab.get_token_from_index(x, namespace=namespace) for x in max_likelihood_sequence]
            else:
                ntoken = len(output_dict["tokens"][isent])
                #tags = [self.vocab.get_token_from_index(0, namespace=namespace) for _ in range(ntoken)]
                tags = ['O' for _ in range(ntoken)]
                for iarg, idx in enumerate(constraints[isent]):
                    ilabel = max_likelihood_sequence[iarg] 
                    tag = self.vocab.get_token_from_index(ilabel, namespace=namespace)
                    tags[idx] = tag
            isent += 1

            all_tags.append(tags)
        returned_dict["srl_tags"] = all_tags

        # gold srl labels
        gold_srl = []
        srl_frames = output_dict["gold_srl"]
        srl_frames = [srl_frames[i].detach().cpu() for i in range(srl_frames.size(0))]
        for srls in srl_frames:
            # FIXME pay attention to this ... do not need to touch gold labels
            tags = [self.vocab.get_token_from_index(x, namespace="srl_tags") for x in srls.tolist()]

            gold_srl.append(tags)
        returned_dict["gold_srl"] = gold_srl 
        return returned_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            return {} # Return an empty dictionary if ignoring the span metric
        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            if isinstance(self.span_metric, FeatureBasedF1Measure):
                return {x: y for x, y in metric_dict.items() if "f1" in x or "co" in x or "pu" in x}
            else:
                return {x: y for x, y in metric_dict.items() if "f1-measure-overall" in x}

