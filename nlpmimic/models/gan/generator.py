from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Linear, Dropout

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import Seq2SeqEncoder
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

from nlpmimic.nn.util import gumbel_softmax
from nlpmimic.training.metrics import DependencyBasedF1Measure

@Model.register("sri_gan_gen")
class SrlGanGenerator(Model):
    def __init__(self, vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 seq_encoder: Seq2SeqEncoder,
                 psign_dim: int, # psign: predicate sign bit (0/1) 
                 tau: float = None,
                 tunable_tau: bool = False,
                 suppress_nonarg: bool = False,
                 seq_projection_dim: int = None, 
                 embedding_dropout: float = 0.,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlGanGenerator, self).__init__(vocab, regularizer)
        
        self.seq_encoder = seq_encoder
        self.token_embedder = token_embedder
        self.psign_embedder = Embedding(2, psign_dim)
        self.embedding_dropout = Dropout(p=embedding_dropout)

        self.tau = None
        self.minimum_tau = 1e-5 
        if tau is not None:
            self.tau = torch.nn.Parameter(torch.tensor(tau), requires_grad=tunable_tau) 
        
        # feature space to label space
        self.suppress_nonarg = suppress_nonarg
        if not self.suppress_nonarg:
            self.nclass = self.vocab.get_vocab_size("srl_tags")
        else: # FIXME: assumming non-argument label has id 0, should ensure this in the configuration file
            self.nclass = self.vocab.get_vocab_size("srl_tags") - 1
        
        self.label_projection_layer = TimeDistributed(
            Linear(self.seq_encoder.get_output_dim(), self.nclass))
        
        # feature space transformation
        self.seq_projection_dim = seq_projection_dim
        self.seq_projection_layer = None
        if self.seq_projection_dim is not None:
            self.seq_projection_layer = \
                Linear(self.seq_encoder.get_output_dim(), self.seq_projection_dim)
        
        self.ignore_span_metric = ignore_span_metric
        self.span_metric = DependencyBasedF1Measure(vocab, 
                                                    unlabeled_vals = self.suppress_nonarg,
                                                    tag_namespace="srl_tags")
        self._label_smoothing = label_smoothing

    def forward(self, 
                tokens: Dict[str, torch.LongTensor],
                predicate_sign: torch.LongTensor, # 1 (is) or 0 (not) predicate position
                lemma_embedder: TextFieldEmbedder = None,
                lemmas: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        
        embedded_tokens = self.token_embedder(tokens)
        embedded_tokens = self.embedding_dropout(embedded_tokens)

        batch_size = embedded_tokens.size(0) 
        mask = get_text_field_mask(tokens)

        embedded_psigns = self.psign_embedder(predicate_sign)
        embedded_tokens_and_psigns = torch.cat([embedded_tokens, embedded_psigns], -1)
        seq_length = embedded_tokens_and_psigns.size(1) # (batch_size, length, dim)
        
        if self.seq_projection_layer is None:
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
    
    def gumbel_relax(self, 
                     mask: torch.Tensor,
                     logits: torch.Tensor):
        seq_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()
        if logits.dim() < 3: # fake batch size
            logits.unsqueeze(0)
        batch_size, seq_length, _ = logits.size()

        self.tau.data.clamp_(min = self.minimum_tau)
        #gumbel_hard, gumbel_soft, gumbel_soft_log = gumbel_softmax(
        #    F.log_softmax(logits.view(-1, self.nclass), dim=-1), tau=self.tau)
        gumbel_hard, gumbel_soft, gumbel_soft_log = gumbel_softmax(
            logits.view(-1, self.nclass), tau=self.tau)
            
        gumbel_hard = gumbel_hard.view([batch_size, seq_length, self.nclass]) 
        gumbel_soft = gumbel_soft.view([batch_size, seq_length, self.nclass]) 
        gumbel_soft_log = gumbel_soft_log.view([batch_size, seq_length, self.nclass]) 
        
        _, labels = torch.max(gumbel_hard, -1)
        for i, length in enumerate(seq_lengths):
            # 0 is in fact the 1st argument label when we suppress the non-argument label
            labels[i, length:] = 0 # FIXME: be careful of this ...
        return gumbel_hard, gumbel_soft, gumbel_soft_log, labels 

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

    def kldivergence(self, pivot: int, 
                     mask: torch.Tensor, 
                     probabilities: torch.Tensor, 
                     loss_type: str = 'unscale_kl'):
        kl_loss = torch.log(probabilities) * probabilities
        if loss_type == 'reverse_kl':
            # (k/n) ln ((k/n)/(1/n))
            divider = 1. / torch.sum(mask, -1).float() 
            kl_loss = kl_loss * divider.unsqueeze(-1).expand(-1, self.nclass)
        elif loss_type == 'unscale_kl':
            pass # k ln ((k/n)/(1/n))  
        else:
            pass # ...
        kl_loss = kl_loss[:, pivot:] # discard the loss for empty labels if pivot is 1

        loss = torch.mean(torch.sum(kl_loss, 1)) # loss at sentence level
        #loss = torch.mean(kl_loss[kl_loss > 0]) # loss at token level 
        return loss 

    def add_outputs(self, 
                    pivot: int,
                    mask: torch.Tensor,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    output_dict: Dict[str, torch.Tensor], 
                    arg_mask: torch.Tensor = None,
                    arg_indices: torch.Tensor = None,
                    metadata: List[Dict[str, Any]] = None) -> None:
        if labels is None: 
            raise ConfigurationError("Prediction loss required but gold labels `labels` is None.")
        
        if not self.ignore_span_metric:
            self.span_metric(logits[pivot:], labels[pivot:], mask[pivot:])
            
        if self.suppress_nonarg: # for decoding
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

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            return {} # Return an empty dictionary if ignoring the span metric
        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "f1-measure-overall" in x}

