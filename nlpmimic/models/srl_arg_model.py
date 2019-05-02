from typing import Dict, List, TextIO, Optional, Any

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode

from nlpmimic.training.metrics import DependencyBasedF1Measure


@Model.register("srl_arg")
class ArgSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 token_embedder: TextFieldEmbedder,
                 lemma_embedder: TextFieldEmbedder,
                 label_embedder: Embedding,
                 predicate_embedder: Embedding,
                 seq_encoder: Seq2SeqEncoder,
                 srl_encoder: Seq2VecEncoder,
                 binary_feature_dim: int,
                 temperature: float = 1.,
                 fixed_temperature: bool = False,
                 mask_empty_labels: bool = True,
                 embedding_dropout: float = 0.,
                 use_label_indicator: bool = False,
                 optimize_lemma_embedding: bool = False,
                 zero_null_lemma_embedding: bool = False, 
                 label_loss_type: str = 'reverse_kl',
                 suppress_nonarg: bool = False,
                 regularized_batch: bool = False,
                 regularized_labels: List[str] = None,
                 regularized_nonarg: bool = False,
                 seq_projection_dim: int = None, 
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False) -> None:
        super(ArgSemanticRoleLabeler, self).__init__(vocab, regularizer)

        self.minimum = 1e-20
        self.minimum_temperature = 1e-5 
        self.mask_empty_labels = mask_empty_labels
        self.regularized_batch = regularized_batch
        
        self.seq_encoder = seq_encoder
        self.srl_encoder = srl_encoder

        self.token_embedder = token_embedder
        self.lemma_embedder = lemma_embedder
        self.label_embedder = label_embedder
        self.predicate_embedder = predicate_embedder

        self.suppress_nonarg = suppress_nonarg
        if not self.suppress_nonarg:
            self.num_classes = self.vocab.get_vocab_size("srl_tags")
        else: # FIXME: assumming non-argument label has id 0, should ensure this in the configuration file
            self.num_classes = self.vocab.get_vocab_size("srl_tags") - 1

        self.null_lemma_idx = self.vocab.get_token_index("NULL_LEMMA", namespace="lemmas")
        
        if zero_null_lemma_embedding:
            # this will make `lemma_embedder.token_embedder_lemmas.weight` become an non-leaf node
            # when the weight is trainable, and further result in errors in optimizer cause the
            # optimizer expects only leaf nodes.
            embedder = getattr(self.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
            embedder.weight[self.null_lemma_idx, :].fill_(0.)
            #embedder.weight[self.null_lemma_idx, :] = 0.
            if optimize_lemma_embedding:
                embedder.weight.requires_grad_()
            """ 
            idx = {'lemmas': torch.tensor(self.null_lemma_idx)}
            vec = self.lemma_embedder(idx)
            print(embedder.weight, embedder.weight.size(), embedder.weight.requires_grad)
            print(vec)
            """
        self.use_label_indicator = use_label_indicator
        if self.use_label_indicator:
            label_indicator = torch.zeros([self.num_classes, self.num_classes], dtype=torch.float)
            for idx in range(self.num_classes):
                label_indicator[idx, idx] = 1.
            self.label_indicator = torch.nn.Embedding.from_pretrained(label_indicator, freeze=True)
        
        self.label_loss_type = label_loss_type
        self.regularized_labels = regularized_labels
        self.regularized_nonarg = regularized_nonarg
        if self.regularized_labels:
            """ # regularize specified labels
            label_indexes = [self.vocab.get_token_index(label, namespace="srl_tags") 
                            for label in self.regularized_labels] 
            label_selector = torch.zeros(self.num_classes)
            label_selector[label_indexes] = 1. 
            self.label_selector = torch.nn.Parameter(label_selector, requires_grad=False)
            """
            pass
        self.seq_projection_layer = None
        seq_vector_dim = seq_projection_dim
        if seq_projection_dim is not None:
            self.seq_projection_layer = Linear(self.seq_encoder.get_output_dim(), seq_projection_dim)
        else:
            seq_vector_dim = 0 

        self.use_graph_srl_encoder = self.srl_encoder.signature == 'graph' 
        if self.use_graph_srl_encoder:
            if self.lemma_embedder.get_output_dim() != self.predicate_embedder.get_output_dim():
                raise ConfigurationError("Embedding dimensions of lemmas and predicates are not equal")
            srl_encoder.add_gcn_parameters(
                self.num_classes, # FIXME: be careful when discarding the non-argument label 
                seq_vector_dim + lemma_embedder.get_output_dim()) 
        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        # FIXME: has been rewritten for discarding the non-argument label
        self.span_metric = DependencyBasedF1Measure(vocab, 
                                                    tag_namespace="srl_tags", 
                                                    unlabeled_vals = self.suppress_nonarg,
                                                    ignore_classes=["V"])
        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=fixed_temperature) 

        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.seq_encoder.get_output_dim(),
                                                           self.num_classes))

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric

        check_dimensions_match(token_embedder.get_output_dim() + binary_feature_dim,
                               seq_encoder.get_input_dim(),
                               "text embedding dim + verb indicator embedding dim",
                               "seq_encoder input dim")
        initializer(self)
    
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                lemmas: Dict[str, torch.LongTensor],
                predicate_indicators: torch.LongTensor,
                predicates: torch.LongTensor,
                argument_indices: torch.LongTensor = None,
                predicate_index: torch.LongTensor = None,
                argument_mask: torch.LongTensor = None,
                retrive_generator_loss: bool = False,
                reconstruction_loss: bool = False,
                only_reconstruction: bool = False,
                srl_frames: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        The first 1/2 batch has verbal predicates, and the following 1/2 batch has nominal predicates.
        For the verbal part, we only need to encode their predicates and  gold/predicted semantic role 
        labels into a feature vector; for the nominal part, we always need to predict semantic role 
        labels, and then encode them and predicates into a feature vectors. The sequence encoder used 
        for verbal and nominal parts are the same one, which is a discirminator implemented as a convolutional 
        neural network.
        """
        embedded_token_input = self.embedding_dropout(self.token_embedder(tokens))
        batch_size, length, _ = embedded_token_input.size() 
        mask = get_text_field_mask(tokens)
        
        if self.training: # FIX ME: avoid unnecessary embedding look up when retrieving generator loss
            embedded_predicates = self.predicate_embedder(predicates)
            # (batch_size, length, dim) -> (batch_size, dim, length)
            embedded_predicates = torch.transpose(embedded_predicates, 1, 2)
            # (batch_size, length, 1)
            pis = torch.unsqueeze(predicate_indicators.float(), -1) 
            # (batch_size, dim, 1); select the predicate embedding
            embedded_predicates = torch.bmm(embedded_predicates, pis)
            # apply dropout to predicate embeddings
            embedded_predicates = self.embedding_dropout(embedded_predicates)
            
            batch_size = batch_size // 2
            # nominal part
            embedded_tokens = embedded_token_input[batch_size:, :]
            used_mask = mask[batch_size:, :]
            used_pi = predicate_indicators[batch_size:, :]
        else:
            embedded_tokens = embedded_token_input
            used_mask = mask
            used_pi = predicate_indicators


        if self.seq_projection_layer is not None:
            this_mask = mask               # all mask
            this_pi = predicate_indicators # all indicators
            this_embedded_tokens = embedded_token_input
            this_embedded_pi = self.binary_feature_embedding(this_pi.long())

            embedded_token_with_pi = torch.cat([this_embedded_tokens, this_embedded_pi], -1)
            _, sequence_length, _ = embedded_token_with_pi.size()
            encoded_token, embedded_sent = self.seq_encoder(embedded_token_with_pi, this_mask)
            embedded_sent = self.seq_projection_layer(embedded_sent)
            if self.training:
                encoded_token = encoded_token[batch_size:]
        else:
            embedded_sent = None
            # Predict srl labels using the srler that needs to be trained.
            # Concatenate the verb feature onto the embedded text. This now
            # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
            embedded_pi = self.binary_feature_embedding(used_pi.long())
            
            embedded_token_with_pi = torch.cat([embedded_tokens, embedded_pi], -1)
            _, sequence_length, _ = embedded_token_with_pi.size()
            encoded_token = self.seq_encoder(embedded_token_with_pi, used_mask)
            

        logits = self.tag_projection_layer(encoded_token)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                                        [batch_size, sequence_length, self.num_classes])
        # We need to retain the mask in the output dictionary so that we can crop the 
        # sequences to remove padding when we do viterbi inference in self.decode.
        output_dict = {"logits": logits, 
                       "class_probabilities": class_probabilities,
                       "mask": used_mask}
        
        if self.training:
            if self.use_graph_srl_encoder:
                gan_loss_input = self.create_discriminator_graph_input(
                                                        length,
                                                        batch_size,
                                                        mask,
                                                        used_mask,
                                                        srl_frames,
                                                        argument_mask,
                                                        argument_indices,
                                                        class_probabilities, 
                                                        embedded_predicates,
                                                        lemmas,
                                                        output_dict,
                                                        retrive_generator_loss,
                                                        embedded_sent)
                self.srl_encoder(gan_loss_input, argument_mask, output_dict, retrive_generator_loss, only_reconstruction)
            else:
                gan_loss_input, full_label_masks = self.create_discriminator_input(
                                                        length,
                                                        batch_size,
                                                        mask,
                                                        used_mask,
                                                        srl_frames,
                                                        class_probabilities, 
                                                        embedded_predicates,
                                                        lemmas,
                                                        output_dict)
                self.srl_encoder(gan_loss_input, full_label_masks, output_dict, retrive_generator_loss, only_reconstruction)
            # assumming we can access gold labels for nominal part
            self.add_outputs(batch_size, srl_frames, logits, used_mask, reconstruction_loss, output_dict, metadata)
        else: # not training
            if self.suppress_nonarg: 
                output_dict['arg_masks'] = argument_mask
                output_dict['arg_idxes'] = argument_indices 
            self.add_outputs(0, srl_frames, logits, used_mask, reconstruction_loss, output_dict, metadata)
        return output_dict
    
    def create_discriminator_graph_input(self,
                                         length: int,
                                         batch_size: int,
                                         mask: torch.Tensor,
                                         used_mask: torch.Tensor,
                                         srl_frames: torch.Tensor,
                                         argument_mask: torch.Tensor,
                                         argument_indices: torch.Tensor,
                                         class_probabilities: torch.Tensor, 
                                         embedded_predicates: torch.Tensor,
                                         lemmas: Dict[str, torch.LongTensor],
                                         output_dict: Dict[str, torch.Tensor],
                                         retrive_generator_loss: bool=False,
                                         embedded_sent: torch.Tensor = None):
        class_probs, noun_labels, _ = \
            self._argmax_logits(class_probabilities, used_mask, argument_indices[batch_size:, :])
        embedded_lemma_input = self.embedding_dropout(self.lemma_embedder(lemmas))
        
        lemma_dim = embedded_lemma_input.size()[-1]
        indices = argument_indices.unsqueeze(-1).expand(-1, -1, lemma_dim)
        
        embedded_nodes = torch.gather(embedded_lemma_input, 1, indices)
        embedded_preds = embedded_predicates.transpose(1, 2)
        embedded_nodes = torch.cat([embedded_preds, embedded_nodes], 1)

        if embedded_sent is not None:
            narg = argument_mask.size(1)
            embedded_sents = embedded_sent.unsqueeze(1).expand(-1, narg + 1, -1)
            embedded_nodes = torch.cat([embedded_nodes, embedded_sents], -1)

        embedded_verb_nodes = embedded_nodes[:batch_size, :]
        embedded_noun_nodes = embedded_nodes[batch_size:, :]
        
        edge_types = torch.gather(torch.cat([srl_frames[:batch_size, :], noun_labels], 0), 1, argument_indices)
        edge_types = edge_types * argument_mask

        # FIXME: edge type 0 indicates the non-argument label, but should indicate 
        # the 1st argument label when the non-argument label is discarded
        verb_edge_types = edge_types[:batch_size, :] 
        noun_edge_types = edge_types[batch_size:, :]
        if self.suppress_nonarg: 
            # verbal argument label indexes >= 0; 0 will refer to the 1st argument label
            # but 0 is as well used as padding number; FIXME: be careful of this ...
            # we will rely on argument masks to remove padding numbers
            verb_edge_types = verb_edge_types - 1 
            verb_edge_types[verb_edge_types < 0] = 0
            # for decoding
            output_dict['arg_masks'] = argument_mask[batch_size:]
            output_dict['arg_idxes'] = argument_indices[batch_size:] 

        onehot_indices = argument_indices.unsqueeze(-1).expand(-1, -1, self.num_classes)
        noun_edge_onehots = torch.gather(class_probs, 1, onehot_indices[batch_size:, :])
        noun_edge_softmax = torch.gather(class_probabilities, 1, onehot_indices[batch_size:, :])
        
        if self.regularized_batch and retrive_generator_loss: # diversity regularization
            output_dict["bp_loss"] = self._regularize_batch(length, batch_size, argument_mask, 
                noun_edge_onehots, edge_types[:batch_size, :]) 
        else:
            output_dict["bp_loss"] = None 
        if self.regularized_labels:
            output_dict["kl_loss"] = self._regularize(length, batch_size, used_mask, class_probs,
                regularized_nonarg=self.regularized_nonarg, 
                argument_indices=argument_indices[batch_size:, :],
                argument_mask=argument_mask[batch_size:])
        else:
            output_dict["kl_loss"] = None

        graph_input = {'v_embedded_nodes': embedded_verb_nodes,
                       'v_edge_types': verb_edge_types,
                       'v_edge_type_onehots': None, 
                       'n_embedded_nodes': embedded_noun_nodes,
                       'n_edge_types': noun_edge_types,
                       'n_edge_type_onehots': noun_edge_onehots,
                       'n_edge_type_softmax': noun_edge_softmax}
        return graph_input 

    def create_discriminator_input(self,
                                   length: int,
                                   batch_size: int,
                                   mask: torch.Tensor,
                                   used_mask: torch.Tensor,
                                   srl_frames: torch.Tensor,
                                   class_probabilities: torch.Tensor, 
                                   embedded_predicates: torch.Tensor,
                                   lemmas: Dict[str, torch.LongTensor],
                                   output_dict: Dict[str, torch.Tensor]):
        class_probs, noun_labels, embedded_noun_labels = \
            self._argmax_logits(class_probabilities, used_mask)
        if self.regularized_labels:
            output_dict["kl_loss"] = self._regularize(length, batch_size, used_mask, class_probs, False)
        else:
            output_dict["kl_loss"] = None
    
        predicted_labels = torch.cat([srl_frames[:batch_size, :], noun_labels], 0)
        
        if self.mask_empty_labels:
            # srl label masks
            full_label_masks = mask.clone() 
            # mask out empty labels 
            full_label_masks[predicted_labels == 0] = 0
        else:
            full_label_masks = mask
            # use a special lemma embedding for empty labels
            lemmas = lemmas['lemmas']
            lemmas = lemmas * (predicted_labels != 0).long()
            lemmas = lemmas + (predicted_labels == 0).long() * self.null_lemma_idx
            lemmas = {'lemmas': lemmas}
        
        embedded_lemma_input = self.embedding_dropout(self.lemma_embedder(lemmas))
        embedded_verb_lemmas = embedded_lemma_input[:batch_size, :] 
        embedded_noun_lemmas = embedded_lemma_input[batch_size:, :] 
            
        #embedded_noun_labels = self.embedding_dropout(self.label_embedder(noun_labels))
        embedded_verb_labels = self.embedding_dropout(self.label_embedder(srl_frames[:batch_size, :]))
        if self.use_label_indicator: # add one-hot vector features, not work, false by default
            verb_label_indicator = self.label_indicator(srl_frames[:batch_size, :]) 
            embedded_verb_labels = torch.cat([embedded_verb_labels, verb_label_indicator], -1)
            noun_label_indicator = self.label_indicator(noun_labels)    
            embedded_noun_labels = torch.cat([embedded_noun_labels, noun_label_indicator], -1)

        # embedded predicates 
        # (batch_size, length, dim); `expand` avoid data copy, unlike `repeat`
        embedded_predicate_input = embedded_predicates.expand(-1, -1, length).transpose(1, 2)
        embedded_verb_predicates = embedded_predicate_input[:batch_size, :] 
        embedded_noun_predicates = embedded_predicate_input[batch_size:, :] 

        gan_loss_input = {'v_embedded_tokens': None,
                          'v_embedded_lemmas': embedded_verb_lemmas,
                          'v_embedded_predicates': embedded_verb_predicates,
                          'v_embedded_labels': embedded_verb_labels,
                          'n_embedded_tokens': None,
                          'n_embedded_lemmas': embedded_noun_lemmas,
                          'n_embedded_predicates': embedded_noun_predicates,
                          'n_embedded_labels': embedded_noun_labels}
        return gan_loss_input, full_label_masks

    def _regularize(self,
                    length: int,
                    batch_size: int,
                    mask: torch.Tensor,
                    class_probs: torch.Tensor,
                    regularized_nonarg: bool=False,
                    argument_indices: torch.Tensor=None,
                    argument_mask: torch.Tensor=None):
        if self.suppress_nonarg:
            onehot_indices = argument_indices.unsqueeze(-1).expand(-1, -1, self.num_classes)
            noun_edge_onehots = torch.gather(class_probs, 1, onehot_indices)
            noun_edge_onehots = noun_edge_onehots * argument_mask.unsqueeze(-1).float()

            batch_probs = torch.sum(noun_edge_onehots, 1)
            batch_probs.clamp_(min = 1.) # a trick to avoid nan = log(0)

            kl_loss = torch.log(batch_probs) * batch_probs
            
            if self.label_loss_type == 'reverse_kl':
                # (k/n) ln ((k/n)/(1/n))
                divider = 1. / torch.sum(mask, -1).float() 
                kl_loss = kl_loss * divider.unsqueeze(-1).expand(-1, self.num_classes)
            elif self.label_loss_type == 'unscale_kl':
                pass # k ln ((k/n)/(1/n))  
            else:
                pass # ...
            loss = torch.mean(torch.sum(kl_loss, 1)) # loss at sentence level
            return loss

        # if we only want to regularize non-argument labels
        # we need to choose the predicted non-argument labels
        if regularized_nonarg:
            # obtain non-argument mask, is there any other more efficient method?
            non_arg_mask = torch.zeros((batch_size, length), device=mask.device)
            non_arg_mask.scatter_(1, argument_indices, 1)
            pad_mask = argument_indices[:, 0] != 0
            non_arg_mask[:, 0].masked_fill_(pad_mask, 0)
            non_arg_mask = non_arg_mask == 0
            # select predicted non-argument labels
            class_probs = class_probs * non_arg_mask.unsqueeze(-1).float() 

        class_probs = class_probs * mask.unsqueeze(-1).float() 
        batch_probs = torch.sum(class_probs, 1)
        
        batch_probs.clamp_(min = 1.) # a trick to avoid nan = log(0)

        kl_loss = torch.log(batch_probs) * batch_probs
        
        if self.label_loss_type == 'reverse_kl':
            # (k/n) ln ((k/n)/(1/n))
            divider = 1. / torch.sum(mask, -1).float() 
            kl_loss = kl_loss * divider.unsqueeze(-1).expand(-1, self.num_classes)
        elif self.label_loss_type == 'unscale_kl':
            pass # k ln ((k/n)/(1/n))  
        else:
            pass # ...
        kl_loss = kl_loss[:, 1:] # discard the loss for empty labels

        loss = torch.mean(torch.sum(kl_loss, 1)) # loss at sentence level
        #loss = torch.mean(kl_loss[kl_loss > 0]) # loss at token level 
        return loss

    def _regularize_batch(self, 
                          length: int,
                          batch_size: int,
                          argument_mask: torch.Tensor,
                          noun_class_probs: torch.Tensor, 
                          verb_labels: torch.Tensor):
        verb_mask = argument_mask[:batch_size, :]
        noun_mask = argument_mask[batch_size:, :]
        #print(noun_class_probs, noun_class_probs.size())
        
        class_probs = noun_class_probs * noun_mask.unsqueeze(-1).float() 
        #print(class_probs, class_probs.size())
        batch_probs = torch.sum(class_probs, 1)
        # torch.histc() has been supported on GPU for the latest version of Pytorch   
        verb_labels = verb_labels * verb_mask
        verb_labels_cpu = verb_labels.detach().cpu().float()
        kl_gold = torch.histc(verb_labels_cpu, bins=self.num_classes + 1, min=0, max=self.num_classes + 1)
        kl_gold = kl_gold.to(noun_class_probs.device) + 1 # gpu
        kl_gold = kl_gold[1:] / torch.sum(kl_gold[1:])
        #print(kl_gold, kl_gold.size())
        """ 
        gold_labels = torch.zeros(self.num_classes, length, device=all_mask.device)
        ones_labels = torch.ones(verb_labels.size(), device=all_mask.device)
        gold_labels.scatter_add_(0, verb_labels, ones_labels)
         
        # ignore non-argument label
        kl_gold = torch.sum(gold_labels, -1) + 1.
        print(kl_gold, kl_gold.size())
        kl_gold = kl_gold[1:] / torch.sum(kl_gold[1:])
        print(kl_gold, kl_gold.size())
        """ 
        kl_pseu = torch.sum(batch_probs, 0) + 1.
        #print(kl_pseu, kl_pseu.size())
        kl_pseu = kl_pseu / torch.sum(kl_pseu) 
        #kl_pseu = kl_pseu[1:] / torch.sum(kl_pseu[1:]) 
        #print(kl_pseu, kl_pseu.size())
        
        kl_loss = kl_gold * torch.log(kl_gold / kl_pseu)
        #print(kl_loss, kl_loss.size())
        kl_loss = torch.sum(kl_loss)
        #print(kl_loss) 
        return kl_loss

    def _argmax_logits(self, 
                       logits: torch.Tensor, 
                       mask: torch.Tensor, 
                       argument_indices: torch.Tensor = None) -> torch.Tensor:
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()
        if logits.dim() < 3: # fake batch size
            logits.unsqueeze(0)
        batch_size, sequence_length, _ = logits.size()

        self.temperature.data.clamp_(min = self.minimum_temperature)

        class_probs = F.gumbel_softmax(
            torch.log(logits.view(-1, self.num_classes) + self.minimum), 
            tau = self.temperature,
            hard = True).view([batch_size, sequence_length, self.num_classes]) 
        # FIXME 
        if not self.suppress_nonarg:
            embeddings = self.label_embedder.weight
        else: # discard the embedding of the non-argument label
            embeddings = self.label_embedder.weight[1:, :]

        embedded_labels =self.embedding_dropout(
            torch.matmul(class_probs, embeddings)) 
        _, max_likelihood_sequence = torch.max(class_probs, -1)
        for i, length in enumerate(sequence_lengths):
            # 0 is in fact the 1st argument label when we suppress the non-argument label
            # FIXME: be careful of this ...
            max_likelihood_sequence[i, length:] = 0
        return class_probs, max_likelihood_sequence, embedded_labels 

    def add_outputs(self, 
                    pivot: int,
                    srl_frames: torch.Tensor,
                    logits: torch.Tensor,
                    mask: torch.Tensor,
                    reconstruction_loss: bool,
                    output_dict: Dict[str, torch.Tensor], 
                    metadata: List[Dict[str, Any]]):
        if (reconstruction_loss or not self.ignore_span_metric) and srl_frames is None: 
            raise ConfigurationError("Prediction loss required but gold labels `srl_frames` is None.")

        if not self.ignore_span_metric:
            gold_labels = srl_frames[pivot:, :] # 0 or batch_size
            self.span_metric(logits, gold_labels, mask)

        if reconstruction_loss:
            gold_labels = srl_frames[pivot:, :] # 0 or batch_size
            if self.suppress_nonarg: 
                gold_labels = gold_labels - 1 
                gold_labels[gold_labels < 0] = 0
            rec_loss = sequence_cross_entropy_with_logits(logits,
                                                          gold_labels,
                                                          mask,
                                                          label_smoothing=self._label_smoothing)
            output_dict["rec_loss"] = rec_loss

        output_dict["gold_srl"] = srl_frames[pivot:] # decoded in `decode` for the purpose of debugging
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

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
        # discard useless staff in the output dict 
        returned_dict = {"tokens": output_dict["tokens"], 
                         "lemmas": output_dict["lemmas"],
                         "pos_tags": output_dict["pos_tags"],
                         "head_ids": output_dict["head_ids"],
                         "predicate": output_dict["predicate"],
                         "predicate_index": output_dict["predicate_index"]}
        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        
        if self.suppress_nonarg:
            arg_masks = output_dict["arg_masks"]
            arg_idxes = output_dict["arg_idxes"]
            
            arg_masks = [arg_masks[i].detach().cpu().tolist() for i in range(arg_masks.size(0))]
            arg_idxes = [arg_idxes[i].detach().cpu().tolist() for i in range(arg_idxes.size(0))]
            
            constraints = []
            for isent, masks in enumerate(arg_masks):
                valid = []
                for iword, mask in enumerate(masks):
                    if mask == 1:
                        valid.append(arg_idxes[isent][iword])    
                    else:
                        break
                constraints.append(valid)
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
        returned_dict["gold_srl"] =gold_srl 
        return returned_dict

    def _logits_to_index(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Deprecated: Sub-procedure for decoding. 
        """
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()
        if logits.dim() < 3: # fake batch size
            logits.unsqueeze(0)
        _, max_likelihood_sequence = torch.max(logits, -1)
        
        for i, length in enumerate(sequence_lengths):
            # FIX ME: assumming 0 as empty label
            labels = max_likelihood_sequence[i, :length]
            nlabel = torch.sum(labels != 0)
            #n = 0
            while nlabel == 0:
                labels = torch.multinomial(logits[i], 1).squeeze(-1)[:length]
                nlabel = torch.sum(labels != 0)
                #n += 1
            #print('--sample {: >{}} times'.format(n, 3))
            max_likelihood_sequence[i, :length] = labels 
            max_likelihood_sequence[i, length:] = 0 
        return max_likelihood_sequence

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the span metric
            return {}
        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "f1-measure-overall" in x}

