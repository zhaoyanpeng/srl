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


@Model.register("srl_gan")
class GanSemanticRoleLabeler(Model):
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
                 regularized_batch: bool = False,
                 mask_empty_labels: bool = True,
                 embedding_dropout: float = 0.,
                 use_label_indicator: bool = False,
                 zero_null_lemma_embedding: bool = False, 
                 label_loss_type: str = 'reverse_kl',
                 regularized_labels: List[str] = None,
                 use_wgan: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False) -> None:
        super(GanSemanticRoleLabeler, self).__init__(vocab, regularizer) 

        self.minimum = 1e-20
        self.minimum_temperature = 1e-5 
        self.mask_empty_labels = mask_empty_labels
        self.regularized_batch = regularized_batch
        

        self.token_embedder = token_embedder
        self.lemma_embedder = lemma_embedder
        self.label_embedder = label_embedder
        self.predicate_embedder = predicate_embedder
        
        self.num_classes = self.vocab.get_vocab_size("srl_tags")
        self.null_lemma_idx = self.vocab.get_token_index("NULL_LEMMA", namespace="lemmas")
        
        if zero_null_lemma_embedding:
            # this will make `lemma_embedder.token_embedder_lemmas.weight` become an non-leaf node
            # when the weight is trainable, and further result in errors in optimizer cause the
            # optimizer expects only leaf nodes.
            embedder = getattr(self.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
            embedder.weight[self.null_lemma_idx, :] = 0.
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
        if self.regularized_labels:
            """
            label_indexes = [self.vocab.get_token_index(label, namespace="srl_tags") 
                            for label in self.regularized_labels] 
            label_selector = torch.zeros(self.num_classes)
            label_selector[label_indexes] = 1. 
            self.label_selector = torch.nn.Parameter(label_selector, requires_grad=False)
            """
            pass
        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.span_metric = DependencyBasedF1Measure(vocab, tag_namespace="srl_tags", ignore_classes=["V"])

        self.seq_encoder = seq_encoder
        self.srl_encoder = srl_encoder
        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=fixed_temperature) 
        
        # set wgan model
        self.use_wgan = use_wgan
        self.srl_encoder.set_wgan(self.use_wgan)

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
            # (batch_size, length, dim); `expand` avoid data copy, unlike `repeat`
            embedded_predicates = embedded_predicates.expand(-1, -1, length).transpose(1, 2)
            # apply dropout to predicate embeddings
            embedded_predicate_input = self.embedding_dropout(embedded_predicates)
            
            batch_size = batch_size // 2
            
            # verbal part
            embedded_verb_tokens = embedded_token_input[:batch_size, :]
            embedded_verb_predicates = embedded_predicate_input[:batch_size, :] 

            # nominal part
            embedded_noun_tokens = embedded_token_input[batch_size:, :]
            embedded_noun_predicates = embedded_predicate_input[batch_size:, :] 
            
            embedded_tokens = embedded_noun_tokens
            used_mask = mask[batch_size:, :]
            used_pi = predicate_indicators[batch_size:, :]
        else:
            embedded_tokens = embedded_token_input
            used_mask = mask
            used_pi = predicate_indicators

        # Predict srl labels using the srler that needs to be learned.
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

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = used_mask
        
        
        if metadata is not None: 
            list_lemmas, list_tokens, list_pos_tags, list_head_ids, list_predicates, list_predicate_indexes = \
                                zip(*[(x["lemmas"], x["tokens"], x["pos_tags"], x["head_ids"], \
                                    x["predicate"], x["predicate_index"], ) for x in metadata])
         
        if self.training:
            class_probs, noun_labels, embedded_noun_labels, kl_loss = \
                self._argmax_logits(class_probabilities, used_mask, retrive_generator_loss)
            #noun_labels = self._logits_to_index(class_probabilities, used_mask) 
            output_dict["kl_loss"] = kl_loss
            output_dict["bp_loss"] = self._regularize_batch(class_probs, 
                                                            srl_frames[:batch_size, :], 
                                                            mask, batch_size, length, retrive_generator_loss) 

            predicted_labels = torch.cat([srl_frames[:batch_size, :], noun_labels], 0)
            full_label_masks = mask.clone() # srl label masks
            """ 
            print('\nlemmas: ', lemmas)
            print('predicted_labels: ', noun_labels, noun_labels.requires_grad)
            print('srl_frames: ', srl_frames, srl_frames.requires_grad)
            """
            if self.mask_empty_labels:
                # mask out empty labels 
                full_label_masks[predicted_labels == 0] = 0
            else:
                # use a special lemma embedding empty labels
                lemmas = lemmas['lemmas']
                lemmas = lemmas * (predicted_labels != 0).long()
                lemmas = lemmas + (predicted_labels == 0).long() * self.null_lemma_idx
                lemmas = {'lemmas': lemmas}
            """
            print('masked lemmas:', lemmas)
            print('label_masks ', full_label_masks, full_label_masks.requires_grad)
            """
            embedded_lemma_input = self.embedding_dropout(self.lemma_embedder(lemmas))
            embedded_verb_lemmas = embedded_lemma_input[:batch_size, :] 
            embedded_noun_lemmas = embedded_lemma_input[batch_size:, :] 
                
            #embedded_noun_labels = self.embedding_dropout(self.label_embedder(noun_labels))
            embedded_verb_labels = self.embedding_dropout(self.label_embedder(srl_frames[:batch_size, :]))
            if self.use_label_indicator:
                label_indicator = self.label_indicator(srl_frames[:batch_size, :]) 
                embedded_verb_labels = torch.cat([embedded_verb_labels, label_indicator], -1)
            """
            print('\nembedded_labels: {}\n'.format(embedded_verb_labels))
            print('\nembedded_size: {}\n'.format(embedded_verb_labels.size()))
            print('\nlabel_sequence:\n{}'.format(srl_frames[:batch_size, :]))
            import sys
            sys.exit(0)
            """
            gan_loss_input = {'v_embedded_tokens': embedded_verb_tokens,
                              'v_embedded_lemmas': embedded_verb_lemmas,
                              'v_embedded_predicates': embedded_verb_predicates,
                              'v_embedded_labels': embedded_verb_labels,
                              'n_embedded_tokens': embedded_noun_tokens,
                              'n_embedded_lemmas': embedded_noun_lemmas,
                              'n_embedded_predicates': embedded_noun_predicates,
                              'n_embedded_labels': embedded_noun_labels}
            self._gan_loss(gan_loss_input, full_label_masks, output_dict, retrive_generator_loss, only_reconstruction)
            
            #print('\nlogits:\n{}'.format(logits))
            #print('\ngold_labels:\n{}'.format(srl_frames[batch_size:, :]))
            #import sys
            #sys.exit(0)

            # assumming we can access gold labels for nominal part
            if reconstruction_loss:
                if srl_frames is None:
                    raise ConfigurationError("Prediction loss required but gold labels `srl_frames` is None.")
                gold_labels = srl_frames[batch_size:, :]
                rec_loss = sequence_cross_entropy_with_logits(logits,
                                                              gold_labels,
                                                              used_mask,
                                                              label_smoothing=self._label_smoothing)
                if not self.ignore_span_metric:
                    self.span_metric(class_probabilities, gold_labels, used_mask)
                output_dict["rec_loss"] = rec_loss
            
            if metadata is not None:
                output_dict["tokens"] = list(list_tokens)[batch_size:]
                output_dict["lemmas"] = list(list_lemmas)[batch_size:]
                output_dict["pos_tags"] = list(list_pos_tags)[batch_size:]
                output_dict["head_ids"] = list(list_head_ids)[batch_size:]
                output_dict["predicate"] = list(list_predicates)[batch_size:]
                output_dict["predicate_index"] = list(list_predicate_indexes)[batch_size:]

        else: # not training
            if reconstruction_loss:
                if srl_frames is None:
                    raise ConfigurationError("Prediction loss required but gold labels `srl_frames` is None.")
                loss = sequence_cross_entropy_with_logits(logits,
                                                          srl_frames,
                                                          used_mask,
                                                          label_smoothing=self._label_smoothing)
                if not self.ignore_span_metric:
                    self.span_metric(class_probabilities, srl_frames, mask)
                output_dict["rec_loss"] = loss
        
            if metadata is not None:
                output_dict["tokens"] = list(list_tokens)
                output_dict["lemmas"] = list(list_lemmas)
                output_dict["pos_tags"] = list(list_pos_tags)
                output_dict["head_ids"] = list(list_head_ids)
                output_dict["predicate"] = list(list_predicates)
                output_dict["predicate_index"] = list(list_predicate_indexes)
                """
                print('\n', output_dict["tokens"], '\n',
                        output_dict["lemmas"], '\n',
                        output_dict["pos_tags"], '\n',
                        output_dict["head_ids"], '\n',
                        output_dict["predicate_index"], '\n')
                """
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
        
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
        
        #print(self.vocab._token_to_index['srl_tags'])
        for predictions, length in zip(predictions_list, sequence_lengths):
            scores, max_likelihood_sequence = torch.max(predictions[:length], 1) 
            max_likelihood_sequence = max_likelihood_sequence.tolist() 
            
            #print(max_likelihood_sequence)

            tags = [self.vocab.get_token_from_index(x, namespace="srl_tags")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        returned_dict["srl_tags"] = all_tags
        return returned_dict
    
    def _regularize_batch(self, 
                          noun_class_probs: torch.Tensor, 
                          verb_labels: torch.Tensor,
                          all_mask: torch.Tensor,
                          batch_size: int,
                          length: int,
                          retrive_generator_loss: bool=False):
        if not self.regularized_batch or not retrive_generator_loss:
            return None

        verb_mask = all_mask[:batch_size, :]
        noun_mask = all_mask[batch_size:, :]
        
        class_probs = noun_class_probs * noun_mask.unsqueeze(-1).float() 
        batch_probs = torch.sum(class_probs, 1)

        verb_labels = verb_labels * verb_mask
        
        gold_labels = torch.zeros(self.num_classes, length, device=all_mask.device)
        ones_labels = torch.ones(verb_labels.size(), device=all_mask.device)
        gold_labels.scatter_add_(0, verb_labels, ones_labels)
         
        # ignore non-argument label
        kl_gold = torch.sum(gold_labels, -1) + 1.
        kl_gold = kl_gold[1:] / torch.sum(kl_gold[1:])
        #print(kl_gold, kl_gold.size())
        
        kl_pseu = torch.sum(batch_probs, 0) + 1.
        kl_pseu = kl_pseu[1:] / torch.sum(kl_pseu[1:]) 
        #print(kl_pseu, kl_pseu.size())
        
        kl_loss = kl_gold * torch.log(kl_gold / kl_pseu)
        #print(kl_loss, kl_loss.size())
        kl_loss = torch.sum(kl_loss)
        #print(kl_loss) 
        #import sys
        #sys.exit(0)
        return kl_loss

    def _gan_loss(self, 
                  input_dict: Dict[str, torch.Tensor], 
                  mask: torch.Tensor, 
                  output_dict: Dict[str, Any],
                  retrive_generator_loss: bool,
                  return_without_computation: bool = False) -> None:
        if return_without_computation:
            default_value = torch.tensor(0.0, dtype=torch.float, device=mask.device)
            if retrive_generator_loss:
                output_dict['gen_loss'] = default_value 
            else:
                output_dict['dis_loss'] = default_value
            return None

        embedded_noun_tokens = input_dict['n_embedded_tokens']
        embedded_noun_lemmas = input_dict['n_embedded_lemmas']
        embedded_noun_predicates = input_dict['n_embedded_predicates']
        embedded_noun_labels = input_dict['n_embedded_labels']
        
        batch_size = embedded_noun_tokens.size(0)
        
        noun_features = [embedded_noun_predicates, 
                         embedded_noun_labels, 
                         embedded_noun_lemmas]
        embedded_noun = torch.cat(noun_features, -1)
        
        """ 
        for k, v in input_dict.items():
            print('\n{} = {}'.format(k, v.size()) )

        print(embedded_noun.size()) 
        """
        
        if retrive_generator_loss:
            mask = mask[batch_size:]
            logits = self.srl_encoder(embedded_noun, mask)  

            if self.use_wgan:
                gen_loss = -torch.mean(logits)                
            else:
                # fake labels 
                real_labels = mask[:, 0].detach().clone().fill_(1).float()
                gen_loss = F.binary_cross_entropy(logits, real_labels, reduction='mean')
            output_dict['gen_loss'] = gen_loss
        else:
            embedded_verb_tokens = input_dict['v_embedded_tokens']
            embedded_verb_lemmas = input_dict['v_embedded_lemmas']
            embedded_verb_predicates = input_dict['v_embedded_predicates']
            embedded_verb_labels = input_dict['v_embedded_labels']
            
            verb_features = [embedded_verb_predicates,
                             embedded_verb_labels,
                             embedded_verb_lemmas]

            embedded_verb = torch.cat(verb_features, -1)
            #embedded_input = torch.cat([embedded_verb, embedded_noun], 0)

            #logits = self.srl_encoder(embedded_input, mask)
            mask_noun = mask[batch_size:]
            logits_noun = self.srl_encoder(embedded_noun, mask_noun) 
            
            mask_verb = mask[:batch_size]
            logits_verb = self.srl_encoder(embedded_verb, mask_verb)

            if self.use_wgan:
                dis_loss = -torch.mean(logits_verb) + torch.mean(logits_noun) 
            else:
                # fake labels
                fake_labels = mask[:batch_size, 0].detach().clone().fill_(0).float()
                real_labels = mask[:batch_size, 0].detach().clone().fill_(1).float()
                
                #dis_loss = F.binary_cross_entropy(logits[:batch_size], real_labels, reduction='mean') \
                #         + F.binary_cross_entropy(logits[batch_size:], fake_labels, reduction='mean') 
                dis_loss = F.binary_cross_entropy(logits_verb, real_labels, reduction='mean') \
                         + F.binary_cross_entropy(logits_noun, fake_labels, reduction='mean') 
            output_dict['dis_loss'] = dis_loss / 2 
        return None 
       
    def _argmax_logits(self, logits: torch.Tensor, mask: torch.Tensor, retrive_generator_loss: bool=False) -> torch.Tensor:
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()
        if logits.dim() < 3: # fake batch size
            logits.unsqueeze(0)
        
        batch_size, sequence_length, _ = logits.size()

        self.temperature.data.clamp_(min = self.minimum_temperature)

        class_probs = F.gumbel_softmax(
            torch.log(logits.view(-1, self.num_classes) + self.minimum), 
            tau = self.temperature,
            hard = True).view([batch_size, sequence_length, self.num_classes]) 
        
        #print('\ngumbel_softmax:\n{}'.format(class_probs))
        #print('\ngumbel_size: {}\n'.format(class_probs.size()))
        
        #print(self.label_embedder.weight)
        """         
        print()
        _, max_likelihood_sequence = torch.max(logits, -1)
        for i, length in enumerate(sequence_lengths):
            max_likelihood_sequence[i, length:] = 0
            tags = [self.vocab.get_token_from_index(x.item(), namespace="srl_tags")
                    for x in max_likelihood_sequence[i, :]]
            print(tags)
        print('\nmax_ll_sequence:\n{}'.format(max_likelihood_sequence))
        """
        
        embedded_labels =self.embedding_dropout(
            torch.matmul(class_probs, self.label_embedder.weight)) 
        _, max_likelihood_sequence = torch.max(class_probs, -1)
        for i, length in enumerate(sequence_lengths):
            max_likelihood_sequence[i, length:] = 0
        
        if self.use_label_indicator:
            label_indicator = self.label_indicator(max_likelihood_sequence)    
            embedded_labels = torch.cat([embedded_labels, label_indicator], -1)

        if self.regularized_labels and retrive_generator_loss:
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
                pass
            kl_loss = kl_loss[:, 1:] # discard the loss for empty labels

            loss = torch.mean(torch.sum(kl_loss, 1)) # loss at sentence level
            #loss = torch.mean(kl_loss[kl_loss > 0]) # loss at token level 
        else:
            loss = None 

        #print('\nembedded_labels: {}\n'.format(embedded_labels))
        #print('\nembedded_size: {}\n'.format(embedded_labels.size()))
        #print('\nlabel_sequence:\n{}'.format(max_likelihood_sequence))
        #import sys
        #sys.exit(0)
        return class_probs, max_likelihood_sequence, embedded_labels, loss 


    def _logits_to_index(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Sub-procedure for decoding. 
        """
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()
        if logits.dim() < 3: # fake batch size
            logits.unsqueeze(0)
        _, max_likelihood_sequence = torch.max(logits, -1)
        
        #print('\npredicted_labels:\n{}'.format(max_likelihood_sequence))

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
        
        #print('\nresampled_labels:\n{}'.format(max_likelihood_sequence))
        #import sys
        #sys.exit(0)
        return max_likelihood_sequence
        
        """
        if logits.dim() == 3:
            batch_size, length, _ = logits.size()
            predictions_list = [logits[i].detach().cpu() for i in range(batch_size)]
        else:
            batch_size, length = 1, logits.size(0)
            predictions_list = [logits]
            
        all_labels = torch.zeros([batch_size, length], dtype=torch.long)
        # FIX ME: assuming 0 is the index of the empty label, but the data reader does not ensure this. 
        for i, (predictions, length) in enumerate(zip(predictions_list, sequence_lengths)):
            _, max_likelihood_sequence = torch.max(predictions[:length], 1) 
            all_labels[i, :length] = max_likelihood_sequence
        return all_labels
        print(all_labels)
        """

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}
        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "f1-measure-overall" in x}

