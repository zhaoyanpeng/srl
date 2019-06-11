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


@Model.register("srl_vae_facto")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 autoencoder: Model,
                 alpha: float = 0.0,
                 nsampling: int = 10,
                 coupled_loss: bool = False,
                 straight_through: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.autoencoder = autoencoder 

        self.alpha = alpha
        self.nsampling = nsampling
        self.coupled_loss = coupled_loss 
        self.straight_through = straight_through
        self.autoencoder.add_parameters(self.classifier.nclass,
                                        self.vocab.get_vocab_size("lemmas"),
                                        None)
        self.tau = self.classifier.tau
        initializer(self)
    
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                lemmas: Dict[str, torch.LongTensor],
                predicates: torch.LongTensor,
                predicate_indicators: torch.LongTensor,
                argument_indices: torch.LongTensor = None,
                predicate_index: torch.LongTensor = None,
                argument_mask: torch.LongTensor = None,
                srl_frames: torch.LongTensor = None,
                retrive_crossentropy: bool = False,
                supervisely_training: bool = False, # deliberately added here
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # -1 if suppress nonarg
        out_dict = self.classifier.encode_patterns(tokens, predicate_indicators, argument_mask, argument_indices) 
        embedded_seqs = out_dict['embedded_seqs']
        logits, mask = out_dict['logits'], out_dict['mask']

        arg_logits, arg_labels, arg_lemmas = self.classifier.select_args(
            logits, srl_frames, lemmas['lemmas'], argument_indices) 

        C = self.classifier.labeled_loss(argument_mask, arg_logits, arg_labels) 
        C = self.alpha * C 
        
        #embedded_nodes = self.classifier.encode_args(
        #    lemmas, predicates, predicate_indicators, argument_indices, None) 
        embedded_nodes = self.classifier.encode_predt(predicates, predicate_indicators)
        encoded_labels = self.classifier.embed_labels(arg_labels, labels_add_one=True)  

        self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, arg_labels, encoded_labels)
        logits = self.autoencoder.logits

        # basic output stuff: only concerned with arg-prediction 
        output_dict = {"logits": logits, "mask": argument_mask, "C": C}
        
        self.classifier.add_arg_outputs(0, argument_mask, logits, arg_lemmas, output_dict, \
            all_labels=srl_frames, arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 

        if retrive_crossentropy: # must be true
            output_dict['ce_loss'] = self.classifier.labeled_loss(argument_mask, logits, arg_lemmas)


        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over


        ### correct loss
        # arg_logits: (batch_size, num_arg, nlabel) 
        # logits:     (batch_size, num_arg, nlemma)
        if not self.coupled_loss:
            loss = output_dict['ce_loss'] + C
        else:
            loss = self.joint_loss(argument_mask, logits, arg_lemmas, arg_logits, arg_labels) 

        # argument loss & label loss
        output_dict['loss'] = loss 

        return output_dict 

    def joint_loss(self, mask: torch.Tensor,
                   args_logits: torch.Tensor, 
                   args_lemmas: torch.Tensor,
                   role_logits: torch.Tensor, 
                   role_labels: torch.Tensor):
        args_logits_flat = args_logits.view(-1, args_logits.size(-1))
        args_log_probs = F.log_softmax(args_logits_flat, dim=-1)
        args_lemmas_flat = args_lemmas.view(-1, 1).long()
        
        role_logits_flat = role_logits.view(-1, role_logits.size(-1))
        role_log_probs = F.log_softmax(role_logits_flat, dim=-1)
        role_labels_flat = role_labels.view(-1, 1).long()
        
        args_neg_ll = -torch.gather(args_log_probs, dim=1, index=args_lemmas_flat)
        role_neg_ll = -torch.gather(role_log_probs, dim=1, index=role_labels_flat)
        
        neg_ll = args_neg_ll + self.alpha * role_neg_ll
        neg_ll = neg_ll.view(*role_labels.size())
        neg_ll = neg_ll * mask.float()
        
        per_batch_loss = neg_ll.sum(1) / (mask.sum(1).float() + 1e-13)
        num_valid_seqs = ((mask.sum(1) > 0).float().sum() + 1e-13)

        loss = per_batch_loss.sum() / num_valid_seqs
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode_args(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

