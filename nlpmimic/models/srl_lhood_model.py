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


@Model.register("srl_vae_lhood")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 autoencoder: Model,
                 alpha: float = 0.0,
                 nsampling: int = 10,
                 straight_through: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.autoencoder = autoencoder 

        self.alpha = alpha
        self.nsampling = nsampling
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
        _, arg_labels, arg_lemmas = self.classifier.select_args(
            None, srl_frames, lemmas['lemmas'], argument_indices) 
        
        #embedded_nodes = self.classifier.encode_args(
        #    lemmas, predicates, predicate_indicators, argument_indices, None) 
        embedded_nodes = self.classifier.encode_predt(predicates, predicate_indicators)

        encoded_labels = self.classifier.embed_labels(arg_labels, labels_add_one=True)  

        self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, arg_labels, encoded_labels)
        logits = self.autoencoder.logits

        # basic output stuff 
        output_dict = {"logits": logits, "mask": argument_mask}
        
        self.classifier.add_arg_outputs(0, argument_mask, logits, arg_lemmas, output_dict, \
            all_labels=srl_frames, arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 

        if retrive_crossentropy:
            output_dict['ce_loss'] = self.classifier.labeled_loss(argument_mask, logits, arg_lemmas)

        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over

        output_dict['loss'] = output_dict['ce_loss']

        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode_args(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

