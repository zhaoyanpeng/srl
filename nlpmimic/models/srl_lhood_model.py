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
                 sim_loss_type: str = None, # l2 or cin
                 straight_through: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.autoencoder = autoencoder 

        self.alpha = alpha
        self.nsampling = nsampling
        self.sim_loss_type = sim_loss_type
        self.straight_through = straight_through

        output_dim = self.vocab.get_vocab_size("lemmas")
        lemma_embedder = getattr(self.classifier.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
        if self.sim_loss_type is not None:
            output_dim = self.classifier.lemma_embedder.get_output_dim()
            self.lemma_vectors = lemma_embedder.weight
        self.autoencoder.add_parameters(self.classifier.nclass, output_dim, lemma_embedder.weight)

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
        
        if self.sim_loss_type is not None:
            embedded_nodes = self.classifier.encode_args(
                lemmas, predicates, predicate_indicators, argument_indices, None) 
            arg_embeddings = embedded_nodes[:, 1:, :]
            embedded_nodes = embedded_nodes[:, :1, :]
        else:
            embedded_nodes = self.classifier.encode_predt(predicates, predicate_indicators)

        encoded_labels = self.classifier.embed_labels(arg_labels, labels_add_one=True)  

        self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, arg_labels, encoded_labels)
        logits = self.autoencoder.logits

        # basic output stuff 
        output_dict = {"logits": logits, "mask": argument_mask}
        #self.classifier.add_arg_outputs(0, argument_mask, logits, arg_lemmas, output_dict, \
        #    all_labels=srl_frames, arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 

        if retrive_crossentropy and self.sim_loss_type is not None:
            output_dict['ce_loss'] = self.sim_loss(logits, arg_embeddings, arg_mask=argument_mask)
        elif retrive_crossentropy:
            output_dict['ce_loss'] = self.classifier.labeled_loss(argument_mask, logits, arg_lemmas)


        # return label embedding matrix and expand it along batch size
        batch_size = encoded_labels.size(0) 
        encoded_labels = self.classifier.embed_labels(None)   
        encoded_labels = encoded_labels.unsqueeze(0).expand(batch_size, -1, -1)
        self.autoencoder(None, None, embedded_nodes, None, encoded_labels)
        roles_logits = self.autoencoder.logits
        
        if self.sim_loss_type is not None:
            roles_logits = roles_logits.unsqueeze(2)
            lemma_vectors = self.lemma_vectors.unsqueeze(0).unsqueeze(0)
            # using negative loss as logits
            roles_logits = -self.sim_loss(roles_logits, lemma_vectors, reduction = None)
        else:
            roles_logits = F.softmax(roles_logits, -1) # to be comparable among roles
             
        index = arg_lemmas.unsqueeze(1).expand(-1, self.classifier.nclass, -1)
        roles_logits = torch.gather(roles_logits, -1, index) 
        roles_logits = roles_logits.transpose(1, 2)

        output_dict['logits'] = roles_logits
        self.classifier.add_argument_outputs(0, argument_mask, roles_logits, arg_labels + 1, output_dict, \
            all_labels=srl_frames, arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 


        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over

        output_dict['loss'] = output_dict['ce_loss']

        return output_dict 
    
    def sim_loss(self, 
                 prediction: torch.Tensor, 
                 arg_embeddings: torch.Tensor,
                 arg_mask: torch.Tensor = None,
                 reduction: str = 'mean') -> torch.Tensor:
        if self.sim_loss_type == 'l2':
            loss = ((prediction - arg_embeddings) ** 2).sum(-1)
        elif self.sim_loss_type == 'cosine':
            loss = 1 - F.cosine_similarity(prediction, arg_embeddings, -1)
        else:
            pass

        if reduction == 'mean':
            loss = loss * arg_mask.float()
            loss = torch.mean(loss.sum(-1))
        return loss 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode_arguments(output_dict)
        #return self.classifier.decode_args(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

