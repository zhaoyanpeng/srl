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
                 reweight: bool = True,
                 coupled_loss: bool = False,
                 sim_loss_type: str = None, # l2 or cin
                 inf_loss_type: str = None, # pa, py, or ex <-> l(a | y, p), p(y | w-a), or expected loss
                 straight_through: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.autoencoder = autoencoder 

        self.alpha = alpha
        self.reweight = reweight
        self.nsampling = nsampling
        self.coupled_loss = coupled_loss 
        self.sim_loss_type = sim_loss_type
        self.inf_loss_type = inf_loss_type
        self.straight_through = straight_through

        output_dim = self.vocab.get_vocab_size("lemmas")
        if self.sim_loss_type is not None:
            output_dim = self.classifier.lemma_embedder.get_output_dim()
            lemma_embedder = getattr(self.classifier.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
            self.lemma_vectors = lemma_embedder.weight
        self.autoencoder.add_parameters(self.classifier.nclass, output_dim, None)
        
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

        out_dict = self.classifier.encode_patterns(tokens, predicate_indicators, argument_mask, argument_indices) 
        #out_dict = self.classifier(tokens, predicate_indicators) 

        embedded_seqs = out_dict['embedded_seqs']
        logits, mask = out_dict['logits'], out_dict['mask']

        arg_logits, arg_labels, arg_lemmas = self.classifier.select_args(
            logits, srl_frames, lemmas['lemmas'], argument_indices) 

        # basic output stuff: only concerned with role-prediction 
        output_dict = {"logits": arg_logits, "mask": argument_mask, 'ce_loss': None}

        if self.sim_loss_type is not None:
            embedded_nodes = self.classifier.encode_args(
                lemmas, predicates, predicate_indicators, argument_indices, None) 
            arg_embeddings = embedded_nodes[:, 1:, :]
            embedded_nodes = embedded_nodes[:, :1, :]
        else:
            arg_embeddings = None 
            embedded_nodes = self.classifier.encode_predt(predicates, predicate_indicators)
        batch_size = embedded_nodes.size(0) 

        if not retrive_crossentropy:  
            self.classifier.add_argument_outputs(0, argument_mask, arg_logits, arg_labels + 1, output_dict, \
                all_labels=srl_frames, arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 
        else: # being used as an indicator of evaluation
            # return label embedding matrix and expand it along batch size
            encoded_labels = self.classifier.embed_labels(None)   
            encoded_labels = encoded_labels.unsqueeze(0).expand(batch_size, -1, -1)
            self.autoencoder(None, None, embedded_nodes, None, encoded_labels)
            roles_logits = self.autoencoder.logits # (batch_size, nrole, nlemma)
            roles_logits = F.softmax(roles_logits, -1) # to be comparable among roles

            if self.sim_loss_type is not None:
                roles_logits = roles_logits.unsqueeze(2)
                lemma_vectors = self.lemma_vectors.unsqueeze(0).unsqueeze(0)
                # using negative loss as logits, the larger the better
                roles_logits = -self.sim_loss(roles_logits, lemma_vectors, reduction = None)
                 
            # extract arg-role pairs (batch_size, narg, nrole)
            index = arg_lemmas.unsqueeze(1).expand(-1, self.classifier.nclass, -1)
            roles_logits = torch.gather(roles_logits, -1, index) 
            roles_logits = roles_logits.transpose(1, 2)
            
            # how to compute coupled logits during inference
            if self.inf_loss_type == 'ex': # expected metric 
                lp_y = F.log_softmax(arg_logits, -1)
                lp_a = F.log_softmax(roles_logits, -1)
                roles_logits = lp_y + lp_a
            elif self.inf_loss_type == 'py': # p(y | w-a)
                roles_logits = arg_logits
            elif self.inf_loss_type == 'pa': # p(a | y, p)
                roles_logits = roles_logits 
            else:
                raise ValueError('Please specify evaluation metric.')

            output_dict['logits'] = roles_logits
            self.classifier.add_argument_outputs(0, argument_mask, roles_logits, arg_labels + 1, output_dict, \
                all_labels=srl_frames, arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 

        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over

        ### labeled halve
        if supervisely_training:
            # used in decoding
            encoded_labels = self.classifier.embed_labels(arg_labels, labels_add_one=True)  
            self.autoencoder(None, None, embedded_nodes, None, encoded_labels)
            arg_prediction = self.autoencoder.logits
            
            C, LL, loss = self.all_loss(argument_mask, 
                                        arg_lemmas, 
                                        arg_prediction, 
                                        arg_embeddings,
                                        arg_logits,
                                        arg_labels)
            output_dict['C'] = C 
            output_dict['LL'] = LL 
            output_dict['L'] = loss 
            output_dict['loss'] = loss 
        else: ### unlabled halve
            encoded_labels = self.classifier.embed_labels(None)   
            encoded_labels = encoded_labels.unsqueeze(0).expand(batch_size, -1, -1)
            self.autoencoder(None, None, embedded_nodes, None, encoded_labels)
            roles_logits = self.autoencoder.logits # (batch_size, nrole, nlemma)
            roles_logits = F.softmax(roles_logits, -1) # to be comparable among roles

            # extract arg-role pairs (batch_size, narg, nrole)
            index = arg_lemmas.unsqueeze(1).expand(-1, self.classifier.nclass, -1)
            roles_logits = torch.gather(roles_logits, -1, index) 
            roles_logits = roles_logits.transpose(1, 2)

            if self.sim_loss_type is not None:
                raise ValueError('Computation of joint probabilitiy needs self.sim_loss_type = None') 

            lp_y = F.log_softmax(arg_logits, -1)
            lp_a = F.log_softmax(roles_logits, -1)
            lp_arg_roles = lp_y + lp_a
            lp_arg = torch.logsumexp(lp_arg_roles, -1)
            lp_arg = lp_arg * argument_mask.float()
            lp_batch = -torch.sum(lp_arg, -1) 

            lls = torch.mean(lp_batch)
            output_dict['L_u'] = lls
            output_dict['loss'] = lls 
        return output_dict 
    
    def all_loss(self,
                 arg_mask: torch.Tensor,
                 arg_lemmas: torch.Tensor,
                 arg_prediction: torch.Tensor,
                 arg_embeddings: torch.Tensor,
                 role_prediction: torch.Tensor,
                 role_gold_label: torch.Tensor):
        C = LL = loss = None
        if not self.coupled_loss:
            # classification loss for the labeled data
            C = self.classifier.labeled_loss(arg_mask, role_prediction, role_gold_label) 
            # argument prediction loss
            if self.sim_loss_type is not None:
                LL = self.sim_loss(arg_prediction, arg_embeddings, arg_mask=arg_mask)
            else:
                LL = self.classifier.labeled_loss(arg_mask, arg_prediction, arg_lemmas)
            loss = LL + self.alpha * C
        else:
            loss = self.joint_loss(arg_mask, 
                                   arg_lemmas, 
                                   arg_prediction, 
                                   arg_embeddings, 
                                   role_prediction,
                                   role_gold_label) 
        return C, LL, loss

    def sim_loss(self, 
                 prediction: torch.Tensor, 
                 arg_embeddings: torch.Tensor,
                 arg_mask: torch.Tensor = None,
                 reduction: str = 'mean') -> torch.Tensor:
        """ p(a | y, p) -- (batch_size, nrole, nlemma)
        """
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

    def joint_loss(self, mask: torch.Tensor,
                   args_lemmas: torch.Tensor,
                   args_logits: torch.Tensor, 
                   args_embedding: torch.Tensor,
                   role_logits: torch.Tensor, 
                   role_labels: torch.Tensor):
        if self.sim_loss_type is not None:
            args_neg_ll = self.sim_loss(args_logits, args_embedding, reduction = None)
            args_neg_ll = args_neg_ll.view(-1, 1)
        else:
            args_logits_flat = args_logits.view(-1, args_logits.size(-1))
            args_log_probs = F.log_softmax(args_logits_flat, dim=-1)
            args_lemmas_flat = args_lemmas.view(-1, 1).long()
            args_neg_ll = -torch.gather(args_log_probs, dim=1, index=args_lemmas_flat)
        
        role_logits_flat = role_logits.view(-1, role_logits.size(-1))
        role_log_probs = F.log_softmax(role_logits_flat, dim=-1)
        role_labels_flat = role_labels.view(-1, 1).long()
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
        return self.classifier.decode_arguments(output_dict)
        #return self.classifier.decode_args(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

