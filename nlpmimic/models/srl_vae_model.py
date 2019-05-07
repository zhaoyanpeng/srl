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


@Model.register("srl_vae")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 autoencoder: Model,
                 alpha: float = 0.0,
                 nsampling: int = 10,
                 tau: float = 1.,
                 tunable_tau: bool = False,
                 straight_through: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.autoencoder = autoencoder
        self.classifier = classifier
        self.nsampling = nsampling

        self.alpha = alpha

        self.autoencoder.add_parameters(self.classifier.nclass,
                                        self.vocab.get_vocab_size("lemmas"))
        
        self.minimum_tau = 1e-5 
        self.tau = torch.nn.Parameter(torch.tensor(tau), requires_grad=tunable_tau) 
        self.straight_through = straight_through

        initializer(self)
    
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                lemmas: Dict[str, torch.LongTensor],
                predicate_indicators: torch.LongTensor,
                predicates: torch.LongTensor,
                argument_indices: torch.LongTensor = None,
                predicate_index: torch.LongTensor = None,
                argument_mask: torch.LongTensor = None,
                srl_frames: torch.LongTensor = None,
                reconstruction_loss: bool = False,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """ The first halve is labeled data; the sencod halve is unlabeled data.
        """
        # shared by labeled and unlabeled halves
        batch_size = predicate_indicators.size(0) // 2 
        pivot = batch_size if self.training else 0
        
        out_dict = self.classifier(tokens, predicate_indicators) 
        embedded_seqs = out_dict['embedded_seqs']
        logits, mask = out_dict['logits'], out_dict['mask']

        arg_logits, arg_labels, arg_lemmas = self.classifier.select_args(
            logits, srl_frames, lemmas['lemmas'], argument_indices) 
        
        # basic output stuff 
        output_dict = {"logits": logits[pivot:],
                       "logits_softmax": out_dict['logits_softmax'][pivot:],
                       "mask": mask[pivot:]}
        
        self.classifier.add_outputs(pivot, mask, logits, srl_frames, output_dict, \
            arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 
        if reconstruction_loss:
            output_dict['rec_loss'] = self.classifier.labeled_loss(
                argument_mask[pivot:], arg_logits[pivot:], arg_labels[pivot:])
        

        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over

        
        # below we finalize all the training stuff
        embedded_nodes = self.classifier.encode_args(
            lemmas, predicates, predicate_indicators, argument_indices, embedded_seqs) 

        ### labeled halve

        # classification loss for the labeled data
        C = self.classifier.labeled_loss(
            argument_mask[:batch_size], arg_logits[:batch_size], arg_labels[:batch_size], average=None) 
        # used in decoding
        encoded_labels = self.classifier.embed_labels(arg_labels[:batch_size], labels_add_one=True)  

        L = self.autoencoder(argument_mask[:batch_size], 
                             arg_lemmas[:batch_size],
                             embedded_nodes[:batch_size], 
                             arg_labels[:batch_size],
                             encoded_labels)
        L = -L
        ### unlabled halve

        self.tau.data.clamp_(min = self.minimum_tau)
        arg_mask = argument_mask[batch_size:]
        y_logs, y_ls = [], []
        for _ in range(self.nsampling):
            # gumbel relaxation for unlabeled halve
            gumbel_hard, gumbel_soft, gumbel_soft_log, sampled_labels = \
                self.classifier.gumbel_relax(arg_mask, arg_logits[batch_size:], tau=self.tau)
            # used in decoding
            labels_relaxed = gumbel_hard if self.straight_through else gumbel_false
            encoded_labels = self.classifier.embed_labels(None, labels_relaxed=labels_relaxed)  
            
            L_y = self.autoencoder(arg_mask, 
                                   arg_lemmas[batch_size:],
                                   embedded_nodes[batch_size:], 
                                   sampled_labels,
                                   encoded_labels)
            
            hard_lprobs = (gumbel_hard * gumbel_soft_log).sum(-1)
            hard_lprobs = hard_lprobs.masked_fill(arg_mask == 0, 0)
            y_log = torch.sum(hard_lprobs, -1)

            y_logs.append(y_log)
            y_ls.append(L_y)

        # average    
        y_logs = torch.stack(y_logs, 0)
        y_ls = torch.stack(y_ls, 0)
        # along sample dimension
        y_probs = torch.exp(y_logs).softmax(0)
        y_ls = y_ls * y_probs
        
        H = torch.log(y_probs + self.minimum_float) * y_probs

        H = -H.sum(0)
        L_u = -y_ls.sum(0)

        total_loss = L + H + L_u + self.alpha * C 

        L, L_u, H, C = torch.mean(L), torch.mean(L_u), torch.mean(H), torch.mean(C) 
        
        # batch dimension
        output_dict['L'] = L 
        output_dict['L_u'] = L_u 
        output_dict['H'] = H 
        output_dict['C'] = C 
        output_dict['vae_loss'] = torch.mean(total_loss)
        #import sys
        #sys.exit(0)
        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

