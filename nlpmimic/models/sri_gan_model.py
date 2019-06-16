from typing import Dict, List, TextIO, Optional, Any
from overrides import overrides

import torch
import torch.nn.functional as F
from torch.nn.modules import Linear, Dropout

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

@Model.register("sri_gan")
class GanSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 generator: Model,
                 discriminator: Model,
                 straight_through: bool = True,
                 use_uniqueness_prior: bool = False,
                 uniqueness_loss_type: str = 'reverse_kl',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(GanSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25
        
        self.generator = generator
        self.discriminator = discriminator
        self.straight_through = straight_through

        self.uniqueness_loss_type = uniqueness_loss_type
        self.use_uniqueness_prior = use_uniqueness_prior

        self.discriminator.add_parameters(self.generator.nclass,
                                          self.generator.suppress_nonarg)
        self.tau = self.generator.tau
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
                optimizing_generator: bool = False,
                relying_on_generator: bool = False, 
                caching_feature_only: bool = False,
                retrive_crossentropy: bool = False,
                supervisely_training: bool = False, # deliberately added here
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        pivot = 0 # will be input to the 'decode' procedure
        if not self.training or optimizing_generator or relying_on_generator:
            out_dict = self.generator(tokens, predicate_indicators) 
            embedded_seqs = out_dict['embedded_seqs']
            logits, mask = out_dict['logits'], out_dict['mask']
             
            arg_logits, arg_labels, arg_lemmas = self.generator.select_args(
                logits, srl_frames, lemmas['lemmas'], argument_indices) 

            # basic output stuff 
            output_dict = {"logits": logits[pivot:],
                           "logits_softmax": out_dict['logits_softmax'][pivot:],
                           "mask": mask[pivot:]}
            
            self.generator.add_outputs(pivot, mask, logits, srl_frames, output_dict, \
                arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 
            
            if retrive_crossentropy:
                if self.generator.suppress_nonarg:
                    output_dict['ce_loss'] = self.generator.labeled_loss(argument_mask, arg_logits, arg_labels)
                else:
                    output_dict['ce_loss'] = self.generator.labeled_loss(mask, logits, srl_frames)
        

        ### eveluation only 
        if not self.training: #or supervisely_training:
            #print('\n------     ---------     ---------     ---------- Evaluation or supervised training')
            output_dict['loss'] = None
            return output_dict
        ### evaluation over


        # below we finalize all the training stuff
        embedded_nodes = self.discriminator.encode_args(
            lemmas, predicates, predicate_indicators, argument_indices) 

        # labeled data input to the discriminator 
        if not optimizing_generator and not relying_on_generator:
            _, arg_labels, _ = self.generator.select_args(None, srl_frames, None, argument_indices) 

            #print('\n------     ---------     ---------     ---------- DIS with real input')
            loss = self.discriminator(argument_mask, embedded_nodes, arg_labels,
                                      optimizing_generator=optimizing_generator,
                                      relying_on_generator=relying_on_generator,
                                      caching_feature_only=caching_feature_only)
            loss_gp = self.discriminator.gradient_penalty(argument_mask, embedded_nodes, arg_labels)
            output_dict = {'gp_loss': loss_gp}

        # unlabeled data input to either the discriminator or generator
        if relying_on_generator:
            # gumbel relaxation for unlabeled halve
            gumbel_hard, gumbel_soft, _, arg_labels = \
                self.generator.gumbel_relax(argument_mask, arg_logits)
            labels_relaxed = gumbel_hard if self.straight_through else gumbel_false

            arg_softmax = None
            if not optimizing_generator: # used in wgan with gradient penalty
                arg_softmax = F.softmax(arg_logits, -1) 

            #print('\n------     ---------     ---------     ---------- Relying on the GEN')
            loss = self.discriminator(argument_mask, embedded_nodes, arg_labels, 
                                      optimizing_generator=optimizing_generator,
                                      relying_on_generator=relying_on_generator,
                                      caching_feature_only=caching_feature_only,
                                      edge_type_softmax=arg_softmax,
                                      edge_type_onehots=labels_relaxed)
        output_dict['loss'] = loss

        # kl divergence, etc.
        output_dict['kl_loss'] = None
        if self.use_uniqueness_prior and optimizing_generator:
            output_dict['kl_loss'] = self.regularize_labels(argument_mask, gumbel_hard, self.uniqueness_loss_type) 
        return output_dict 

    def regularize_labels(self, mask: torch.Tensor, label_onehots: torch.Tensor, loss_type: str):
        label_onehots = label_onehots * mask.unsqueeze(-1).float()
        batch_probs = torch.sum(label_onehots, 1)
        batch_probs.clamp_(min = 1.) # a trick to avoid nan = log(0)
        # assumming no empty label so pivot is 0
        kl_loss = self.generator.kldivergence(0, mask, batch_probs, loss_type) 
        return kl_loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.generator.decode(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.generator.get_metrics(reset=reset)

