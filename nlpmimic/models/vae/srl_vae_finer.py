from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed 
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from nlpmimic.modules.seq2vec_encoders.sampler import SamplerUniform

@Model.register("srl_finer_ae")
class SrlFinerAutoencoder(Model):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder = None, # a latent z
                 decoder: Seq2SeqEncoder = None, # a sequence of recovered inputs (e.g., arguments & contexts)
                 sampler: Seq2VecEncoder = None, # posterior distribution
                 kl_alpha: float = 1.0,   # kl weight
                 ll_alpha: float = 1.0,   # ll weight
                 re_alpha: float = 0.5,   # regularizer for duplicate roles in an instance
                 nsample: int = 1,        # # of samples from the posterior distribution 
                 label_smoothing: float = None,
                 b_use_z: bool = True,
                 b_ctx_lemma: bool = False,
                 b_ctx_predicate: bool = False, # b: boolean value
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlFinerAutoencoder, self).__init__(vocab, regularizer)
        self.signature = 'srl_lstms_ae'
        
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.nsample = nsample
        self._label_smoothing = label_smoothing
        self._b_ctx_predicate = b_ctx_predicate
        self._b_ctx_lemma = b_ctx_lemma
        self._b_use_z = b_use_z
        
        self.kl_alpha = kl_alpha
        self.ll_alpha = ll_alpha
        self.re_alpha = re_alpha

        self.kldistance = None 
        self.likelihood = None 
    
    def add_parameters(self, nlabel: int, nlemma: int, lemma_embedder_weight: torch.Tensor):
        self.nlabel = nlabel
        self.nlemma = nlemma
        if self.encoder is not None:
            self.encoder.add_parameters(nlabel)
        if self.decoder is not None:
            self.decoder.add_parameters(nlemma, lemma_embedder_weight)

    def forward(self, 
                mask: torch.Tensor,
                node_types: torch.Tensor,
                embedded_nodes: torch.Tensor,  
                edge_types: torch.Tensor,
                embedded_edges: torch.Tensor,
                edge_type_onehots: torch.Tensor = None,
                contexts: torch.Tensor = None) -> torch.Tensor:
        # encoding them if it is needed
        if self._b_use_z or self._b_ctx_predicate:
            z = self.encoder(mask, embedded_nodes, edge_types, edge_type_onehots)
            z = z.unsqueeze(1)
        embedded_predicates = embedded_nodes[:, [0], :]
        if self._b_ctx_predicate:
            embedded_predicates = self.encoder.embedded_predicates
        if not self._b_use_z:
            z = None
        
        ctx_lemmas = None
        # be aware other lemmas
        if self._b_ctx_lemma:
            embedded_lemmas = embedded_nodes[:, 1:, :]
            labeled_lemmas = torch.cat([embedded_lemmas, embedded_edges], -1)
            labeled_lemmas = labeled_lemmas * mask.unsqueeze(-1).float()
            
            labeled_lemmas_sum = torch.sum(labeled_lemmas, -2, keepdim=True)
            ctx_lemmas = labeled_lemmas_sum - labeled_lemmas

            divider = torch.sum(mask, -1, keepdim=True) - 1
            divider = torch.clamp(divider, min = 1).unsqueeze(-1).float()
            ctx_lemmas = ctx_lemmas / divider

        # reconstruction (argument) loss (batch_size,)
        logits = self.decoder(z, embedded_edges, embedded_predicates, nodes_contexts=ctx_lemmas)
        llh = self._likelihood(mask, logits, node_types, average = None) 
        self.likelihood = self.ll_alpha * llh 

        return -self.likelihood 

    def kld(self, posteriors, # logrithm 
            mask: torch.Tensor = None,
            node_types: torch.Tensor = None,
            embedded_nodes: torch.Tensor = None,  
            embedded_edges: torch.Tensor = None,
            edge_type_onehots: torch.Tensor = None,
            contexts: torch.Tensor = None) -> torch.Tensor:
        if isinstance(self.sampler, SamplerUniform):
            # priors of role labels
            nlabel = torch.tensor(self.nlabel, device=posteriors.device).float()
            pz_logs = -torch.sum(mask, -1).float() * torch.log(nlabel)
            self.kldistance = self.kl_alpha * (posteriors - pz_logs)
        else: # use estimated statistics of role labels as priors
            #self.kldistance = self.kl_alpha * posteriors 
            raise ValueError('Not supported.')

        uniqueness = None
        # an additional regularizer for duplicate roles
        if edge_type_onehots is not None: 
            label_onehots = edge_type_onehots * mask.unsqueeze(-1).float()
            batch_probs = torch.sum(label_onehots, 1) # along argument dim

            batch_probs.clamp_(min = 1.)   # a trick to avoid nan = log(0)

            kl_loss = torch.log(batch_probs) * batch_probs
            kl_loss = torch.sum(kl_loss, 1) # along label dim
            
            # loss on sentence level
            uniqueness = self.re_alpha * kl_loss 
        return (self.kldistance, uniqueness) 

    def _likelihood(self,
                    mask: torch.Tensor,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    average: str = 'batch',
                    contexts: torch.Tensor = None,
                    context_inputs: torch.Tensor = None):

        loss_ce = sequence_cross_entropy_with_logits(
            logits, labels, mask, average=average, label_smoothing=self._label_smoothing)

        return loss_ce 

    def _kldistance(self, 
                    z: torch.Tensor, 
                    q_params: Tuple[torch.Tensor, torch.Tensor],
                    p_params: Tuple[torch.Tensor, torch.Tensor] = None):
        q_mu, q_std = q_params
        lq_z = self.sampler.lprob(z, q_mu, q_std)

        if p_params is None:
            lp_z = self.sampler.lprob(z)
        else:
            p_mu, p_std = p_params
            lp_z = self.sampler.lprob(z, p_mu, p_std)

        kl = lq_z - lp_z
        return kl



