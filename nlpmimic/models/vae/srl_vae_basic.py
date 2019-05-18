from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.common.checks import ConfigurationError
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from nlpmimic.modules.seq2vec_encoders.sampler import SamplerUniform

@Model.register("srl_basic_ae")
class SrlBasicAutoencoder(Model):
    def __init__(self, vocab: Vocabulary,
                 decoder: Seq2SeqEncoder, # a sequence of recovered inputs (e.g., arguments & contexts)
                 sampler: Seq2VecEncoder = None, # posterior distribution
                 encoder: Seq2VecEncoder = None, # a latent z
                 alpha: float = 0.5,      # kl weight
                 nsample: int = 1,        # # of samples from the posterior distribution 
                 label_smoothing: float = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlBasicAutoencoder, self).__init__(vocab, regularizer)
        self.signature = 'graph'
        
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.nsample = nsample
        self._label_smoothing = label_smoothing

        self.alpha = alpha
        self.kl_loss = None
    
    def add_parameters(self, nlabel: int, nlemma: int):
        if self.encoder is not None:
            self.encoder.add_parameters(nlabel)
        self.decoder.add_parameters(nlemma)

    def forward(self, 
                mask: torch.Tensor,
                node_types: torch.Tensor,
                embedded_nodes: torch.Tensor,  
                edge_types: torch.Tensor,
                embedded_edges: torch.Tensor,
                edge_type_onehots: torch.Tensor = None,
                contexts: torch.Tensor = None) -> torch.Tensor:
        ### discarded, call 'encode' and 'decode' instead
        self.kldistance = None 
        self.likelihood = None 
        return None 

    def decode(self, 
               mask: torch.Tensor,
               node_types: torch.Tensor,
               embedded_nodes: torch.Tensor,  
               embedded_edges: torch.Tensor,
               contexts: torch.Tensor = None) -> torch.Tensor:
        batch_size, nnode = mask.size()

        # (batch_size, n_node, n_lemma)
        embedded_predicates = embedded_nodes[:, [0], :]
        logits = self.decoder(embedded_edges, embedded_predicates)

        # reconstruction loss
        llh = self._likelihood(mask, logits, node_types, average = None) 

        return -llh  

    def kld(self,
            posteriors,
            mask: torch.Tensor = None,
            node_types: torch.Tensor = None,
            embedded_nodes: torch.Tensor = None,  
            embedded_edges: torch.Tensor = None,
            contexts: torch.Tensor = None) -> torch.Tensor:
        if isinstance(self.sampler, SamplerUniform):
            return self.alpha * posteriors
        else:
            return self.alpha * posteriors
    
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

