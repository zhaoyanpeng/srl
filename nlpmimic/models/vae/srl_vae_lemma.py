from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode


@Model.register("srl_lemma_ae")
class SrlLemmaAutoencoder(Model):
    def __init__(self, vocab: Vocabulary,
                 decoder: Seq2SeqEncoder, # a sequence of recovered inputs (e.g., arguments & contexts)
                 sampler: Seq2VecEncoder = None, # posterior distribution
                 encoder: Seq2VecEncoder = None, # a latent z
                 nsample: int = 1,        # # of samples from the posterior distribution 
                 label_smoothing: float = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlLemmaAutoencoder, self).__init__(vocab, regularizer)
        self.signature = 'graph'
        
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.nsample = nsample
        self._label_smoothing = label_smoothing

        self.kl_loss = None
    
    def add_parameters(self, nlabel: int, nlemma: int, lemma_embedder_weight: torch.Tensor):
        if self.encoder is not None:
            self.encoder.add_parameters(nlabel)
        if self.decoder is not None:
            self.decoder.add_parameters(nlemma, lemma_embedder_weight, nlabel=nlabel)

    def forward(self, 
                mask: torch.Tensor,
                node_types: torch.Tensor,
                embedded_nodes: torch.Tensor,  
                edge_types: torch.Tensor,
                embedded_edges: torch.Tensor,
                edge_type_onehots: torch.Tensor = None,
                contexts: torch.Tensor = None) -> torch.Tensor:
        embedded_predicates = embedded_nodes[:, [0], :]
        # reconstruction (argument) loss (batch_size,)
        if embedded_edges is None: # stupid compatibility
            embedded_edges = embedded_nodes[:, 1:, :]

        if self.decoder.signature == 'srl_basic_decoder':
            logits = self.decoder(None, embedded_edges, embedded_predicates, edge_types)
        else:
            logits = self.decoder(None, embedded_edges, embedded_predicates)
        self.logits = logits.squeeze(0)

        self.kldistance = 0
        self.likelihood = 0 
        elbo = -self.likelihood - self.kldistance
        return elbo 
    
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

