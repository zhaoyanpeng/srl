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

@Model.register("srl_graph_ae")
class SrlGraphAutoencoder(Model):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder, # a latent z
                 decoder: Seq2SeqEncoder, # a sequence of recovered inputs (e.g., arguments & contexts)
                 sampler: Seq2VecEncoder, # posterior distribution
                 kl_alpha: float = 1.0,   # kl weight
                 ll_alpha: float = 1.0,   # ll weight
                 nsample: int = 1,        # # of samples from the posterior distribution 
                 label_smoothing: float = None, # 
                 b_use_z: bool = True,
                 b_ctx_predicate: bool = False, # b: boolean value
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlGraphAutoencoder, self).__init__(vocab, regularizer)
        self.signature = 'graph'
        
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.nsample = nsample
        self._label_smoothing = label_smoothing
        self._b_ctx_predicate = b_ctx_predicate
        self._b_use_z = b_use_z

        self.kl_alpha = kl_alpha
        self.ll_alpha = ll_alpha
        self.kl_loss = None
    
    def add_parameters(self, nlabel: int, nlemma: int, lemma_embedder_weight: torch.Tensor):
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
        batch_size, nnode = mask.size()
        # encode graph into 'z'
        z = self.encoder(mask, embedded_nodes, edge_types, edge_type_onehots)
        # sample a 'z' (batch_size, nsample, dim)
        z = self.sampler(z, nsample = self.nsample)
        # parameters of the distribution from which 'z' is sampled
        z_mu, z_std = self.sampler.mu, self.sampler.std
        
        # KL(q(z|args, labels, predicates) || p(z))
        # kld = self._kldistance(z, (z_mu, z_std)) 
        kld = self._kld_simple(z_mu, z_std) 
        # feature dimension, then sample dimension
        kld = torch.mean(torch.sum(kld, -1), -1) 
        self.kldistance = self.kl_alpha * kld 
        
        # (nsample, batch_size, n_node, n_lemma)
        embedded_predicates = embedded_nodes[:, [0], :]
        if self._b_ctx_predicate:
            embedded_predicates = self.encoder.embedded_predicates
        logits = self.decoder(z, embedded_edges, embedded_predicates)

        # reconstruction loss
        mask = mask.unsqueeze(0).expand(self.nsample, -1, -1)
        node_types = node_types.unsqueeze(0).expand(self.nsample, -1, -1)
        # 'expand' introduces noncontinuity, here we have to rely on 'contiguous'        
        llh = self._likelihood(mask.contiguous().view([-1, nnode]), 
                               logits.view([-1, nnode, logits.size(-1)]), 
                               node_types.contiguous().view([-1, nnode]),
                               average = None) 
        # average along sample dimension
        llh = torch.mean(llh.view([self.nsample, batch_size]), 0)
        self.likelihood = self.ll_alpha * llh 

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

    def _kld_simple(self, mu, std):
        var = torch.pow(std, 2) + 1e-15 # sanity check
        return -0.5 * (torch.log(var) - var - torch.pow(mu, 2) + 1) 

    def compute_mutual_info(self,
                            mask: torch.Tensor,
                            node_types: torch.Tensor,
                            embedded_nodes: torch.Tensor,  
                            edge_types: torch.Tensor,
                            embedded_edges: torch.Tensor = None,
                            edge_type_onehots: torch.Tensor = None,
                            contexts: torch.Tensor = None) -> torch.Tensor:
        batch_size, nnode = mask.size()
        # encode graph into 'z'
        z = self.encoder(mask, embedded_nodes, edge_types, edge_type_onehots)
        # sample a 'z' (batch_size, nsample, dim)
        z = self.sampler(z, nsample = self.nsample)
        # parameters of the distribution from which 'z' is sampled
        z_mu, z_std = self.sampler.mu, self.sampler.std
        
        
        ## 1st method 
        print()
        z_dim = z.size(-1)
        var = torch.pow(z_std, 2) + 1e-15 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Differential_entropy
        # entropy = 0.5 * z_dim * (np.log(2 * np.pi) + 1) + 0.5 * torch.log(var).sum(-1)
        entropy = 0.5 * z_dim * np.log(2 * np.pi) + 0.5 * (1 + torch.log(var)).sum(-1) 
        # negative entropy; along z dim, then batch_size & nsample 
        entropy = -torch.mean(entropy) 
        
        lq_z = self.sampler.lprob(z, z_mu, z_std)
        lq_z = torch.sum(lq_z, -1) 
        #print(lq_z, lq_z.size())

        #lq_z = -0.5 * (z_dim * np.log(2 * np.pi) + torch.log(var).sum(-1)) \
        #       -0.5 * (torch.pow(z - z_mu, 2) / var).sum(-1) 
        #print(lq_z, lq_z.size())

        # along z dim, then batch_size dim, then nsample
        lq_z = torch.logsumexp(lq_z, 0) - np.log(batch_size) 
        #lq_z = torch.exp(lq_z) * lq_z
        lq_z = torch.mean(lq_z)
        
        print("\n\n{} - {}\n\n".format(entropy, lq_z))
        mi = entropy - lq_z

        import sys
        sys.exit(0)
        
        """
        ## 3rd method 
        print()
        z_dim = z.size(-1)
        var = torch.pow(z_std, 2) + 1e-15 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Differential_entropy
        # entropy = 0.5 * z_dim * (np.log(2 * np.pi) + 1) + 0.5 * torch.log(var).sum(-1)
        entropy = 0.5 * z_dim * np.log(2 * np.pi) + 0.5 * (1 + torch.log(var)).sum(-1) 
        # negative entropy; along z dim, then batch_size & nsample 
        entropy = -torch.mean(entropy) 
        
        #lq_z = self.sampler.lprob(z, z_mu, z_std)
        #lq_z = torch.sum(lq_z, -1) 
        #print(lq_z, lq_z.size())

        z = z.unsqueeze(2)
        var = var.transpose(0, 1).unsqueeze(0)
        z_mu = z_mu.transpose(0, 1).unsqueeze(0)
        #print(z.size(), var.size(), z_mu.size())

        lq_z = -0.5 * (z_dim * np.log(2 * np.pi) + torch.log(var).sum(-1)) \
               -0.5 * (torch.pow(z - z_mu, 2) / var).sum(-1) 

        # along z dim, then batch_size dim, then nsample
        lq_z = torch.logsumexp(lq_z, -1) - np.log(batch_size) 
        lq_z = torch.mean(lq_z)
        
        #print("\n\n{} - {}\n\n".format(entropy, lq_z))
        mi = entropy - lq_z

        #import sys
        #sys.exit(0)
        """
        
        """
        ## Another method
        print()
        kl_0 = self._kld_simple(z_mu, z_std)
        kl_0 = torch.mean(torch.sum(kl_0, -1))

        lq_z = self.sampler.lprob(z, z_mu, z_std)
        lp_z = self.sampler.lprob(z)

        lq_z = torch.sum(lq_z, -1)
        lp_z = torch.sum(lp_z, -1)
        
        lq_z = torch.logsumexp(lq_z, 0) - np.log(batch_size)
        kl_1 = torch.mean(lq_z) - torch.mean(lp_z)

        print(kl_1)

        lp_z = torch.logsumexp(lp_z, 0) - np.log(batch_size)
        kl_1 = torch.mean(lq_z - lp_z)
        
        print(kl_1)

        #kl_1 = lq_z - lp_z
        #kl_1 = torch.logsumexp(kl_1, 0) - np.log(batch_size)
        #kl_1 = torch.mean(kl_1)
        #print(kl_1)

        mi = kl_0 - kl_1
        
        print("\n\n{} - {}\n\n".format(kl_0, kl_1))

        #print(lq_z, lq_z.size())
        #print(lp_z, lp_z.size())
        print(mi)
        
        #import sys
        #sys.exit(0)
        """

        return mi, batch_size

