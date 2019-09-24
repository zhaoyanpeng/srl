from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed 
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from nlpmimic.modules.seq2vec_encoders.sampler import SamplerUniform

@Model.register("srl_ae_hub_z")
class SrlHubzAutoencoder(Model):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder = None, # a latent z
                 decoder: Seq2SeqEncoder = None, # a sequence of recovered inputs (e.g., arguments & contexts)
                 sampler: Seq2VecEncoder = None, # posterior distribution
                 kl_alpha: float = 1.0,   # kl(q(z) | p(z)) 
                 ky_alpha: float = 1.0,   # log p(y) 
                 ky_prior: str = 'uniform', # uniform or gumbel
                 ll_alpha: float = 1.0,   # ll weight
                 re_alpha: float = 0.5,   # regularizer for duplicate roles in an instance
                 re_type: str = 'relu',   # type of regularizer: kl or relu
                 n_sample: int = 1,       # # of samples from the posterior distribution 
                 b_z_mean: bool = False,  # use the mean value not the sampled ones
                 b_ctx_argument: bool = False,  # contextualized arguments 
                 b_ctx_predicate: bool = False, # b: boolean value
                 label_smoothing: float = None, # in the generative model
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlHubzAutoencoder, self).__init__(vocab, regularizer)
        self.signature = 'srl_ae_hub_z'
        
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.re_type = re_type
        self._label_smoothing = label_smoothing
        self._b_ctx_predicate = b_ctx_predicate
        self._b_ctx_argument = b_ctx_argument
        self._b_z_mean = b_z_mean
        
        self.n_sample = n_sample
        self.kl_alpha = kl_alpha
        self.ky_alpha = ky_alpha
        self.ky_prior = ky_prior
        self.ll_alpha = ll_alpha
        self.re_alpha = re_alpha

        self.kldistance = None 
        self.likelihood = None 
    
    def add_parameters(self, nlabel: int, nlemma: int, lemma_embedder: TextFieldEmbedder):
        self.nlabel = nlabel # needed in computing log p(y) = \sum log p(y_i)
        self.nlemma = nlemma # 
        if self.encoder is not None:
            self.encoder.add_parameters(nlabel)
        if self.decoder is not None:
            self.decoder.add_parameters(nlemma, lemma_embedder)

    def anneal_kl(self, kl_alpha: float=1., ky_alpha: float=1.):
        self.kl_alpha = kl_alpha
        self.ky_alpha = ky_alpha
        #print('-------------------------', self.kl_alpha, self.ky_alpha)

    def forward(self, 
                mask: torch.Tensor,
                node_types: torch.Tensor,
                embedded_nodes: torch.Tensor,  
                edge_types: torch.Tensor,
                embedded_edges: torch.Tensor,
                ctx_lemmas: torch.Tensor,
                ctx_global: Tuple[torch.Tensor],
                edge_type_onehots: torch.Tensor = None,
                contexts: torch.Tensor = None) -> torch.Tensor:
        batch_size, nnode = mask.size()
        embedded_lemmas = embedded_nodes[:, 1:, :]
        embedded_predicates = embedded_nodes[:, [0], :]
        if ctx_global is not None:
            # contextualized arguments and predicates
            z_ctx, a_ctx, p_ctx = ctx_global
            # (batch_size, nsample, dim)
            z = self.sampler(z_ctx, nsample=self.n_sample) 
            z_mu, z_std = self.sampler.mu, self.sampler.std
            kld = self._kld_simple(z_mu, z_std) 
            # feature dimension, then sample dimension
            kld = torch.mean(torch.sum(kld, -1), -1) 
            self.kldistance = self.kl_alpha * kld 
            # optional choice, false by default
            if self._b_ctx_predicate:
                embedded_predicates = p_ctx
            if self._b_ctx_argument:
                embedded_lemmas = a_ctx
            if self._b_z_mean:
                z = z_mu
        else:
            self.kldistance = None 
            z = None


        context = None
        if ctx_lemmas is not None: 
            # be aware of other lemmas p(a_i | a-i)
            labeled_lemmas = torch.cat([embedded_lemmas, embedded_edges], -1)
            labeled_lemmas = labeled_lemmas * mask.unsqueeze(-1).float()
            # sum all the other labeled arguments 
            labeled_lemmas_sum = torch.sum(labeled_lemmas, -2, keepdim=True)
            context = labeled_lemmas_sum - labeled_lemmas
            # ctx may be all zero, adding a context lemma 
            divider = torch.sum(mask, -1, keepdim=True) - 1
            divider = torch.clamp(divider, min = 1).unsqueeze(-1).float()
            context = context / divider

            context[:, :, :ctx_lemmas.size(-1)] += ctx_lemmas

        mask = mask.unsqueeze(0).expand(self.n_sample, -1, -1)
        mask = mask.contiguous().view([-1, nnode])
        # reconstruction (argument) loss (batch_size,)
        logits = self.decoder(z, embedded_edges, embedded_predicates, nodes_contexts=context)
        logits = logits.view([-1, nnode, logits.size(-1)])


        # reconstruction (argument) loss (batch_size,)
        #logits = self.decoder(z, embedded_edges, embedded_predicates)
        llh = self._likelihood(mask, logits, node_types, average = None) 
        self.likelihood = self.ll_alpha * llh 

        # -L(x, y) = ELBO = E_{z ~ q}[log p(x | y, z) + log p(y) + log p(z) - log q(z | x, y)]
        elbo = -self.likelihood
        if self.kldistance is not None:
            self.kldistance.clamp_(max=100) # heuristics avoiding loss explosion
            elbo = elbo - self.kldistance
        return elbo 

    def _kld_simple(self, mu, std):
        var = torch.pow(std, 2) + 1e-15 # sanity check
        return -0.5 * (torch.log(var) - var - torch.pow(mu, 2) + 1) 

    def kld(self,
            posteriors = None, # log 
            mask: torch.Tensor = None,
            node_types: torch.Tensor = None,
            embedded_nodes: torch.Tensor = None,  
            embedded_edges: torch.Tensor = None,
            edge_type_onehots: torch.Tensor = None,
            contexts: torch.Tensor = None) -> torch.Tensor:
        if self.ky_prior == 'uniform': 
            # priors of role labels: log p(y) = \sum log p(y_i)
            y_logs = -torch.sum(mask, -1).float() * np.log(self.nlabel)
            if posteriors is not None:
                kl_loss = -posteriors + y_logs 
            else:
                kl_loss = y_logs # priors 
        elif self.ky_prior == 'gumbel':
            #euler = 0.5772
            ratio = self.sampler.tau_ratio
            gamma = math.gamma(1 + ratio)
            posteriors = ratio * posteriors * mask.unsqueeze(-1).float()
            #kl_loss = mast.sum() * (-math.log(ratio) - 1 + euler * (ratio - 1)
            kl_loss = posteriors.sum() + gamma * torch.exp(-posteriors).sum()
        else: # use estimated statistics of role labels as priors
            raise ValueError('Not supported.')
        kldistance = self.ky_alpha * kl_loss

        uniqueness = None
        # an additional regularizer for duplicate roles
        if edge_type_onehots is not None: 
            label_onehots = edge_type_onehots * mask.unsqueeze(-1).float()
            batch_probs = torch.sum(label_onehots, 1) # along argument dim
            if self.re_type == 'kl':
                batch_probs.clamp_(min = 1.)   # a trick to avoid nan = log(0)
                loss = torch.log(batch_probs) * batch_probs
                loss = torch.sum(loss, 1) # along label dim
            elif self.re_type == 'relu':
                loss = torch.relu(batch_probs - 1).sum() 
            # loss on the sentence level
            uniqueness = self.re_alpha * loss 
        return (kldistance, uniqueness) 

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

