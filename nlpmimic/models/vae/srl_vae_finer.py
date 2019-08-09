from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import math
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed 
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from nlpmimic.modules.seq2vec_encoders.sampler import SamplerGumbel, SamplerUniform

@Model.register("srl_finer_ae")
class SrlFinerAutoencoder(Model):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder = None, # a latent z
                 decoder: Seq2SeqEncoder = None, # a sequence of recovered inputs (e.g., arguments & contexts)
                 sampler: Seq2VecEncoder = None, # posterior distribution
                 kl_alpha: float = 1.0,   # kl weight
                 ll_alpha: float = 1.0,   # ll weight
                 re_alpha: float = 0.5,   # regularizer for duplicate roles in an instance
                 re_type: str = 'relu',       # type of regularizer: kl or relu
                 nsample: int = 1,        # # of samples from the posterior distribution 
                 generative_loss: str = 'crossentropy', # loss type for p(a | p, r); cs: crossentropy; mm: max-margin 
                 negative_sample: int = 10, # number of negative samples
                 label_smoothing: float = None,
                 b_use_z: bool = True,
                 b_ctx_predicate: bool = False, # b: boolean value
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlFinerAutoencoder, self).__init__(vocab, regularizer)
        self.signature = 'srl_lstms_ae'
        
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.nsample = nsample
        self.re_type = re_type
        self._generative_loss = generative_loss
        self._negative_sample = negative_sample
        self._label_smoothing = label_smoothing
        self._nsampling_power = 0.75
        self._b_ctx_predicate = b_ctx_predicate
        self._b_use_z = b_use_z
        
        self.kl_alpha = kl_alpha
        self.ll_alpha = ll_alpha
        self.re_alpha = re_alpha

        self.kldistance = None 
        self.likelihood = None 
    
    def add_parameters(self, nlabel: int, nlemma: int, lemma_embedder: TextFieldEmbedder):
        self.nlabel = nlabel
        self.nlemma = nlemma
        if self.encoder is not None:
            self.encoder.add_parameters(nlabel)

        self.lemma_embedder = lemma_embedder 
        nlemma = self.nlemma if self._generative_loss == 'cs' else lemma_embedder.get_output_dim() 
        if self.decoder is not None:
            self.decoder.add_parameters(nlemma, lemma_embedder)
        
        ns = "lemmas"
        self.nlemma = self.vocab.get_vocab_size(ns)
        # categorical distribution over arguments
        lemma_distr = torch.zeros((self.nlemma,))
        lemma_table = list(self.vocab._retained_counter[ns].items())
        for k, v in self.vocab._retained_counter[ns].items():
            lemma_index = self.vocab.get_token_index(k, namespace=ns)
            lemma_distr[lemma_index] = v
        self.lemma_distr = torch.pow(lemma_distr, self._nsampling_power)

    def forward(self, 
                mask: torch.Tensor,
                node_types: torch.Tensor,
                embedded_nodes: torch.Tensor,  
                edge_types: torch.Tensor,
                embedded_edges: torch.Tensor,
                ctx_lemmas: torch.Tensor,
                edge_type_onehots: torch.Tensor = None,
                contexts: torch.Tensor = None) -> torch.Tensor:
        batch_size, nnode = mask.size()
        # encoding them if it is needed
        if self._b_use_z or self._b_ctx_predicate:
            z = self.encoder(mask, embedded_nodes, edge_types, edge_type_onehots)
            z = z.unsqueeze(1)
        embedded_predicates = embedded_nodes[:, [0], :]
        if self._b_ctx_predicate:
            embedded_predicates = self.encoder.embedded_predicates
        if not self._b_use_z:
            z = None
        
        # be aware of other lemmas p(a_i | a-i)
        encoded_lemmas = embedded_nodes[:, 1:, :]
        labeled_lemmas = torch.cat([encoded_lemmas, embedded_edges], -1)
        labeled_lemmas = labeled_lemmas * mask.unsqueeze(-1).float()
        # sum all the other labeled arguments 
        labeled_lemmas_sum = torch.sum(labeled_lemmas, -2, keepdim=True)
        context = labeled_lemmas_sum - labeled_lemmas
        # ctx may be all zero, adding a context lemma 
        divider = torch.sum(mask, -1, keepdim=True) - 1
        divider = torch.clamp(divider, min = 1).unsqueeze(-1).float()
        context = context / divider

        context[:, :, :ctx_lemmas.size(-1)] += ctx_lemmas

        mask = mask.unsqueeze(0).expand(self.nsample, -1, -1)
        mask = mask.contiguous().view([-1, nnode])
        # gold arguments
        logits = self.decoder(z, embedded_edges, embedded_predicates, nodes_contexts=context)
        logits = logits.view([-1, nnode, logits.size(-1)])

        # reconstruction (argument) loss (batch_size,)
        if self._generative_loss == 'crossentropy':   # crossentropy loss by default
            node_types = node_types.unsqueeze(0).expand(self.nsample, -1, -1)
            node_types = node_types.contiguous().view([-1, nnode])

            llh = self._likelihood(mask, logits, node_types, average = None) 
            llh = torch.mean(llh.view([self.nsample, batch_size]), 0)
        elif self._generative_loss == 'maxmargin': # max-margin loss
            gold_lemmas = encoded_lemmas.unsqueeze(0).expand(self.nsample, -1, -1, -1)
            gold_lemmas = gold_lemmas.contiguous().view(-1, gold_lemmas.size(-1))

            logits = self.decoder(z, embedded_edges, embedded_predicates, nodes_contexts=context)
            logits = logits.view([-1, logits.size(-1)]).unsqueeze(-1)
            # (, 1, 1)
            gold_logits = torch.bmm(gold_lemmas.unsqueeze(-2), logits)
            gold_logits = F.logsigmoid(gold_logits).squeeze(-1)
            
            loss = 0
            nsample = batch_size * nnode
            distr = self.lemma_distr.to(device=logits.device)
            distr = distr.unsqueeze(0).expand(nsample, -1)
            samples = torch.multinomial(distr, self._negative_sample , replacement=False)
            samples = samples.view(batch_size, -1, self._negative_sample) 
            samples = (samples + 1 + node_types.unsqueeze(-1)) % self.nlemma
            samples = {"lemmas": samples}
            # (bsize, narg, nsample, dim)
            fake_lemmas = self.lemma_embedder(samples)
            fake_lemmas = fake_lemmas.unsqueeze(0).expand(self.nsample, -1, -1, -1, -1)
            fake_lemmas = fake_lemmas.contiguous().view(-1, self._negative_sample, fake_lemmas.size(-1))
            # (, nsample, 1)
            fake_logits = torch.bmm(fake_lemmas, logits)
            fake_logits = F.logsigmoid(-fake_logits).squeeze(-1)

            if (mask.sum(-1) == 0).any():
                raise ValueError("Empty argument set encountered.")

            loss = torch.cat([gold_logits, fake_logits], -1) 
            loss = loss * mask.view(-1, 1).float()
            loss = loss.sum(-1) # per-argument
            loss = loss.view(-1, nnode).sum(-1)
            loss = loss / mask.sum(-1).float() 
            llh = -torch.mean(loss.view([self.nsample, batch_size]), 0)

        self.likelihood = self.ll_alpha * llh 
        return -self.likelihood 

    def kld(self, posteriors, # logrithm 
            mask: torch.Tensor = None,
            node_types: torch.Tensor = None,
            embedded_nodes: torch.Tensor = None,  
            embedded_edges: torch.Tensor = None,
            edge_type_onehots: torch.Tensor = None,
            contexts: torch.Tensor = None) -> torch.Tensor:
        if isinstance(self.sampler, SamplerUniform): # should be the Gumbel distribution
            # priors of role labels
            kl_loss = 0
            if self.kl_alpha != 0 or True:
                nlabel = torch.tensor(self.nlabel, device=posteriors.device).float()
                pz_logs = -torch.sum(mask, -1).float() * torch.log(nlabel)
                kl_loss = posteriors - pz_logs
        elif isinstance(self.sampler, SamplerGumbel):
            #euler = 0.5772
            kl_loss = 0
            if self.kl_alpha != 0 or True:
                ratio = self.sampler.tau_ratio
                gamma = math.gamma(1 + ratio)
                posteriors = ratio * posteriors * mask.unsqueeze(-1).float()
                #kl_loss = mast.sum() * (-math.log(ratio) - 1 + euler * (ratio - 1)
                kl_loss = posteriors.sum() + gamma * torch.exp(-posteriors).sum()
        else: # use estimated statistics of role labels as priors
            #self.kldistance = self.kl_alpha * posteriors 
            raise ValueError('Not supported.')
        self.kldistance = self.kl_alpha * kl_loss

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
            # loss on sentence level
            uniqueness = self.re_alpha * loss 
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



