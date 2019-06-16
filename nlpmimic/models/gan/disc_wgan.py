from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from torch import autograd
from torch.nn.modules import Dropout

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder

@Model.register("sri_wgan_dis")
class SrlWganDiscriminator(Model):
    def __init__(self, vocab: Vocabulary,
                 lemma_embedder: TextFieldEmbedder,
                 label_embedder: Embedding, 
                 predt_embedder: Embedding, # predt: predicate
                 encoder: Seq2VecEncoder, # a latent z
                 embedding_dropout: float = 0.,
                 b_use_wgan: bool = True, # b bool
                 grad_penal_weight: float = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlWganDiscriminator, self).__init__(vocab, regularizer)
        
        self.encoder = encoder
        self.lemma_embedder = lemma_embedder
        self.label_embedder = label_embedder
        self.predt_embedder = predt_embedder
        self.embedding_dropout = Dropout(p=embedding_dropout)

        self.b_use_wgan = True
        self.feature_retained = None
        self.grad_penal_weight = grad_penal_weight
        
        self.logitor = torch.nn.Linear(self.encoder.get_output_dim(), 1, bias=True)
    
    def add_parameters(self, nlabel: int, suppress_nonarg: bool):
        self.suppress_nonarg = suppress_nonarg
        self.encoder.add_parameters(nlabel)

    def forward(self, 
                mask: torch.Tensor,
                embedded_nodes: torch.Tensor,  
                edge_types: torch.Tensor,
                optimizing_generator: bool,
                relying_on_generator: bool,
                caching_feature_only: bool = False,
                edge_type_softmax: torch.Tensor = None,
                edge_type_onehots: torch.Tensor = None,
                contexts: torch.Tensor = None) -> torch.Tensor:
        features = self.encoder(mask, embedded_nodes, edge_types, edge_type_onehots=edge_type_onehots)
        if optimizing_generator:
            # generator loss: sigmoid or logits loss 
            logits = self.logitor(features).squeeze(-1) 
            if not self.b_use_wgan:
                labels = mask[:, 0].detach().clone().fill_(1).float()
                loss_ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
            else:
                loss_ce = torch.mean(logits) # minimize confidence 
                #loss_ce = -torch.mean(logits) # maximize confidence gen 
            #print('--------------------------------------- GEN: fake input')
            return loss_ce
        elif not optimizing_generator and relying_on_generator:
            # discriminator loss: fake data part
            logits = self.logitor(features).squeeze(-1) 
            if not self.b_use_wgan:
                labels = mask[:, 0].detach().clone().fill_(0).float()
                loss_ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
            else:
                loss_ce = -torch.mean(logits) # maximize confidence
                #loss_ce = torch.mean(logits) # minimize confidence fake 
            if self.gradient_penalty is not None: # need to cache nominal data
                self.feature_retained = {'noun_mask': mask,
                                         'noun_embedded_nodes': embedded_nodes,
                                         'noun_edge_types': edge_types,
                                         'noun_edge_type_softmax': edge_type_softmax,
                                         'noun_edge_type_onehots': edge_type_onehots} 
            #print('--------------------------------------- DIS: fake input')
            return loss_ce
        elif not optimizing_generator:
            # discriminator loss: real data part
            logits = self.logitor(features).squeeze(-1) 
            if not self.b_use_wgan:
                labels = mask[:, 0].detach().clone().fill_(0).float()
                loss_ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
            else:
                loss_ce = torch.mean(logits) # minimize confidence
                #loss_ce = -torch.mean(logits) # maximize confidence real  
            #print('--------------------------------------- DIS: real input')
            return loss_ce
        else:
            return None

    def gradient_penalty(self,
                         verb_mask: torch.Tensor,
                         verb_embedded_nodes: torch.Tensor,  
                         verb_edge_types: torch.Tensor) -> torch.Tensor:
        if not self.grad_penal_weight or not self.feature_retained: 
            return None
        # expand cached data
        noun_mask = self.feature_retained['noun_mask']
        noun_embedded_nodes = self.feature_retained['noun_embedded_nodes']
        noun_edge_types = self.feature_retained['noun_edge_types']
        noun_edge_type_softmax = self.feature_retained['noun_edge_type_softmax']

        batch_size, num_nodes = noun_mask.size()
        # interpolation
        alpha = torch.rand(batch_size, device=noun_mask.device)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        i = torch.arange(batch_size).unsqueeze(-1) # (batch_size, 1)
        j = torch.arange(num_nodes).unsqueeze(0)   # (1, num_nodes)

        edge_type_softmax = noun_edge_type_softmax.detach()
        edge_type_softmax = edge_type_softmax * (1 - alpha) 
        #print(i.size(), j.size(), verb_edge_types.size(), edge_type_softmax.size())

        edge_type_softmax[i, j, verb_edge_types] += alpha.squeeze(-1)
        edge_type_softmax *= noun_mask.unsqueeze(-1).float()
        edge_type_softmax.requires_grad_(True)
        
        embedded_nodes = verb_embedded_nodes.detach() * alpha + \
                         noun_embedded_nodes.detach() * (1 - alpha)
        embedded_nodes[:, 1:, :] *= noun_mask.unsqueeze(-1).float()
        embedded_nodes.requires_grad_(True)

        features = self.encoder(noun_mask, embedded_nodes, noun_edge_types, edge_type_onehots=edge_type_softmax)
        logits = self.logitor(features).squeeze(-1) 

        #pseudo_loss = -torch.mean(logits) # maximize confidence 
        gradients, lemma_grads = autograd.grad(outputs=logits,
                                  inputs=[edge_type_softmax, embedded_nodes],
                                  grad_outputs=torch.ones(logits.size(), device=noun_mask.device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)

        effective_grads = gradients
        effective_grads = effective_grads.view(batch_size, -1)
        # to avoid float underflowing
        gradients_norm = torch.sqrt(torch.sum(effective_grads ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() 
        #gradient_penalty = ((effective_grads.norm(2, dim=1) - 1) ** 2).mean() 
        
        #print('--------------------{}'.format(gradient_penalty))
        effective_grads = lemma_grads
        effective_grads = effective_grads.view(batch_size, -1)
        # to avoid float underflowing
        gradients_norm = torch.sqrt(torch.sum(effective_grads ** 2, dim=1) + 1e-12)
        gradient_penalty += ((gradients_norm - 1) ** 2).mean() 
        #gradient_penalty += ((effective_grads.norm(2, dim=1) - 1) ** 2).mean() 
        
        #print('++++++++++++++++++++{}'.format(gradient_penalty))
        gradient_penalty = self.grad_penal_weight * gradient_penalty / 2.
        return gradient_penalty

    def embed_labels(self,
                     labels: torch.Tensor,
                     labels_add_one: bool = False,
                     labels_relaxed: torch.Tensor = None):
        """ assumming there can be an additional non-argument label
            labels_add_one: increase label indices by 1
        """
        if not self.suppress_nonarg:
            label_embeddings = self.label_embedder.weight
        else: # discard the embedding of the non-argument label
            label_embeddings = self.label_embedder.weight[1:, :]

        if labels_relaxed is not None:
            encoded_labels = torch.matmul(labels_relaxed, label_embeddings) 
        elif labels is not None:  
            if labels_add_one:
                labels = labels + 1
            encoded_labels = self.label_embedder(labels)
        else:
            encoded_labels = None
        encoded_labels = self.embedding_dropout(encoded_labels)
        return encoded_labels

    def encode_args(self,
                    lemmas: Dict[str, torch.LongTensor],
                    predicates: torch.Tensor,
                    predicate_sign: torch.LongTensor,
                    arg_indices: torch.LongTensor,
                    embedded_seqs: torch.Tensor = None):
        embedded_predicates = self.predt_embedder(predicates)
        # (batch_size, length, dim) -> (batch_size, dim, length)
        embedded_predicates = torch.transpose(embedded_predicates, 1, 2)
        # (batch_size, length, 1)
        psigns = torch.unsqueeze(predicate_sign.float(), -1) 
        # (batch_size, dim, 1); select the predicate embedding
        embedded_predicates = torch.bmm(embedded_predicates, psigns)
         
        embedded_lemmas = self.lemma_embedder(lemmas)

        lemma_dim = embedded_lemmas.size()[-1]
        arg_indices = arg_indices.unsqueeze(-1).expand(-1, -1, lemma_dim)
        
        embedded_arg_lemmas = torch.gather(embedded_lemmas, 1, arg_indices)
        embedded_predicates = embedded_predicates.transpose(1, 2)
        embedded_nodes = torch.cat([embedded_predicates, embedded_arg_lemmas], 1)

        if embedded_seqs is not None:
            narg = arg_indices.size(1)
            embedded_seqs = embedded_seqs.unsqueeze(1).expand(-1, narg + 1, -1)
            embedded_nodes = torch.cat([embedded_nodes, embedded_seqs], -1)
        
        # apply dropout to predicate embeddings
        embedded_nodes = self.embedding_dropout(embedded_nodes)
        return embedded_nodes
