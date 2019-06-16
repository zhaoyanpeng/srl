from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder

@Model.register("sri_gan_dis")
class SrlGanDiscriminator(Model):
    def __init__(self, vocab: Vocabulary,
                 lemma_embedder: TextFieldEmbedder,
                 label_embedder: Embedding, 
                 predt_embedder: Embedding, # predt: predicate
                 encoder: Seq2VecEncoder, # a latent z
                 embedding_dropout: float = 0.,
                 label_smooth_val: float = None,
                 feature_matching: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlGanDiscriminator, self).__init__(vocab, regularizer)
        
        self.encoder = encoder
        self.lemma_embedder = lemma_embedder
        self.label_embedder = label_embedder
        self.predt_embedder = predt_embedder
        self.embedding_dropout = Dropout(p=embedding_dropout)

        self.label_smooth_val = label_smooth_val
        self.feature_matching = feature_matching
        self.feature_retained = None
        
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
        if caching_feature_only: # for feature matching
            self.feature_retained = torch.mean(features, 0)
            #print('------------------------------------- GEN: cache features')
            return None
        
        if optimizing_generator and self.feature_matching:
            # generator loss: using feature matching
            diff = torch.mean(features, 0) - self.feature_retained 
            fm_loss = torch.sum(diff ** 2)
            #print('--------------------------------------- GEN: fm loss')
            return fm_loss
        elif optimizing_generator:
            # generator loss: sigmoid loss
            logits = self.logitor(features).squeeze(-1) 
            labels = mask[:, 0].detach().clone().fill_(1).float()

            loss_ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
            #print('--------------------------------------- GEN: fake input')
            return loss_ce
        elif not optimizing_generator and relying_on_generator:
            # discriminator loss: fake data part
            logits = self.logitor(features).squeeze(-1) 
            labels = mask[:, 0].detach().clone().fill_(0).float()

            loss_ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

            #print('--------------------------------------- DIS: fake input')
            return loss_ce
        elif not optimizing_generator:
            # discriminator loss: real data part
            logits = self.logitor(features).squeeze(-1) 
            labels = mask[:, 0].detach().clone().fill_(0).float()
            
            noise = 1. if not self.label_smooth_val else self.label_smooth_val
            if noise > 1.: # using random noise
                noise = torch.empty(mask.size(0), device=mask.device).uniform_(0.7, 1.2) 
            labels += noise 

            loss_ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
            #print('--------------------------------------- DIS: real input')
            return loss_ce
        else:
            return None

    def gradient_penalty(self, arg1, arg2, arg3):
        return None

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
