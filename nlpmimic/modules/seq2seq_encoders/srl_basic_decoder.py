from typing import Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode


@Seq2SeqEncoder.register("srl_basic_decoder")
class SrlBasicDecoder(Seq2SeqEncoder):

    def __init__(self, 
                 input_dim: int, 
                 dense_layer_dims: List[int],
                 bilinear_type: int = None,
                 hidden_dim: int = None,
                 dropout: float = 0.0) -> None:
        super(SrlBasicDecoder, self).__init__()
        self.signature = 'srl_basic_decoder'

        self._input_dim = input_dim 
        self._bilinear_type = bilinear_type
        self._dense_layer_dims = dense_layer_dims 

        if self._bilinear_type is not None:
            self._input_dim //= 2
            self._dense_layer_dims = []
        
        self._dense_layers = []
        for i_layer, dim in enumerate(self._dense_layer_dims):
            dense_layer = torch.nn.Linear(input_dim, dim, bias=True)
            setattr(self, 'dense_layer_{}'.format(i_layer), dense_layer)
            self._dense_layers.append(dense_layer)
            input_dim = dim
        
        self._dropout = torch.nn.Dropout(dropout)
    
    def add_parameters(self, output_dim: int, lemma_embedder_weight: torch.Tensor, nlabel: int = None) -> None:
        self._nlabel = nlabel
        self._output_dim = output_dim 
        self._lemma_embedder_weight = lemma_embedder_weight 

        if not self._bilinear_type:
            label_layer = torch.nn.Linear(self._dense_layer_dims[-1], self._output_dim)
            setattr(self, 'label_projection_layer', label_layer)
        else:
            output_dim = self._output_dim
            if self._bilinear_type == 1:
                output_dim = self._nlabel 
            dense_layer = torch.nn.Bilinear(self._input_dim, self._input_dim, output_dim, bias=False) 
            setattr(self, 'bilinear_layer', dense_layer)
            dense_layer = torch.nn.Linear(self._input_dim, output_dim) 
            setattr(self, 'linear_layer', dense_layer)

    def forward(self, 
                z: torch.Tensor,
                embedded_edges: torch.Tensor,
                embedded_predicates: torch.Tensor,
                edge_types: torch.Tensor = None) -> torch.Tensor:
        nnode = embedded_edges.size(1)
        embedded_nodes = []

        if z is not None:
            nsample = z.size(1) 
            # (nsample, batch_size, nnode, dim)
            z = z.transpose(0, 1).unsqueeze(2).expand(-1, -1, nnode, -1) 
            embedded_nodes.append(z)
        else:
            nsample = 1

        embedded_edges = embedded_edges.unsqueeze(0).expand(nsample, -1, -1, -1) 
        embedded_nodes.append(embedded_edges)

        if embedded_predicates is not None: # must not be None
            embedded_predicates = embedded_predicates.unsqueeze(0).expand(nsample, -1, nnode, -1)
            embedded_nodes.append(embedded_predicates)

        if not self._bilinear_type:
            embedded_nodes = torch.cat(embedded_nodes, -1)
            embedded_nodes = self.multi_dense(embedded_nodes)
            logits = self.label_projection_layer(embedded_nodes)
        elif self._bilinear_type == 1: #  general matrix: annoying case 
            # (1, bsize, nnode, dim)
            #shape = embedded_edges.size() 
            nlemma = self._lemma_embedder_weight.size(0)

            interval = 10
            nslice = nlemma // interval 
            if nlemma % interval > 0:
                nslice += 1
            #print(nlemma, interval)
            rets = []
            for i in range(nslice):
                beg = i * interval
                end = (i + 1) * interval
                lemma_vectors = self._lemma_embedder_weight[beg : end]
                nlemma = lemma_vectors.size(0)
                #print(lemma_vectors.size())

                w_b = self.bilinear_layer.weight.unsqueeze(1) # no bias
                w_l = lemma_vectors.unsqueeze(0).unsqueeze(-1)

                l_w_b = torch.matmul(w_b, w_l).transpose(-1, -2).unsqueeze(1)

                w_p = embedded_nodes[1][:1, :, [0], :].unsqueeze(-2).transpose(-1, -2) 
                b_logits = torch.matmul(l_w_b, w_p).squeeze(-1).squeeze(-1)
                b_logits = b_logits.transpose(0, 1)
                l_logits = self.linear_layer(lemma_vectors)
                l_logits = l_logits.transpose(0, 1)

                logits = b_logits + l_logits.unsqueeze(0)

                if edge_types is not None:
                    indices = edge_types.unsqueeze(-1).expand(-1, -1, nlemma)
                    logits = torch.gather(logits, 1, indices)
                logits = logits.unsqueeze(0)
                #print(logits.size())

                rets.append(logits)
            logits = torch.cat(rets, -1)
            #print(logits.size())
            """
            if edge_types is not None:
                a = embedded_nodes[1].unsqueeze(-2).expand(-1, -1, -1, nlemma, -1) 
                b = self._lemma_embedder_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)\
                    .expand(shape[0], shape[1], shape[2], -1, -1)

                shape = a.size() 
                a = a.contiguous().view(shape)
                b = b.contiguous().view(shape)
                b_logits = self.bilinear_layer(a, b)
                l_logits = self.linear_layer(self._lemma_embedder_weight)

                l_logits = l_logits.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                logits = b_logits + l_logits
            
                indices = edge_types.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)\
                    .expand(shape[0], -1, -1, nlemma, -1) 
                logits = torch.gather(logits, -1, indices).squeeze(-1)
            else:
                a = embedded_nodes[1][:1, :, [0], :].expand(-1, -1, nlemma, -1) 
                b = self._lemma_embedder_weight.unsqueeze(0).unsqueeze(0)\
                    .expand(shape[0], shape[1], -1, -1)

                shape = a.size() 
                a = a.contiguous().view(shape)
                b = b.contiguous().view(shape)
                b_logits = self.bilinear_layer(a, b)
                l_logits = self.linear_layer(self._lemma_embedder_weight)

                l_logits = l_logits.unsqueeze(0).unsqueeze(0)
                logits = b_logits + l_logits
                logits = torch.transpose(logits, -1, -2)
            """
        elif self._bilinear_type == 2: # diagonal matrix: embedded_edges.diag() 
            pass
        else:
            shape = embedded_edges.size()
            embedded_nodes[0] = embedded_nodes[0].contiguous().view(shape)
            embedded_nodes[1] = embedded_nodes[1].contiguous().view(shape)
            b_logits = self.bilinear_layer(embedded_nodes[0], embedded_nodes[1])
            l_logits = self.linear_layer(embedded_edges[0])

            logits = b_logits + l_logits
            
        return logits 
    
    def multi_dense(self, embedded_nodes: torch.Tensor) -> torch.Tensor:
        for dense_layer in self._dense_layers:
            embedded_nodes = torch.tanh(dense_layer(embedded_nodes))
            embedded_nodes = self._dropout(embedded_nodes)
        return embedded_nodes
    
    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

