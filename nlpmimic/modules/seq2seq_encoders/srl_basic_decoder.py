"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""
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
                 dropout: float = 0.0) -> None:
        super(SrlBasicDecoder, self).__init__()
        self.signature = 'srl_basic_decoder'

        self._input_dim = input_dim 
        self._dense_layer_dims = dense_layer_dims 
        
        self._dense_layers = []
        for i_layer, dim in enumerate(self._dense_layer_dims):
            dense_layer = torch.nn.Linear(input_dim, dim, bias=True)
            setattr(self, 'dense_layer_{}'.format(i_layer), dense_layer)
            self._dense_layers.append(dense_layer)
            input_dim = dim
        
        self._dropout = torch.nn.Dropout(dropout)
    
    def add_parameters(self, output_dim: int) -> None:
        self._output_dim = output_dim 
        label_layer = torch.nn.Linear(self._dense_layer_dims[-1], self._output_dim)
        setattr(self, 'label_projection_layer', label_layer)

    def forward(self, 
                embedded_edges: torch.Tensor,
                embedded_predicates: torch.Tensor = None) -> torch.Tensor:
        nnode = embedded_edges.size(1)
        # (batch_size, nnode, dim)
        embedded_nodes = embedded_edges 
        if embedded_predicates is not None:
            embedded_predicates = embedded_predicates.expand(-1, nnode, -1)
            embedded_nodes = torch.cat([embedded_nodes, embedded_predicates], -1)

        embedded_nodes = self.multi_dense(embedded_nodes)
        logits = self.label_projection_layer(embedded_nodes)
        
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

