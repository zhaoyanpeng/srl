from typing import Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import TextFieldEmbedder
from allennlp.common.checks import ConfigurationError

from nlpmimic.nn.util import gumbel_softmax

@Seq2SeqEncoder.register("srl_lstms_decoder")
class SrlLstmsDecoder(Seq2SeqEncoder):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = None,
                 b_ignore_z: bool = False, # b: bool
                 always_use_z: bool = False,
                 rnn_cell_type: str = 'gru',
                 straight_through: bool = True,
                 always_use_predt: bool = False,
                 dense_layer_dims: List[int] = None,
                 dropout: float = 0.0) -> None:
        super(SrlLstmsDecoder, self).__init__()
        self.signature = 'srl_lstms_decoder'

        self._input_dim = input_dim 
        self._hidden_dim = hidden_dim
        self._b_ignore_z = b_ignore_z
        self._always_use_z = always_use_z
        self._dense_layer_dims = dense_layer_dims 
        self._straight_through = straight_through
        self._always_use_predt = always_use_predt
        
        if rnn_cell_type == 'gru':
            self._rnn = torch.nn.GRUCell(self._input_dim, self._hidden_dim)
        elif rnn_cell_type == 'rnn':
            self._rnn = torch.nn.RNNCell(self._input_dim, self._hidden_dim)
        elif rnn_cell_type == 'lstm':
            self._rnn = torch.nn.LSTMCell(self._input_dim, self._hidden_dim)
        else:
            raise ConfigurationError('Invalid rnn cell type.')
        
        self._dense_layers = []
        input_dim = hidden_dim # output of rnn
        if dense_layer_dims is not None:
            for i_layer, dim in enumerate(self._dense_layer_dims):
                dense_layer = torch.nn.Linear(input_dim, dim, bias=True)
                setattr(self, 'dense_layer_{}'.format(i_layer), dense_layer)
                self._dense_layers.append(dense_layer)
                input_dim = dim
        
        self._dropout = torch.nn.Dropout(dropout)
    
    def add_parameters(self, output_dim: int, lemma_embedder: TextFieldEmbedder, nlabel: int = None) -> None: 
        self._output_dim = output_dim 

        lemma_embedder = getattr(lemma_embedder, 'token_embedder_{}'.format('lemmas'))
        self._lemma_embedder_weight = lemma_embedder.weight 
        
        input_dim = self._hidden_dim
        if len(self._dense_layers) > 1: 
            input_dim = self._dense_layer_dims[-1] 
        label_layer = torch.nn.Linear(input_dim, self._output_dim)
        setattr(self, 'label_projection_layer', label_layer)

    def forward(self, 
                z: torch.Tensor,
                embedded_labels: torch.Tensor,
                embedded_predicates: torch.Tensor,
                nodes_contexts: torch.Tensor = None) -> torch.Tensor:
        batch_size, ntimestep, _ = embedded_labels.size()

        if not self._b_ignore_z and z is not None: # (batch_size, nsample, dim)
            _, nsample, z_dim = z.size()
            _, nnode, label_dim = embedded_labels.size()
            predicate_dim = embedded_predicates.size(-1)
            
            embedded_predicates = embedded_predicates.expand(-1, nsample, -1).transpose(0, 1)
            embedded_predicates = embedded_predicates.contiguous().view([-1, predicate_dim])

            z = z.transpose(0, 1).contiguous().view([-1, z_dim]) # (nsample * batch_size, dim)
            
            embedded_labels = embedded_labels.unsqueeze(0).expand(nsample, -1, -1, -1)
            embedded_labels = embedded_labels.contiguous().view([-1, nnode, label_dim])
            
            if self._hidden_dim < z_dim:
                raise ConfigurationError('Dim of the latent semantics is too large for RNN hidden states.')

            new_batch_size = batch_size = z.size(0) # nsample * batch_size

            n_z_dim = self._hidden_dim // z_dim
            n_zeros = self._hidden_dim % z_dim
            hx_z = [z for _ in range(n_z_dim)]
            hx_zero = [torch.zeros((new_batch_size, n_zeros), device=embedded_labels.device)]
            hx = torch.cat(hx_z + hx_zero, -1) # initial hidden states 
        else:
            hx = torch.zeros((batch_size, self._hidden_dim), device=embedded_labels.device)
            if embedded_predicates.dim() == 3:
                embedded_predicates = embedded_predicates.squeeze(1)
            nsample = 1

        logits = []
        # predicates and labels will be concatenated together
        embedded_args = dummy = torch.zeros((nsample * batch_size, 0), device=embedded_labels.device)
        if self._always_use_predt:
            embedded_args = torch.zeros_like(embedded_predicates)
        for i in range(ntimestep):
            if nodes_contexts is not None:
                embedded_args = nodes_contexts[:, i, :]

            if self._always_use_z and z is not None:
                input_data = [embedded_predicates, z, embedded_args, embedded_labels[:, i, :]] 
            else:
                input_data = [embedded_predicates, embedded_args, embedded_labels[:, i, :]]

            input_i = torch.cat(input_data, -1) 
            hx = self._rnn(input_i, hx) 
            embedded_args, logits_i = self.customize_hidden_states(hx)

            logits.append(logits_i)
            if not self._always_use_predt:
                embedded_predicates = dummy 
        logits = torch.stack(logits, 1) # (batch_size, nnode, nlemma)

        return logits 
   
    def customize_hidden_states(self, hidden_states: torch.Tensor, nodes_contexts: torch.Tensor = None):
        hidden_states = self.multi_dense(hidden_states)
        logits = self.label_projection_layer(hidden_states)

        hidden_states = None
        if nodes_contexts is None:
            gumbel_hard, gumbel_soft, _ = gumbel_softmax(logits)
            label_relaxed = gumbel_hard if self._straight_through else gumbel_soft
            hidden_states = torch.matmul(label_relaxed, self._lemma_embedder_weight) 

        return hidden_states, logits

    def multi_dense(self, embedded_nodes: torch.Tensor) -> torch.Tensor:
        for dense_layer in self._dense_layers:
            embedded_nodes = torch.tanh(dense_layer(embedded_nodes))
            embedded_nodes = self._dropout(embedded_nodes)
        return embedded_nodes
    
    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

