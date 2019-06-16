from typing import Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed

@Seq2VecEncoder.register("srl_simple_encoder")
class SrlGraphEncoder(Seq2VecEncoder):
    """
    A srl encoder takes as input SRL features of predicate, arguments, and labels and 
    outputs its vector representation, which can be seen as the inference of p(z|x).

    This version does NOT have residual layers which use GRUCell and do not support
    twice-backward operation in Pytorch 1.1.0. However, twice-backward operation is
    required by WGAN model. The tradeoff is removing residual layers. An alternative
    is implementing our own GRUCell which supports twice-backward.

    Parameters
    ----------

    Returns
    -------
    z: float tensor  
        z = p(z|x) 
    """
    def __init__(self, 
                 input_dim: int,
                 layer_timesteps: List[int] = [2, 2, 2, 2],
                 residual_connection_layers: Dict[int, List[int]] = dict(),
                 dense_layer_dims: List[int] = [100], 
                 node_msg_dropout: float = 0.3,
                 residual_dropout: float = 0.3,
                 combined_vectors: bool = True,
                 aggregation_type: str = 'a') -> None:
        super(SrlGraphEncoder, self).__init__()
        self.signature = 'graph'
        # AllenNLP's jsonnet configuration does not support `int` key
        residual_layers = dict()   
        for k, v in residual_connection_layers.items():
            residual_layers[int(k)] = v

        self._input_dim = input_dim

        self._layer_timesteps = layer_timesteps
        self._residual_connection_layers = residual_layers
        self._node_msg_dropout = torch.nn.Dropout(node_msg_dropout)
        self._residual_dropout = torch.nn.Dropout(residual_dropout)
        self._aggregation_type = aggregation_type
        self._combined_vectors = combined_vectors
        self._dense_layer_dims = dense_layer_dims
    
    def add_parameters(self, num_edge_types: int) -> None: 
        self._num_edge_types = num_edge_types

        self._residual_layers = []
        self._edge_function_layers = []
        for i_layer in range(len(self._layer_timesteps)):
            current_edge_function_layer = torch.nn.ModuleList(
                [torch.nn.Linear(self._input_dim, self._input_dim, bias=True) 
                    for _ in range(self._num_edge_types)]) 
            setattr(self, 'edge_function_layer_{}'.format(i_layer), current_edge_function_layer)
            self._edge_function_layers.append(current_edge_function_layer)
            """ 
            residual_connection_layer = self._residual_connection_layers.get(i_layer, [])  
            current_residual_layer = torch.nn.GRUCell(
                self._input_dim * (1 + len(residual_connection_layer)),
                self._input_dim)
            setattr(self, 'residual_layer_{}'.format(i_layer), current_residual_layer)
            self._residual_layers.append(current_residual_layer)
            """
        input_dim = self._input_dim
        if self._combined_vectors:
            input_dim = 2 * input_dim 
        self._aggregation_gate = torch.nn.Linear(input_dim, self._input_dim, bias=True)
        self._aggregation_info = torch.nn.Linear(input_dim, self._input_dim, bias=True)
        
        self._dense_layers = []
        input_dim = self._input_dim
        for i_layer, dim in enumerate(self._dense_layer_dims):
            dense_layer = torch.nn.Linear(input_dim, dim, bias=True)
            setattr(self, 'dense_layer_{}'.format(i_layer), dense_layer)
            self._dense_layers.append(dense_layer)
            input_dim = dim
        self._logit = torch.nn.Linear(self._input_dim, 1, bias=True)
        if self._aggregation_type == 'b':
            self._weight = torch.nn.Linear(self._input_dim, 1, bias=True)

    def forward(self, 
                mask: torch.Tensor,
                embedded_nodes: torch.Tensor,  
                edge_types: torch.Tensor,
                edge_type_onehots: torch.Tensor = None) -> torch.Tensor:
        batch_size, num_nodes, _ = embedded_nodes.size()
        valid_edge_types = set(edge_types.view(-1).tolist()) 

        output = self.multi_layer_gcn_encoder(
                                          embedded_nodes,
                                          edge_type_onehots,
                                          valid_edge_types,
                                          edge_types,
                                          batch_size,
                                          num_nodes,
                                          mask)
        z = self.gcn_aggregation(output, num_nodes, mask)
        return z 

    def multi_layer_gcn_encoder(self, 
                                embedded_nodes: torch.Tensor, 
                                edge_type_onehots: torch.Tensor, 
                                valid_edge_types: Set[int], 
                                edge_types: torch.Tensor,
                                batch_size: int, 
                                num_nodes: int,
                                node_mask: torch.Tensor,
                                last_layer_output: bool = True): 
        dummy = torch.zeros([batch_size, num_nodes, self._input_dim], device=embedded_nodes.device) 
        msg_to_predicate_idx = range(num_nodes - 1)
        msg_to_predicate_idy = [i + 1 for i in msg_to_predicate_idx]
        # (batch_size, num_edges)
        edge_average = node_mask.float() / torch.sum(node_mask, -1).unsqueeze(-1).float()

        embedded_nodes_per_layer = [embedded_nodes]
        for i_layer, n_timestep in enumerate(self._layer_timesteps):
            """
            residual_layer = self._residual_layers[i_layer]
            """
            edge_fun_layer = self._edge_function_layers[i_layer] 
            """
            residual_connections = self._residual_connection_layers.get(i_layer, [])
            residual_msg = [embedded_nodes_per_layer[i] for i in residual_connections]
            """
            layer_input = embedded_nodes_per_layer[-1] 
            for i_timestep in range(n_timestep):
                output_nodes_per_type = [edge_fun_layer[i_type](layer_input) 
                    if i_type in valid_edge_types else dummy for i_type in range(self._num_edge_types)] 
                # (batch_size, num_edge_types, num_nodes, hidden_dim)
                output_nodes_per_type = torch.stack(output_nodes_per_type, 1)                
                
                if edge_type_onehots is not None: 
                    # to enable gradient propagation
                    selector = edge_type_onehots.unsqueeze(-1).unsqueeze(-1)#.\
                        #expand(-1, -1, -1, num_nodes, self._input_dim) 
                    # (batch_size, num_edges, num_nodes, hidden_dim); num_edges = num_nodes - 1
                    output_types_per_node = torch.sum(torch.mul(output_nodes_per_type.unsqueeze(1), selector), 2) 
                elif edge_types is not None:
                    selector = edge_types.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_nodes, self._input_dim) 
                    output_types_per_node = torch.gather(output_nodes_per_type, 1, selector)
                else:
                    raise ConfigurationError(f"`edge_type_onehots` and `edge_types` should not be both None")
                output_types_per_node = self._node_msg_dropout(output_types_per_node)

                # each argument has only one edge type, the predicate has all of the edge types

                # (batch_size, num_edges, hidden_dim)
                msg_to_arguments = output_types_per_node[:, :, 0, :] 
                #msg_to_arguments = msg_to_arguments * node_mask.unsqueeze(-1).float()
                msg_to_predicate = torch.cat([output_types_per_node[:, x, y, :].unsqueeze(1) 
                    for x, y in zip(msg_to_predicate_idx, msg_to_predicate_idy)], 1)
                # node masks are applied here 
                msg_to_predicate = msg_to_predicate * edge_average.unsqueeze(-1) 
                # (batch_size, 1, hidden_dim)
                msg_to_predicate = torch.sum(msg_to_predicate, 1, keepdim=True)
                # (batch_size, num_edges + 1, hidden_dim); num_nodes = num_edges + 1
                msg_this_layer = torch.cat([msg_to_predicate, msg_to_arguments], 1)
                
                """
                # residual layer input
                residual_layer_input = torch.cat(residual_msg + [msg_this_layer], -1)

                residual_input_view = residual_layer_input.view(-1, residual_layer_input.size(-1))
                layer_input_view = layer_input.view(-1, layer_input.size(-1))

                layer_output = residual_layer(residual_input_view, layer_input_view)
                layer_output = self._residual_dropout(layer_output).view_as(layer_input) 
                layer_input = layer_output 
                """
                layer_input = msg_this_layer
            
            #layer_input = # do not need to mask out dummy nodes here 
            embedded_nodes_per_layer.append(layer_input)

        if last_layer_output:
            output = embedded_nodes_per_layer[-1]
        else:
            output = embedded_nodes_per_layer[1:]
        return output

    def gcn_multi_dense(self, embedded_nodes: torch.Tensor) -> torch.Tensor:
        for dense_layer in self._dense_layers:
            embedded_nodes = torch.tanh(dense_layer(embedded_nodes))
            embedded_nodes = self._node_msg_dropout(embedded_nodes)
        return embedded_nodes

    def gcn_aggregation(self,
                        embedded_nodes: torch.Tensor,
                        num_nodes: int,
                        node_mask: torch.Tensor) -> torch.Tensor:
        # we may need the contextualized predicate vectors
        self.embedded_predicates = embedded_nodes[:, [0], :]  
        embedded_arguments = embedded_nodes[:, 1:, :]
        if self._combined_vectors: # concatenate vectors of predicates and arguments
            embedded_predicate = embedded_nodes[:, [0], :]
            embedded_predicate = embedded_predicate.expand(-1, num_nodes - 1, -1) 
            embedded_arguments = torch.cat([embedded_arguments, embedded_predicate], -1)

        gate = torch.sigmoid(self._aggregation_gate(embedded_arguments))
        info = torch.tanh(self._aggregation_info(embedded_arguments))
        
        # wit is an abstract representation of the graph
        if self._aggregation_type == 'a':
            wit = gate * info * node_mask.unsqueeze(-1).float()
            wit = self._node_msg_dropout(torch.tanh(torch.sum(wit, 1))) 
            wit = self.gcn_multi_dense(wit)
        elif self._aggregation_type == 'b':
            wit = gate * info * node_mask.unsqueeze(-1).float()
            wit = torch.tanh(self._trans(wit))
            # (batch_size, num_edges, 1)
            wit_weight = self._weight(wit)
            wit_weight = wit_weight.masked_fill(node_mask.unsqueeze(-1) == 0, -1e20)
            wit_weight = F.softmax(wit_weight, 1)
            # weighted sum of wit TODO 
            #
        else:
            wit = None
        return wit
    
    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._dense_layer_dims[-1]

