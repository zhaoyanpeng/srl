"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""
from typing import Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.modules import Seq2VecEncoder
from allennlp.common.checks import ConfigurationError

@Seq2VecEncoder.register("srl_graph_dis")
class GanSrlDiscriminator(Seq2VecEncoder):
    """
    A discriminator takes as input SRL features and gold labels, and output a loss.

    Parameters
    ----------

    Returns
    -------
    logits: float tensor  
        The output is tensor of size (batch_size, 1) and is used for binary classificaiton.
    """
    def __init__(self,
                 layer_timesteps: List[int] = [2, 2, 2, 2],
                 residual_connection_layers: Dict[int, List[int]] = {2: [0], 3: [0, 1]},
                 node_msg_dropout: float = 0.3,
                 residual_dropout: float = 0.3,
                 combined_vectors: bool = True,
                 aggregation_type: str = 'a',
                 num_layers_dense: int = 1,
                 label_smooth_val: float = None,
                 feature_matching: bool = False) -> None:
        super(GanSrlDiscriminator, self).__init__()
        self.signature = 'graph'
        # AllenNLP's jsonnet configuration does not support `int` key
        residual_layers = dict()   
        for k, v in residual_connection_layers.items():
            residual_layers[int(k)] = v

        self._layer_timesteps = layer_timesteps
        self._residual_connection_layers = residual_layers
        self._node_msg_dropout = torch.nn.Dropout(node_msg_dropout)
        self._residual_dropout = torch.nn.Dropout(residual_dropout)
        self._aggregation_type = aggregation_type
        self._combined_vectors = combined_vectors
        self._num_layers_dense = num_layers_dense
        self._label_smooth_val = label_smooth_val
        self._feature_matching = feature_matching
    
    def add_gcn_parameters(self, 
                           num_edge_types: int, 
                           gcn_hidden_dim: int): 
        self._gcn_hidden_dim = gcn_hidden_dim
        self._num_edge_types = num_edge_types

        self._residual_layers = []
        self._edge_function_layers = []
        for i_layer in range(len(self._layer_timesteps)):
            current_edge_function_layer = torch.nn.ModuleList(
                [torch.nn.Linear(self._gcn_hidden_dim, self._gcn_hidden_dim, bias=True) 
                    for _ in range(self._num_edge_types)]) 
            setattr(self, 'edge_function_layer_{}'.format(i_layer), current_edge_function_layer)
            self._edge_function_layers.append(current_edge_function_layer)
            
            residual_connection_layer = self._residual_connection_layers.get(i_layer, [])  
            current_residual_layer = torch.nn.GRUCell(
                self._gcn_hidden_dim * (1 + len(residual_connection_layer)),
                self._gcn_hidden_dim)
            setattr(self, 'residual_layer_{}'.format(i_layer), current_residual_layer)
            self._residual_layers.append(current_residual_layer)

        input_size = self._gcn_hidden_dim
        if self._combined_vectors:
            input_size = 2 * input_size 
        self._aggregation_gate = torch.nn.Linear(input_size, self._gcn_hidden_dim, bias=True)
        self._aggregation_info = torch.nn.Linear(input_size, self._gcn_hidden_dim, bias=True)
        
        self._trans = []
        for i_layer in range(self._num_layers_dense):
            dense_layer = torch.nn.Linear(self._gcn_hidden_dim, self._gcn_hidden_dim, bias=True)
            setattr(self, 'dense_layer_{}'.format(i_layer), dense_layer)
            self._trans.append(dense_layer)
        self._logit = torch.nn.Linear(self._gcn_hidden_dim, 1, bias=True)
        if self._aggregation_type == 'b':
            self._weight = torch.nn.Linear(self._gcn_hidden_dim, 1, bias=True)

    def forward(self, 
                input_dict: Dict[str, torch.Tensor], 
                mask: torch.Tensor, 
                output_dict: Dict[str, Any],
                retrive_generator_loss: bool,
                return_without_computation: bool = False) -> None:
        """
        Parameters
        ----------
        input_dict : all stuff needed for computation.
        output_dict: which functions as a container storing computed results.
        mask : mask tensor.
            A tensor of shape (batch_size, num_timesteps) filled with 1 or 0

        Returns
        -------
        logits : logits which are further input to a loss function .
        """
        self.multi_layer_gcn_loss(input_dict, 
                                  mask, 
                                  output_dict, 
                                  retrive_generator_loss, 
                                  return_without_computation)
        return None 

    def multi_layer_gcn_loss(self, 
                             input_dict: Dict[str, torch.Tensor], 
                             mask: torch.Tensor, 
                             output_dict: Dict[str, Any],
                             retrive_generator_loss: bool,
                             return_without_computation: bool = False) -> None:
        if return_without_computation:
            default_value = torch.tensor(0.0, dtype=torch.float, device=mask.device)
            if retrive_generator_loss:
                output_dict['gen_loss'] = default_value 
            else:
                output_dict['dis_loss'] = default_value
            return None

        embedded_noun_nodes = input_dict['n_embedded_nodes']
        noun_edge_types = input_dict['n_edge_types']
        noun_edge_type_onehots = input_dict['n_edge_type_onehots']
        noun_valid_edge_types = set(noun_edge_types.view(-1).tolist()) 

        batch_size, num_nodes, _ = embedded_noun_nodes.size()

        if retrive_generator_loss:
            noun_mask = mask[batch_size:]
            output_noun = self.multi_layer_gcn_encoder(
                                              embedded_noun_nodes,
                                              noun_edge_type_onehots,
                                              noun_valid_edge_types,
                                              noun_edge_types,
                                              batch_size,
                                              num_nodes,
                                              noun_mask)
            
            _, probs, _ = self.gcn_aggregation(output_noun, num_nodes, noun_mask)
            labels = mask[batch_size:, 0].detach().clone().fill_(1).float()

            gen_loss = F.binary_cross_entropy(probs, labels, reduction='mean')
            output_dict['gen_loss'] = gen_loss
        else:
            embedded_verb_nodes = input_dict['v_embedded_nodes']
            verb_edge_types = input_dict['v_edge_types']
            verb_edge_type_onehots = input_dict['v_edge_type_onehots']
            verb_valid_edge_types = set(verb_edge_types.view(-1).tolist()) 
            #valid_edge_types = noun_valid_edge_types | verb_valid_edge_types
            
            noun_mask = mask[batch_size:]
            output_noun = self.multi_layer_gcn_encoder(
                                              embedded_noun_nodes,
                                              noun_edge_type_onehots,
                                              noun_valid_edge_types,
                                              noun_edge_types,
                                              batch_size,
                                              num_nodes,
                                              noun_mask)

            verb_mask = mask[:batch_size]
            output_verb = self.multi_layer_gcn_encoder(
                                              embedded_verb_nodes,
                                              verb_edge_type_onehots,
                                              verb_valid_edge_types,
                                              verb_edge_types,
                                              batch_size,
                                              num_nodes,
                                              verb_mask)

            output = torch.cat([output_verb, output_noun], 0)

            logits, probs, features = self.gcn_aggregation(output, num_nodes, mask)
            labels = mask[:, 0].detach().clone().fill_(0).float()
            
            if self._feature_matching:
                v_features = features[:batch_size]
                n_features = features[batch_size:]
                diff = torch.mean(v_features, 0) - torch.mean(n_features, 0) 
                gf_loss = torch.sum(diff ** 2)
                output_dict['gen_loss'] = gf_loss
            
            if self._label_smooth_val is None:
                labels[:batch_size] += 1.0 
                dis_loss = F.binary_cross_entropy(probs, labels, reduction='mean')
            else:
                noise = self._label_smooth_val
                if self._label_smooth_val <= 0.5:
                    noise = torch.empty(batch_size, device=labels.device).uniform_(0.7, 1.2) 
                labels[:batch_size] += noise 
                dis_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

            output_dict['dis_loss'] = dis_loss #/ 2 
        return None 

    def multi_layer_gcn_encoder(self, 
                                embedded_nodes: torch.Tensor, 
                                edge_type_onehots: torch.Tensor, 
                                valid_edge_types: Set[int], 
                                edge_types: torch.Tensor,
                                batch_size: int, 
                                num_nodes: int,
                                node_mask: torch.Tensor,
                                last_layer_output: bool = True): 
        dummy = torch.zeros([batch_size, num_nodes, self._gcn_hidden_dim], device=embedded_nodes.device) 
        msg_to_predicate_idx = range(num_nodes - 1)
        msg_to_predicate_idy = [i + 1 for i in msg_to_predicate_idx]
        # (batch_size, num_edges)
        edge_average = node_mask.float() / torch.sum(node_mask, -1).unsqueeze(-1).float()

        embedded_nodes_per_layer = [embedded_nodes]
        for i_layer, n_timestep in enumerate(self._layer_timesteps):
            residual_layer = self._residual_layers[i_layer]
            edge_fun_layer = self._edge_function_layers[i_layer] 
            
            residual_connections = self._residual_connection_layers.get(i_layer, [])
            residual_msg = [embedded_nodes_per_layer[i] for i in residual_connections]

            layer_input = embedded_nodes_per_layer[-1] 
            for i_timestep in range(n_timestep):
                output_nodes_per_type = [edge_fun_layer[i_type](layer_input) 
                    if i_type in valid_edge_types else dummy for i_type in range(self._num_edge_types)] 
                # (batch_size, num_edge_types, num_nodes, hidden_dim)
                output_nodes_per_type = torch.stack(output_nodes_per_type, 1)                
                
                if edge_type_onehots is not None: 
                    # to enable gradient propagation
                    selector = edge_type_onehots.unsqueeze(-1).unsqueeze(-1)#.\
                        #expand(-1, -1, -1, num_nodes, self._gcn_hidden_dim) 
                    # (batch_size, num_edges, num_nodes, hidden_dim); num_edges = num_nodes - 1
                    output_types_per_node = torch.sum(torch.mul(output_nodes_per_type.unsqueeze(1), selector), 2) 
                elif edge_types is not None:
                    selector = edge_types.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_nodes, self._gcn_hidden_dim) 
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
                
                # residual layer input
                residual_layer_input = torch.cat(residual_msg + [msg_this_layer], -1)

                residual_input_view = residual_layer_input.view(-1, residual_layer_input.size(-1))
                layer_input_view = layer_input.view(-1, layer_input.size(-1))

                layer_output = residual_layer(residual_input_view, layer_input_view)
                layer_output = self._residual_dropout(layer_output).view_as(layer_input) 
                layer_input = layer_output 
            
            #layer_input = # do not need to mask out dummy nodes here 
            embedded_nodes_per_layer.append(layer_input)

        if last_layer_output:
            output = embedded_nodes_per_layer[-1]
        else:
            output = embedded_nodes_per_layer[1:]
        return output

    def gcn_multi_dense(self, embedded_nodes: torch.Tensor):
        for dense_layer in self._trans:
            embedded_nodes = torch.tanh(dense_layer(embedded_nodes))
            embedded_nodes = self._node_msg_dropout(embedded_nodes)
        return embedded_nodes

    def gcn_aggregation(self,
                        embedded_nodes: torch.Tensor,
                        num_nodes: int,
                        node_mask: torch.Tensor):
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
            wit = self._node_msg_dropout(torch.sum(wit, 1)) 
            logit = self._logit(torch.tanh(self._trans(wit))).squeeze(-1)
            probs = torch.sigmoid(logit)
        elif self._aggregation_type == 'b':
            wit = gate * info * node_mask.unsqueeze(-1).float()
            wit = torch.tanh(self._trans(wit))
            # (batch_size, num_edges, 1)
            wit_logits = torch.sigmoid(self._logit(wit)) 
            # (batch_size, num_edges, 1)
            wit_weight = self._weight(wit)
            wit_weight = wit_weight.masked_fill(node_mask.unsqueeze(-1) == 0, -1e20)
            wit_weight = F.softmax(wit_weight, 1)
            # weighted vote 
            probs = torch.bmm(wit_logits.transpose(1, 2), wit_weight)
            probs = probs.squeeze(-1).squeeze(-1)
            probs.clamp_(max = 1.0) # fix float precision problem
            logit = None
        elif self._aggregation_type == 'c':
            wit = gate * info * node_mask.unsqueeze(-1).float()
            wit = self._node_msg_dropout(torch.tanh(torch.sum(wit, 1))) 

            wit = self.gcn_multi_dense(wit)

            logit = self._logit(wit).squeeze(-1)
            probs = torch.sigmoid(logit)
        else:
            logit = probs = wit = None
        return logit, probs, wit
