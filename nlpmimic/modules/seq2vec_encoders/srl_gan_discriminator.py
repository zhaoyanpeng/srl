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

@Seq2VecEncoder.register("srl_gan_dis")
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
                 embedding_dim: int,
                 projected_dim: int,
                 module_choice: str = 'c',
                 hidden_size: int = None,
                 attent_size: int = None,
                 num_layer: int = None,
                 num_model: int = 0,
                 activation: str = 'relu') -> None:
        super(GanSrlDiscriminator, self).__init__()
        
        self.module_choice = module_choice

        self.num_model = num_model
        if self.num_model > 0:
            if self.module_choice == 'e':
                for model_idx in range(self.num_model):
                    _gru = torch.nn.GRU(embedding_dim, hidden_size, num_layer, batch_first=True, bidirectional=True)
                    _attent = torch.nn.Linear(2 * hidden_size, attent_size, bias=True) 
                    _attent.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / (2 * hidden_size)))
                    _attent.bias.data.fill_(0.0)

                    _weight = torch.nn.Linear(attent_size, 1, bias=False)
                    _weight.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / attent_size))
                    
                    self.add_module('gru_{}'.format(model_idx), _gru)
                    self.add_module('attent_{}'.format(model_idx), _attent)
                    self.add_module('weight_{}'.format(model_idx), _weight)
            else:
                raise ConfigurationError(f"unknown module choice {self.module_choice}")
        
        if self.module_choice == 'gcn':
            pass
        elif self.module_choice in {'a', 'b', 'c', 'd', 'e'}:
            self.initialize_single(embedding_dim, projected_dim, hidden_size, attent_size, num_layer, activation) 
        else:
            raise ConfigurationError(f"unknown module choice {self.module_choice}")
    
    def set_wgan(self, use_wgan: bool = False):
        self.use_wgan = use_wgan

    def add_gcn_parameters(self, 
                           num_edge_types: int, 
                           gcn_hidden_dim: int, 
                           layer_timesteps: List[int] = [2, 2, 2, 2],
                           residual_connection_layers: Dict[int, List[int]] = {2: [0], 3: [0, 1]},
                           node_msg_dropout: float = 0.3,
                           residual_dropout: float = 0.3,
                           aggregation_type: str = 'a'):
        self._gcn_hidden_dim = gcn_hidden_dim
        self._num_edge_types = num_edge_types
        self._layer_timesteps = layer_timesteps
        self._residual_connection_layers = residual_connection_layers
        self._node_msg_dropout = torch.nn.Dropout(node_msg_dropout)
        self._residual_dropout = torch.nn.Dropout(residual_dropout)
        self._aggregation_type = aggregation_type

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

        self._aggregation_gate = torch.nn.Linear(2 * self._gcn_hidden_dim, self._gcn_hidden_dim, bias=True)
        self._aggregation_info = torch.nn.Linear(2 * self._gcn_hidden_dim, self._gcn_hidden_dim, bias=True)
        
        self._trans = torch.nn.Linear(self._gcn_hidden_dim, self._gcn_hidden_dim, bias=True)
        self._logit = torch.nn.Linear(self._gcn_hidden_dim, 1, bias=True)
        if self._aggregation_type == 'b':
            self._weight = torch.nn.Linear(self._gcn_hidden_dim, 1, bias=True)

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
            
            logits = self.gcn_aggregation(output_noun, num_nodes, noun_mask)
            labels = mask[batch_size:, 0].detach().clone().fill_(1).float()

            gen_loss = F.binary_cross_entropy(logits, labels, reduction='mean')
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

            output = torch.cat([output_noun, output_verb], 0)

            logits = self.gcn_aggregation(output, num_nodes, mask)
            labels = mask[:, 0].detach().clone().fill_(0).float()
            labels[batch_size:] += 1.
            
            dis_loss = F.binary_cross_entropy(logits, labels, reduction='mean')
            output_dict['dis_loss'] = dis_loss / 2 
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

    def gcn_aggregation(self,
                        embedded_nodes: torch.Tensor,
                        num_nodes: int,
                        node_mask: torch.Tensor):
        embedded_arguments = embedded_nodes[:, 1:, :]
        embedded_predicate = embedded_nodes[:, [0], :]
        embedded_predicate = embedded_predicate.expand(-1, num_nodes - 1, -1) 

        embedded_arguments = torch.cat([embedded_arguments, embedded_predicate], -1)
        gate = torch.sigmoid(self._aggregation_gate(embedded_arguments))
        info = torch.tanh(self._aggregation_info(embedded_arguments))

        # wit is the abstract representation of the graph
        if self._aggregation_type == 'a': 
            wit = gate * info * node_mask.unsqueeze(-1).float()
            wit = self._node_msg_dropout(torch.sum(wit, 1)) 
            probs = torch.sigmoid(self._logit(torch.tanh(self._trans(wit))))
            probs = probs.squeeze(-1)
        elif self._aggregation_type == 'c':
            edge_average = node_mask.float() / torch.sum(node_mask, -1).unsqueeze(-1).float()
            wit = gate * info * node_mask.unsqueeze(-1).float()
            wit = self._node_msg_dropout(torch.sum(wit, 1)) 
            probs = torch.sigmoid(self._logit(torch.tanh(self._trans(wit))))
            probs = probs.squeeze(-1)
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
        return probs 

    def initialize_single(self, 
                          embedding_dim: int, 
                          projected_dim: int, 
                          hidden_size: int, 
                          attent_size: int,
                          num_layer: int, 
                          activation: str):
        # Projection layer: always num_filters -> projection_dim
        self._projection = torch.nn.Linear(embedding_dim, projected_dim, bias=True)
        self._projection.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / embedding_dim))
        self._projection.bias.data.fill_(0.0)
        
        self._logits = torch.nn.Linear(projected_dim, 1, bias=True)
        self._logits.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / projected_dim))
        self._logits.bias.data.fill_(0.0)

        if self.module_choice == 'c':
            self._gru = torch.nn.GRU(embedding_dim, hidden_size, num_layer, batch_first=True, bidirectional=True)
            
            self._attent = torch.nn.Linear(2 * hidden_size, attent_size, bias=True) 
            self._attent.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / (2 * hidden_size)))
            self._attent.bias.data.fill_(0.0)

            self._weight = torch.nn.Linear(attent_size, 1, bias=False)
            self._weight.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / attent_size))
            #self._weight.bias.data.fill_(0.0)
        elif self.module_choice == 'a' or self.module_choice == 'b':
            if activation == 'tanh':
                self._activation = torch.nn.functional.tanh
            elif activation == 'relu':
                self._activation = torch.nn.functional.relu
            else:
                raise ConfigurationError(f"unknown activation {activation}")
        elif self.module_choice == 'd':
            # Projection layer: always num_filters -> projection_dim
            self._projection_gate = torch.nn.Linear(embedding_dim, projected_dim, bias=True)
            self._projection_gate.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / embedding_dim))
            self._projection_gate.bias.data.fill_(0.0)
            
            self._logits_gate = torch.nn.Linear(projected_dim, 1, bias=True)
            self._logits_gate.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / projected_dim))
            self._logits_gate.bias.data.fill_(0.0)
        else:
            pass

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters
        ----------
        tokens : embedding representations.
            A tensor of shape (batch_size, num_timesteps, input_size)

        mask : mask tensor.
            A tensor of shape (batch_size, num_timesteps) filled with 1 or 0

        Returns
        -------
        logits : logits which are further input to a loss function .
        """
        if self.module_choice == 'a':
            logits = self.model_a(tokens, mask)
        elif self.module_choice == 'b':
            logits = self.model_b(tokens, mask)
        elif self.module_choice == 'c':
            if self.use_wgan:
                logits = self.model_c_wgan(tokens, mask)
            else:
                logits = self.model_c(tokens, mask)
        elif self.module_choice == 'd':
            logits = self.model_d(tokens, mask)
        elif self.module_choice == 'e':
            logits = self.model_e(tokens, mask)
        elif self.module_choice == 'gcn':
            logits = None 
        else:
            raise ConfigurationError(f"unknown discriminator type: {self.module_choice}")
        return logits 

    def model_c_wgan(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """ Average-based model outputs could be dominated by non-argument labels.
        """
        logits = torch.tanh(self._projection(tokens))
        # wgan without sigmoid
        individual_probs = self._logits(logits)
        #print(individual_probs.size())
        
        #print(mask, mask.size())
        input_lengths = torch.sum(mask, -1)
        #print(input_lengths)
        
        tokens_packed = torch.nn.utils.rnn.pack_padded_sequence(tokens, input_lengths, batch_first=True)
        outputs, _ = self._gru(tokens_packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #print(outputs, outputs.size())

        attent_vectors = torch.tanh(self._attent(outputs))
        weights = self._weight(attent_vectors)
        
        used_mask = mask[:, :input_lengths[0]].unsqueeze(-1)
        #print(used_mask, used_mask.size())
        
        weights = weights.masked_fill(used_mask == 0, -1e20)
        #print(weights, weights.size())
        weights = F.softmax(weights, dim=1)
        #print(weights, weights.size())
        
        individual_probs = individual_probs[:, :input_lengths[0], :]
        #print(individual_probs, individual_probs.size())
        
        probs = torch.bmm(individual_probs.transpose(1, 2), weights)
        probs = probs.squeeze(-1).squeeze(-1)
        #print(probs, probs.size())
        return probs

    def model_c(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """ Average-based model outputs could be dominated by non-argument labels.
        """
        logits = torch.tanh(self._projection(tokens))
        individual_probs = torch.sigmoid(self._logits(logits))
        #print(individual_probs.size())
        
        #print(mask, mask.size())
        input_lengths = torch.sum(mask, -1)
        #print(input_lengths)
        
        tokens_packed = torch.nn.utils.rnn.pack_padded_sequence(tokens, input_lengths, batch_first=True)
        outputs, _ = self._gru(tokens_packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #print(outputs, outputs.size())

        attent_vectors = torch.tanh(self._attent(outputs))
        weights = self._weight(attent_vectors)
        
        used_mask = mask[:, :input_lengths[0]].unsqueeze(-1)
        #print(used_mask, used_mask.size())
        
        weights = weights.masked_fill(used_mask == 0, -1e20)
        #print(weights, weights.size())
        weights = F.softmax(weights, dim=1)
        #print(weights, weights.size())
        
        individual_probs = individual_probs[:, :input_lengths[0], :]
        #print(individual_probs, individual_probs.size())
        
        probs = torch.bmm(individual_probs.transpose(1, 2), weights)
        probs = probs.squeeze(-1).squeeze(-1)
        #print(probs, probs.size())
        
        probs.clamp_(max = 1.0) # fix float precision problem
        return probs
    
    def model_e(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """ Average-based model outputs could be dominated by non-argument labels.
        """
        logits = torch.tanh(self._projection(tokens))
        individual_probs = torch.sigmoid(self._logits(logits))
        #print(individual_probs.size())
        
        #print(mask, mask.size())
        input_lengths = torch.sum(mask, -1)
        #print(input_lengths)
        used_mask = mask[:, :input_lengths[0]].unsqueeze(-1)
        #print(used_mask, used_mask.size())
        individual_probs = individual_probs[:, :input_lengths[0], :]
        #print(individual_probs, individual_probs.size())
        
        tokens_packed = torch.nn.utils.rnn.pack_padded_sequence(tokens, input_lengths, batch_first=True)
        
        all_probs = []
        for model_idx in range(self.num_model):
            _gru = getattr(self, 'gru_{}'.format(model_idx))
            _attent = getattr(self, 'attent_{}'.format(model_idx))
            _weight = getattr(self, 'weight_{}'.format(model_idx))
             
            outputs, _ = _gru(tokens_packed)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            #print(outputs, outputs.size())

            weights = _weight(torch.tanh(_attent(outputs)))
        
            weights = weights.masked_fill(used_mask == 0, -1e20)
            weights = F.softmax(weights, dim=1)
            #print(weights, weights.size())
        
            probs = torch.bmm(individual_probs.transpose(1, 2), weights)
            probs = probs.squeeze(-1).squeeze(-1)
            #print(probs, probs.size())
            #probs.clamp_(max = 1.0) # fix float precision problem
            all_probs.append(probs)
        
        all_probs = torch.stack(all_probs, dim=1)
        #print(all_probs, all_probs.size())
        probs = torch.mean(all_probs, 1)
        probs.clamp_(max = 1.0) # fix float precision problem
        #print(probs, probs.size())
        return probs

    def model_d(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """ Gated probabilities without correlations.
        """
        _, length, _ = tokens.size()
        #tokens = tokens * mask.unsqueeze(-1).float() 
        
        # (batch_size, length, dim) -> (batch_size, length, projected_dim)
        logits = torch.tanh(self._projection(tokens))
        # (batch_size, length, 1)
        individual_probs = torch.sigmoid(self._logits(logits))
        
        # (batch_size, length, dim) -> (batch_size, length, projected_dim)
        logits_gate = torch.tanh(self._projection_gate(tokens))
        # (batch_size, length, 1)
        individual_gates = torch.sigmoid(self._logits_gate(logits_gate))

        # weighted and masked (batch_size, length, 1)
        weighted_probs = individual_probs * individual_gates * mask.unsqueeze(-1).float()
        
        # average to reduce length bias 
        divider = 1. / torch.sum(mask, -1).float()
        divider = divider.unsqueeze(-1).unsqueeze(-1).expand(-1, length, -1)

        # -- * (batch_size, length, 1) -> (batch_size, 1, 1) -> (batch_size,)
        logits = torch.bmm(weighted_probs.transpose(1, 2), divider)
        probs = torch.sigmoid(logits).squeeze(-1).squeeze(-1)
        return probs 

    def model_b(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """ Average logits from projecting feature vectors.
        """
        _, length, _ = tokens.size()
        tokens = tokens * mask.unsqueeze(-1).float() 
        
        # (batch_size, length, dim) -> (batch_size, length, projected_dim)
        projected_tokens = self._activation(self._projection(tokens))
        # (batch_size, length, projected_dim) -> (batch_size, length, 1)
        all_logits = self._logits(projected_tokens)

        divider = 1. / torch.sum(mask, -1).float()
        divider = divider.unsqueeze(-1).unsqueeze(-1).expand(-1, length, -1)
        # -- * (batch_size, length, 1) -> (batch_size, 1, 1) -> (batch_size,)
        logits = torch.bmm(all_logits.transpose(1, 2), divider)
        probs = torch.sigmoid(logits).squeeze(-1).squeeze(-1)
        return probs 
    
    def model_a(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """ Average feature vectors.
        """
        # reset masks with 1. if there are NOT any srl labels 
        #divider = torch.sum(mask, -1).float()
        #zero_indexes = divider == 0.
        #mask[zero_indexes, :] = 1.
        _, length, _ = tokens.size()
        tokens = tokens * mask.unsqueeze(-1).float() 
        
        # (batch_size, length, dim) -> (batch_size, length, projected_dim)
        projected_tokens = self._activation(self._projection(tokens))
        
        divider = 1. / torch.sum(mask, -1).float()
        divider = divider.unsqueeze(-1).unsqueeze(-1).expand(-1, length, -1)
        
        # (batch_size, dim, length) * (batch_size, length, 1) 
        features = torch.bmm(projected_tokens.transpose(1, 2), divider)
        features = features.squeeze(-1)

        probs = torch.sigmoid(self._logits(features)).squeeze(-1) 
        return probs 

