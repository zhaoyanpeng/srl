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

@Seq2VecEncoder.register("srl_naive_dis")
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
                 num_model: int = 0) -> None:
        super(GanSrlDiscriminator, self).__init__()
        self.signature = 'naive'
        
        self.module_choice = module_choice
        self.num_model = num_model
        
        if self.module_choice in {'a', 'b', 'c', 'd', 'e'}:
            self.initialize(embedding_dim, projected_dim, hidden_size, attent_size, num_layer) 
        else:
            raise ConfigurationError(f"unknown module choice {self.module_choice}")
        self.use_wgan = False 

    def set_wgan(self, use_wgan: bool = False):
        self.use_wgan = use_wgan

    def initialize(self, 
                   embedding_dim: int, 
                   projected_dim: int, 
                   hidden_size: int, 
                   attent_size: int,
                   num_layer: int):
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
        elif self.module_choice == 'd':
            # Projection layer: always num_filters -> projection_dim
            self._projection_gate = torch.nn.Linear(embedding_dim, projected_dim, bias=True)
            self._projection_gate.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / embedding_dim))
            self._projection_gate.bias.data.fill_(0.0)
            
            self._logits_gate = torch.nn.Linear(projected_dim, 1, bias=True)
            self._logits_gate.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / projected_dim))
            self._logits_gate.bias.data.fill_(0.0)
        elif self.module_choice == 'e':
            if self.num_model > 0:
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
                raise ConfigurationError(f"invalid module configuration self.num_model = {self.num_model}")
        else:
            pass

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
        self.gan_loss(input_dict, 
                      mask, 
                      output_dict, 
                      retrive_generator_loss, 
                      return_without_computation)
        return None 

    def gan_loss(self, 
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

        #embedded_noun_tokens = input_dict['n_embedded_tokens']
        embedded_noun_lemmas = input_dict['n_embedded_lemmas']
        embedded_noun_predicates = input_dict['n_embedded_predicates']
        embedded_noun_labels = input_dict['n_embedded_labels']
        
        batch_size = embedded_noun_lemmas.size(0)
        
        noun_features = [embedded_noun_predicates, 
                         embedded_noun_labels, 
                         embedded_noun_lemmas]
        embedded_noun = torch.cat(noun_features, -1)
        
        if retrive_generator_loss:
            mask = mask[batch_size:]
            logits = self.dispatch(embedded_noun, mask)  
            # fake labels 
            real_labels = mask[:, 0].detach().clone().fill_(1).float()
            gen_loss = F.binary_cross_entropy(logits, real_labels, reduction='mean')
            output_dict['gen_loss'] = gen_loss
        else:
            #embedded_verb_tokens = input_dict['v_embedded_tokens']
            embedded_verb_lemmas = input_dict['v_embedded_lemmas']
            embedded_verb_predicates = input_dict['v_embedded_predicates']
            embedded_verb_labels = input_dict['v_embedded_labels']
            
            verb_features = [embedded_verb_predicates,
                             embedded_verb_labels,
                             embedded_verb_lemmas]

            embedded_verb = torch.cat(verb_features, -1)

            mask_noun = mask[batch_size:]
            logits_noun = self.dispatch(embedded_noun, mask_noun) 
            
            mask_verb = mask[:batch_size]
            logits_verb = self.dispatch(embedded_verb, mask_verb)
            
            # fake labels
            fake_labels = mask[:batch_size, 0].detach().clone().fill_(0).float()
            real_labels = mask[:batch_size, 0].detach().clone().fill_(1).float()
            
            dis_loss = F.binary_cross_entropy(logits_verb, real_labels, reduction='mean') \
                     + F.binary_cross_entropy(logits_noun, fake_labels, reduction='mean') 
            output_dict['dis_loss'] = dis_loss / 2 
        return None 


    def dispatch(self, tokens: torch.Tensor, mask: torch.Tensor = None):
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
        projected_tokens = torch.tanh(self._projection(tokens))
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
        projected_tokens = torch.tanh(self._projection(tokens))
        
        divider = 1. / torch.sum(mask, -1).float()
        divider = divider.unsqueeze(-1).unsqueeze(-1).expand(-1, length, -1)
        
        # (batch_size, dim, length) * (batch_size, length, 1) 
        features = torch.bmm(projected_tokens.transpose(1, 2), divider)
        features = features.squeeze(-1)

        probs = torch.sigmoid(self._logits(features)).squeeze(-1) 
        return probs 

