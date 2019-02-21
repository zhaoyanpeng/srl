"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""
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
                 activation: str = 'relu') -> None:
        super(GanSrlDiscriminator, self).__init__()
        
        self.module_choice = module_choice

        if self.module_choice == 'c':
            # Projection layer: always num_filters -> projection_dim
            self._projection = torch.nn.Linear(embedding_dim, projected_dim, bias=True)
            self._projection.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / embedding_dim))
            self._projection.bias.data.fill_(0.0)
            
            self._logits = torch.nn.Linear(projected_dim, 1, bias=True)
            self._logits.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / projected_dim))
            self._logits.bias.data.fill_(0.0)
            
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
            
            # Projection layer: always num_filters -> projection_dim
            self._projection = torch.nn.Linear(embedding_dim, projected_dim, bias=True)
            self._projection.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / embedding_dim))
            self._projection.bias.data.fill_(0.0)
            
            self._logits = torch.nn.Linear(projected_dim, 1, bias=True)
            self._logits.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / projected_dim))
            self._logits.bias.data.fill_(0.0)
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
            logits = self.model_c(tokens, mask)
        else:
            raise ConfigurationError(f"unknown discriminator type: {self.module_choice}")
        return logits 

    def model_c(self, tokens: torch.Tensor, mask: torch.Tensor = None):
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
        
        if (probs > 1).sum() > 0 or (probs < 0).sum() > 0:
            print(probs)
            print(individual_probs)
            print(weights)

            index = probs > 1 
            torch.set_printoptions(precision=25)
            print(probs)
            for i, x in enumerate(index):
                if x:
                    print(i, probs[i])
                    print(individual_probs[i, :, :])
                    print(weights[i, :, :])
                    print()

            print('happy shitting') 
            index = probs < 0 
            torch.set_printoptions(precision=25)
            print(probs)
            for i, x in enumerate(index):
                if x:
                    print(i, probs[i])
                    print(individual_probs[i, :, :])
                    print(weights[i, :, :])
                    print()
        return probs
    
    def model_b(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        _, length, _ = tokens.size()
        tokens = tokens * mask.unsqueeze(-1).float() 
        
        # (batch_size, length, dim) -> (batch_size, length, projected_dim)
        projected_tokens = self._projection(tokens)
        projected_tokens = self._activation(projected_tokens)
        
        # (batch_size, length, dim) -> (batch_size, length, 1)
        all_logits = self._logits(projected_tokens)
        all_logits = self._activation(all_logits)

        divider = 1. / torch.sum(mask, -1).float()
        divider = divider.unsqueeze(-1).unsqueeze(-1).expand(-1, length, -1)
        # -- * (batch_size, length, 1) -> (batch_size, 1, 1) -> (batch_size,)
        logits = torch.bmm(all_logits.transpose(1, 2), divider)
        probs = torch.sigmoid(logits).squeeze(-1).squeeze(-1)
        return probs 
    
    def model_a(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        # reset masks with 1. if there are NOT any srl labels 
        #divider = torch.sum(mask, -1).float()
        #zero_indexes = divider == 0.
        #mask[zero_indexes, :] = 1.
        
        _, length, _ = tokens.size()
        tokens = tokens * mask.unsqueeze(-1).float() 
        
        # (batch_size, length, dim) -> (batch_size, length, projected_dim)
        projected_tokens = self._projection(tokens)
        projected_tokens = self._activation(projected_tokens)
        
        divider = 1. / torch.sum(mask, -1).float()
        divider = divider.unsqueeze(-1).unsqueeze(-1).expand(-1, length, -1)
        
        # (batch_size, dim, length) * (batch_size, length, 1) 
        features = torch.bmm(projected_tokens.transpose(1, 2), divider)
        features = features.squeeze(-1)

        probs = torch.sigmoid(self._logits(features)).squeeze(-1) 
        return probs 

