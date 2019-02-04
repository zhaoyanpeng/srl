"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""
import numpy as np
import torch

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
                 activation: str = 'relu') -> None:
        super(GanSrlDiscriminator, self).__init__()

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
        # reset masks with 1. if there are NOT any srl labels 
        divider = torch.sum(mask, -1).float()
        zero_indexes = divider == 0.
        mask[zero_indexes, :] = 1.
        
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

        logits = self._logits(features)
        
        return logits 
