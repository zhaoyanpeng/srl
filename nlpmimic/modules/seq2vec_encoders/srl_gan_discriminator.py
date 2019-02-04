"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_dropout_mask
from allennlp.nn.initializers import block_orthogonal


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
                 num_filters: int,
                 output_dim: int,
                 activation: str = 'relu') -> None:
        super(GanSrlDiscriminator, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.encoder = CnnEncoder(embedding_dim, num_filters, output_dim = output_dim) 
        
        num_filters = 6
        projection_dim = output_dim = 4 

        if activation == 'tanh':
            self._activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self._activation = torch.nn.functional.relu
        else:
            raise ConfigurationError(f"unknown activation {activation}")
        
        # Projection layer: always num_filters -> projection_dim
        self._projection = torch.nn.Linear(num_filters, projection_dim, bias=True)
        self._projection.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / num_filters))
        self._projection.bias.data.fill_(0.0)

        self._logits = torch.nn.Linear(projection_dim, 1, bias=True)
        self._logits.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / num_filters))
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
        logits : logits which are further input to the encoder.
        """
        
        batch_size, length, dim = tokens.size()
        tokens = tokens * mask.unsqueeze(-1).float() 

        print(tokens.size())
        
        projected_tokens = self._projection(tokens)
        projected_tokens = self._activation(projected_tokens)
        print(projected_tokens.size())
        
        divider = (1. / torch.sum(mask, -1).float()).unsqueeze(-1)
        divider = divider.unsqueeze(-1).expand(-1, -1, length)
        
        print(divider.size())
        
        features = torch.bmm(projected_tokens.transpose(1, 2), 
                             divider.transpose(1, 2))
        print(features.size())
        features = features.squeeze(-1)
        print(features.size())

        logits = self._logits(features)
        
        print(logits.size())
        print(logits)
        import sys
        sys.exit(0)
        
        x = self.encoder(inputs, mask)
        return x 
