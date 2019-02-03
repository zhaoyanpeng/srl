"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""

from typing import Optional, Tuple

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
                 output_dim: int) -> None:
        super(GanSrlDiscriminator, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.encoder = CnnEncoder(embedding_dim, num_filters, output_dim = output_dim) 
        print(self.encoder) 

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : feature representations.
            A tensor of shape (batch_size, num_timesteps, input_size)

        mask : mask tensor.
            A tensor of shape (batch_size, num_timesteps) filled with 1 or 0

        Returns
        -------
        logits : logits which are further input to the encoder.
        """
        x = self.encoder(inputs, mask)
        return x 
