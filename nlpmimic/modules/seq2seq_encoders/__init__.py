"""
Modules that transform a sequence of input vectors
into a sequence of output vectors.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.
"""
from typing import Type
import logging

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper 
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm

from nlpmimic.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import MimicSeq2SeqWrapper
from nlpmimic.modules.dio_lstm import DioLstm 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class _MimicSeq2SeqWrapper:
    PYTORCH_MODELS = [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]

    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
        self._module_class = module_class

    def __call__(self, **kwargs) -> MimicSeq2SeqWrapper:
        return self.from_params(Params(kwargs))

    # Logic requires custom from_params
    def from_params(self, params: Params) -> MimicSeq2SeqWrapper:
        if not params.pop_bool('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        if self._module_class in self.PYTORCH_MODELS:
            params['batch_first'] = True
        stateful = params.pop_bool('stateful', False)
        module = self._module_class(**params.as_dict())
        return MimicSeq2SeqWrapper(module, stateful=stateful)

# pylint: disable=protected-access
Seq2SeqEncoder.register("dio_lstm")(_Seq2SeqWrapper(DioLstm))
Seq2SeqEncoder.register("stacked_bilstm")(_MimicSeq2SeqWrapper(StackedBidirectionalLstm))
