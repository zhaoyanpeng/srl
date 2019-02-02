"""
Modules that transform a sequence of input vectors
into a sequence of output vectors.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.
"""

from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper 
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from nlpmimic.modules.dio_lstm import DioLstm 


# pylint: disable=protected-access
Seq2SeqEncoder.register("dio_lstm")(_Seq2SeqWrapper(DioLstm))
