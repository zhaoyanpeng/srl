"""
Modules that transform a sequence of input vectors
into a sequence of output vectors.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.
"""

from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper 
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

from nlpmimic.modules.seq2vec_encoders.sampler import SamplerGumbel
from nlpmimic.modules.seq2vec_encoders.sampler import SamplerUniform
from nlpmimic.modules.seq2vec_encoders.sampler import SamplerGaussian
from nlpmimic.modules.seq2vec_encoders.srl_gan_discriminator import GanSrlDiscriminator


# pylint: disable=protected-access
#Seq2VecEncoder.register("srl_gan_dis")(_Seq2VecWrapper(GanSrlDiscriminator))
