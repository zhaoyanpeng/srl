# pylint: disable=no-self-use,invalid-name
import numpy
import pytest, re
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params

from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.modules.seq2vec_encoders.sampler import SamplerGaussian 

class TestSampler(NlpMimicTestCase):

    def test_sampler(self):
        sampler = SamplerGaussian(5, 3)

        z = torch.empty([4, 5]).random_(1)
        z = sampler(z, 2)
        
        z_mu, z_std = sampler.mu, sampler.std
        print()

        print(z, z.size())
        print(z_mu, z_mu.size())
        print(z_std, z_std.size())

        pp = sampler.lprob(z)
        print(pp, pp.size())
        pass
