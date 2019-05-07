# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch
from torch.nn import Embedding, Parameter

from allennlp.common.testing import AllenNlpTestCase
from nlpmimic.modules import Nlpmimic  


class TestNlpmimic(AllenNlpTestCase):
    def test_nlpmimic(self):
        print('\n---here is the nlpmimic tester.\n')

        nlp = Nlpmimic()
        nlp()
