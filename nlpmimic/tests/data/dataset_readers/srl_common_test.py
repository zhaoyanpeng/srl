# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2009Reader():
    @pytest.mark.parametrize("lazy", (True,))
    #@pytest.mark.parametrize("feature_labels", [['pos'], ['dep']])
    #def test_read_from_file(self, lazy, feature_labels):
    def test_read_from_file(self, lazy):
        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=True,
                                              allow_null_predicate = False)
        print('\n')
        print(NlpMimicTestCase.PROJECT_ROOT)
        print(NlpMimicTestCase.MODULE_ROOT)
        print(NlpMimicTestCase.TOOLS_ROOT)
        print(NlpMimicTestCase.TESTS_ROOT)
        print(NlpMimicTestCase.FIXTURES_ROOT)
        
        #data_file = str(NlpMimicTestCase.FIXTURES_ROOT / 'data' / 'trial.bit')
        FIXTURES_ROOT = '/afs/inf.ed.ac.uk/user/s18/s1847450/Code/nlpmimic/nlpmimic/tests/fixtures'
        data_file = FIXTURES_ROOT + '/data' + '/trial.bit'
        instances = conll_reader.read(data_file)
        instances = ensure_list(instances)
        
        instance = instances[1]
        
        for instance in instances:
            sentence = Conll2009Sentence.instance(instance)
            print()
            print(sentence.format())

