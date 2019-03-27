# pylint: disable=no-self-use,invalid-name
import pytest, re

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2009Reader():
    
    @pytest.mark.parametrize("lazy", (False,))
    @pytest.mark.parametrize("move", (True,))
    def test_read_from_file(self, lazy, move):
        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=move,
                                              instance_type='srl_graph',
                                              allow_null_predicate = False)
        print('\n')
        print(NlpMimicTestCase.PROJECT_ROOT)
        print(NlpMimicTestCase.MODULE_ROOT)
        print(NlpMimicTestCase.TOOLS_ROOT)
        print(NlpMimicTestCase.TESTS_ROOT)
        print(NlpMimicTestCase.FIXTURES_ROOT)
        
        #data_file = str(NlpMimicTestCase.FIXTURES_ROOT / 'data' / 'trial.bit')
        FIXTURES_ROOT = '/afs/inf.ed.ac.uk/user/s18/s1847450/Code/nlpmimic/nlpmimic/tests/fixtures'
        ifile = FIXTURES_ROOT + '/data' + '/trial.bit'
        
        """
        diff_file = '/disk/scratch1/s1847450/data/conll09/bitgan/devel.noun.diff'
        not_move = self.not_move_heads(diff_file)
        constraints = not_move.keys()
        print(constraints)
        """

        droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        ifile = droot + 'noun.bit'
        
        ofile = ifile + '.untoched' 
        conll_reader.move_preposition_head = False 
        instances = conll_reader.read(ifile)
        instances = ensure_list(instances)
                
        print()
        for idx, instance in enumerate(instances):
            for k, v in instance.fields.items():
                print('{}: {}'.format(k, v))
                pass
            if idx == 2:
                break

        print() 
        meta = instances[0].fields['metadata']
        for k, v in meta.items():
            print(k, v)
        
        pos_tags = meta['pos_tags']
        print('# of instance is {}'.format(len(instances)))
