# pylint: disable=no-self-use,invalid-name
import pytest, re

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2009Reader():
    
    def not_move_heads(self, ifile):
        regx = '(\d+)c(\d+)'
        not_move = dict()
        with open(ifile, 'r') as fr:
            search = False
            for line in fr:
                line = line.strip()
                if not search and re.search(regx, line):
                    line_nums = line.split('c')[0].split(',') 
                    if len(line_nums) > 1:
                        line_nums = list(range(int(line_nums[0]), 1 + int(line_nums[-1]))) 
                    search = True
                    cnt = 0
                    continue
                if search and line.startswith('<'):
                    key = line.split()[-1] 
                    if key != '_':
                        not_move[int(line_nums[cnt])] = line
                    cnt += 1
                    continue
                if search and line == '---':
                    search = False
        return not_move 
    
    @pytest.mark.parametrize("lazy", (False,))
    @pytest.mark.parametrize("move", (True,))
    def test_read_from_file(self, lazy, move):
        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=move,
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

        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ifile = droot + 'devel.noun'
        
        ofile = ifile + '.untoched' 
        conll_reader.move_preposition_head = False 
        instances = conll_reader.read(ifile)
        instances = ensure_list(instances)
                
        with open(ofile, 'w') as fw:
            for instance in instances:
                sentence = Conll2009Sentence.instance(instance)
                fw.write(sentence.format() + '\n')
         
        ofile = ifile + '.moved' 
        conll_reader.move_preposition_head = True
        instances = conll_reader.read(ifile)
        
        with open(ofile, 'w') as fw:
            for instance in instances:
                sentence = Conll2009Sentence.instance(instance)
                fw.write(sentence.format() + '\n')
        
        ofile = ifile + '.restored' 
        with open(ofile, 'w') as fw:
            start_pos = 0
            for instance in instances:
                sentence = Conll2009Sentence.instance(instance)
                #sentence.restore_preposition_head('_')
                sentence.restore_preposition_head_theory('_')
                #start_pos = sentence.restore_preposition_head_with_conditions(start_pos, constraints, '_')
                fw.write(sentence.format() + '\n')

