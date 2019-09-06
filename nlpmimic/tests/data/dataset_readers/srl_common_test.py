# pylint: disable=no-self-use,invalid-name
import pytest, re

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2003Reader(NlpMimicTestCase):
    
    @pytest.mark.skip(reason="mute")
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
    
    def test_read_from_file(self):
        ofile = min_valid_lemmas = valid_srl_labels = None
        conll_reader = Conll2009DatasetReader(lazy=True, 
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              feature_labels=['pos', 'dep'], 
                                              moved_preposition_head=['IN', 'TO'],
                                              instance_type='srl_graph',
                                              maximum_length = 2019,
                                              min_valid_lemmas = min_valid_lemmas,
                                              max_num_argument = 7, 
                                              valid_srl_labels = valid_srl_labels,
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
        ifile = droot + 'test.noun'
        #ifile = droot + 'xx.noun'
        
        ofile = ifile + '.untoched.new' 
        conll_reader.moved_preposition_head = [] 
        instances = conll_reader.read(ifile)
        instances = ensure_list(instances)
                
        with open(ofile, 'w') as fw:
            for instance in instances:
                sentence = Conll2009Sentence.instance(instance)
                fw.write(sentence.format() + '\n')
         
        ofile = ifile + '.moved.new' 
        conll_reader.moved_preposition_head = ["IN"] 
        instances = conll_reader.read(ifile)
        
        with open(ofile, 'w') as fw:
            for instance in instances:
                sentence = Conll2009Sentence.instance(instance)
                fw.write(sentence.format() + '\n')
        
        ofile = ifile + '.restored.new' 
        with open(ofile, 'w') as fw:
            start_pos = 0
            for instance in instances:
                sentence = Conll2009Sentence.instance(instance)
                #sentence.restore_preposition_head('_')
                for pos in conll_reader.moved_preposition_head:
                    sentence.restore_preposition_head_general(empty_label='_', preposition=pos)
                #start_pos = sentence.restore_preposition_head_with_conditions(start_pos, constraints, '_')
                fw.write(sentence.format() + '\n')

