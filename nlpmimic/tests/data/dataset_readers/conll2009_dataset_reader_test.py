# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2003Reader():
    @pytest.mark.parametrize("lazy", (True,))
    #@pytest.mark.parametrize("feature_labels", [['pos'], ['dep']])
    #def test_read_from_file(self, lazy, feature_labels):
    def test_read_from_file(self, lazy):
        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=True,
                                              allow_null_predicate = True)
        
        
        instances = conll_reader.read(str(NlpMimicTestCase.FIXTURES_ROOT / 'data' / 'conll2009.txt'))
        instances = ensure_list(instances)

        for instance in instances:
            for k, v in instance.fields.items():
                #print('{}: {}'.format(k, v))
                pass
            #print(instance.fields['metadata'].metadata)
        pos_tags = ['FAKE', 'RBS', 'JJ', 'NN', 'MD', 'VB', 'DT', 'NNP', 'NN', 'NN', 'NN', 'JJ', 'IN', 'NN', '.'] 
        head_ids = [4, 3, 4, 5, 0, 5, 11, 11, 10, 11, 6, 13, 11, 13, 5] 
        dep_rels = ['NMOD', 'FAKE', 'NMOD', 'SBJ', 'ROOT', 'VC', 'NMOD', 'NMOD', 'NMOD', 'NMOD', 'PRD', 'AMOD', 'APPO', 'TMP', 'P']
        
        tokens = ['The', 'most', 'troublesome', 'report', 'may', 'be', 'the', 'August', 'merchandise', 'trade', 'deficit', 'due', 'out', 'tomorrow', '.'] 
        lemmas = ['the', 'most', 'troublesome', 'report', 'may', 'be', 'the', 'august', 'merchandise', 'trade', 'deficit', 'due', 'out', 'tomorrow', '.']
        
        mapping = {'pos_tags': pos_tags, 'head_ids': head_ids, 'dep_rels': dep_rels}

        
        sentence = Conll2009Sentence.instance(instances[0])
        print('\n{}'.format(sentence.format()))

             
        fields = instances[-1].fields
        for k, v in fields['metadata'].items():
            if k in mapping:
                assert v == mapping[k]

        assert [t.text for t in fields['tokens']] == tokens
        assert fields['metadata']['lemmas'] == lemmas 
        assert fields['metadata']['predicate'] == None
        assert fields['metadata']['predicate_sense'] == None
        
        """ 
        sentence = Conll2009Sentence.instance(instances[0])
        print('\n{}'.format(sentence.format()))
        
        sentence = Conll2009Sentence.instance(instances[1])
        print('\n{}'.format(sentence.format()))
        
        sentence.move_preposition_head('_')
        print('\n{}'.format(sentence.format()))

        sentence.restore_preposition_head('_')
        print('\n{}'.format(sentence.format()))
        
        sentence = Conll2009Sentence.instance(instances[2])
        print('\n{}'.format(sentence.format()))

        sentence = Conll2009Sentence.instance(instances[3])
        print('\n{}'.format(sentence.format()))
        
        sentence.move_preposition_head('_')
        print('\n{}'.format(sentence.format()))

        sentence.restore_preposition_head('_')
        print('\n{}'.format(sentence.format()))
        """
        """
        sentence = Conll2009Sentence.instance(instances[4])
        print('\n{}'.format(sentence.format()))
        """
