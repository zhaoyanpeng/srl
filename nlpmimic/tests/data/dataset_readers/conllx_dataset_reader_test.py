# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2003Reader():
    @pytest.mark.parametrize("lazy", (True,))
    #@pytest.mark.parametrize("feature_labels", [['pos'], ['dep']])
    #def test_read_from_file(self, lazy, feature_labels):
    def test_read_from_file(self, lazy):
        """
        conll_reader = ConllxUnlabeledDatasetReader(
                                 feature_labels = ['pos', 'dep'], 
                                 move_preposition_head = False,
                                 allow_null_predicate = True,
                                 instance_type = 'basic') # pylint: disable=protected-access
        droot = '/disk/scratch1/s1847450/data/nyt_annotated/xchen/' 
        context_file = droot + 'nytimes.45.lemma.small'
        appendix_file = droot + 'nytimes.pre.verb.small'

        instances = conll_reader._read(context_file, appendix_file)
        instances = ensure_list(instances)

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

        for _, instance in enumerate(instances):
            meta = instance.fields['metadata']
            print(meta['predicate'])

        """
        conll_reader = ConllxUnlabeledDatasetReader(
                                 feature_labels = ['pos', 'dep'], 
                                 move_preposition_head = False,
                                 allow_null_predicate = False,
                                 instance_type = 'srl_gan') # pylint: disable=protected-access
        
        droot = '/disk/scratch1/s1847450/data/nyt_annotated/xchen/' 
        context_file = droot + 'nytimes.45.lemma.small'
        #appendix_file = droot + 'nytimes.pre.verb.small'
        appendix_file = droot + 'nytimes.verb.small.picked'
        
        instances = conll_reader._read(context_file, appendix_file, appendix_type='nyt_learn')
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
        """
        for _, instance in enumerate(instances):
            meta = instance.fields['metadata']
            print(meta['predicate'])
        """
