# pylint: disable=no-self-use,invalid-name
import pytest, re
from tqdm import tqdm
from collections import Counter

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

    
class TestConll2009Reader():
    """ 
    @pytest.mark.parametrize("lazy", (True,))
    #@pytest.mark.parametrize("feature_labels", [['pos'], ['dep']])
    #def test_read_from_file(self, lazy, feature_labels):
    def test_read_from_file(self, lazy):
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ofile = droot + 'all.moved.arg.vocab' 
        conll_reader = ConllxUnlabeledDatasetReader(
                                 #lemma_file = ofile,
                                 #lemma_use_firstk = 5,
                                 feature_labels = ['pos', 'dep'], 
                                 move_preposition_head = True,
                                 allow_null_predicate = False,
                                 instance_type = 'srl_gan') # pylint: disable=protected-access
        
        droot = '/disk/scratch1/s1847450/data/nyt_annotated/xchen/' 
        context_file = droot + 'nytimes.45.lemma.small'
        #appendix_file = droot + 'nytimes.pre.verb.small'
        appendix_file = droot + 'nytimes.verb.small.picked'
        
        instances = conll_reader._read(context_file, appendix_file, appendix_type='nyt_learn')
        instances = ensure_list(instances)

        lemma_dict = Counter()
        
        # no gold lemmas
        for instance in tqdm(instances):
            lemmas = instance['metadata']['lemmas']
            print(lemmas)
            for lemma in lemmas:
                if lemma == Conll2009DatasetReader._EMPTY_LEMMA:
                    continue
                lemma_dict[lemma] += 1
        
        print('\n|vocab of lemmas| is {}'.format(len(conll_reader.lemma_set)))
        print('|vocab of lemmas| is {}'.format(len(lemma_dict)))
    """

    @pytest.mark.parametrize("lazy", (False,))
    @pytest.mark.parametrize("move", (True,))
    def test_vocabulary_file(self, lazy, move):
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ofile = droot + 'all.moved.arg.vocab' 
        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=move,
                                              instance_type='srl_graph',
                                              allow_null_predicate = False)

        

        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'noun.bit'

        ifile = droot + 'vocab.src'
        instances = conll_reader.read(ifile)
        instances = ensure_list(instances)

        lemma_dict = Counter()

        for instance in tqdm(instances):
            lemmas = instance['metadata']['lemmas']
            print(lemmas)
            for lemma in lemmas:
                if lemma == Conll2009DatasetReader._EMPTY_LEMMA:
                    continue
                lemma_dict[lemma] += 1
        
        print('\n|vocab of lemmas| is {}'.format(len(conll_reader.lemma_set)))
        print('|vocab of lemmas| is {}'.format(len(lemma_dict)))

    """
    @pytest.mark.parametrize("lazy", (False,))
    @pytest.mark.parametrize("move", (False,))
    def test_read_from_file(self, lazy, move):
        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=move,
                                              instance_type='srl_graph',
                                              allow_null_predicate = False)
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ifile = droot + 'vocab.src'
        
        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'noun.bit'
        
        ofile = droot + 'all.moved.arg.vocab' 
        instances = conll_reader.read(ifile)
        instances = ensure_list(instances)
        
        arg_dict = Counter()

        for instance in tqdm(instances):
            lemmas = instance['metadata']['lemmas']
            labels = instance['srl_frames'].labels

            #print(labels)
            #print(lemmas)
            for idx, (label, lemma) in enumerate(zip(labels, lemmas)):
                if label == Conll2009DatasetReader._EMPTY_LABEL:
                    continue 
                if lemma == 'be':
                    for a, b in zip(lemmas, labels):
                        print(a, b)
                    import sys
                    sys.exit(0)
                arg_dict[lemma] += 1 
    """
    """ 
        with open(ofile, 'w') as fw:
            for k, v in arg_dict.most_common():
                fw.write('{}\t{}\n'.format(k, v))
    """
