# pylint: disable=no-self-use,invalid-name
import pytest

from tqdm import tqdm
from collections import Counter
from collections import defaultdict 

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2003Reader(NlpMimicTestCase):


    @pytest.mark.skip(reason="mute")
    def test_read_from_file(self):
        valid_srl_labels = ["A0", "A1", "A2", "A3", "A4", "A5", "AM-ADV", "AM-CAU", "AM-DIR", 
                            "AM-EXT", "AM-LOC", "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"]
        valid_srl_labels = set(valid_srl_labels)
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ofile = droot + 'all.moved.arg.vocab' 
        min_valid_lemmas = 0.5
        conll_reader = Conll2009DatasetReader(lazy=False, 
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=True,
                                              instance_type='srl_graph',
                                              maximum_length = 80,
                                              min_valid_lemmas = min_valid_lemmas,
                                              max_num_argument = 7, 
                                              valid_srl_labels = valid_srl_labels,
                                              allow_null_predicate = False)

        
        word = 'v20.0' 
        name = 'train.verb'
        droot = "/disk/scratch1/s1847450/data/conll09/morph.word/"
        ifile = droot + '{}/{}'.format(word, name)

        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'noun.bit'

        #ifile = droot + 'vocab.src'
        instances = conll_reader.read(ifile)

        """
        instances = ensure_list(instances)

        lemma_dict = Counter()

        for instance in tqdm(instances):
            lemmas = instance['metadata']['lemmas']
            print(lemmas)
            labels = instance['srl_frames']
            print(labels)
            for lemma in lemmas:
                if lemma == Conll2009DatasetReader._EMPTY_LEMMA:
                    continue
                lemma_dict[lemma] += 1
        
        print('\n|vocab of lemmas| is {}'.format(len(conll_reader.lemma_set)))
        print('|vocab of lemmas| is {}'.format(len(lemma_dict)))
        """
