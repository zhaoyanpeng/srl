# pylint: disable=no-self-use,invalid-name
import pytest, re
from tqdm import tqdm
from collections import Counter

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

from nlpmimic.training import util  
    
class TestConll2009Reader():
    @pytest.mark.parametrize("lazy", (False,))
    @pytest.mark.parametrize("move", (True,))
    def test_vocabulary_file(self, lazy, move):
        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=move,
                                              instance_type='srl_graph',
                                              allow_null_predicate = False)

        

        droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        ifile = droot + 'noun.bit'

        instances = conll_reader.read(ifile)
        instances = ensure_list(instances)

        lemma_dict = Counter()

        for instance in tqdm(instances):
            arg_indices = instance['argument_indices'].sequence_index 
            tokens = instance['metadata']['tokens']
            print(arg_indices)
       
        util.shuffle_argument_indices(instances)


        for instance in tqdm(instances):
            arg_indices = instance['argument_indices'].sequence_index 
            tokens = instance['metadata']['tokens']
            print(arg_indices)

        util.shuffle_argument_indices(instances)


        for instance in tqdm(instances):
            arg_indices = instance['argument_indices'].sequence_index 
            tokens = instance['metadata']['tokens']
            print(arg_indices)
