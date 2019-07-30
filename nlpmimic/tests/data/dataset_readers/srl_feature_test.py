# pylint: disable=no-self-use,invalid-name
import pytest, sys
import itertools, json

from tqdm import tqdm
from collections import Counter
from collections import defaultdict 

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009bare import Conll2009BareDatasetReader
from nlpmimic.data.dataset_readers.conll2009bare import SrlStructure, SrlInstance 






class TestConll2009Reader(NlpMimicTestCase):

    #@pytest.mark.skip(reason="mute")
    def test_read_file(self):
        ofile = min_valid_lemmas = valid_srl_labels = None
        conll_reader = Conll2009BareDatasetReader(lazy=True, 
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              feature_labels=['pos', 'dep'], 
                                              moved_preposition_head=['IN', 'TO'],
                                              instance_type='srl_graph',
                                              maximum_length = 2019,
                                              flatten_number = True, # special treatment
                                              min_valid_lemmas = min_valid_lemmas,
                                              max_num_argument = 7, 
                                              valid_srl_labels = valid_srl_labels,
                                              allow_null_predicate = False)
        """
        fname = 'train.noun.morph.only'
        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"
        """

        fname = 'verb.bit'
        droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"

        ifile = droot + fname
        ofile = droot + fname + '.moved'
        
        with open(ofile, 'w') as fw:
            for srl in conll_reader.read_sentences(ifile):
                #pidx = srl.predicate_index
                #pfeature = SrlInstance(srl)
                #inst = SrlInstance(srl.tokens[pidx], srl.predicate)
                
                print(srl)
                print()
                sys.exit(0)


        """
        with open(ofile, 'w') as fw:
            for sentence in conll_reader._sentences(ifile):
                for head in conll_reader.moved_preposition_head:
                    sentence.move_preposition_head(moved_label=head)
                for _, _, frame in sentence.srl_frames:
                    for i, x in enumerate(frame):
                        if x == 'O':
                            frame[i] = '_'
                #print(sentence.format())
                fw.write(sentence.format() + '\n')
        """
