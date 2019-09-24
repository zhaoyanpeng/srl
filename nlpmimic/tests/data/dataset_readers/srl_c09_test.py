# pylint: disable=no-self-use,invalid-name
import sys, pytest

from tqdm import tqdm
from collections import Counter
from collections import defaultdict 

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

class TestConll2003Reader(NlpMimicTestCase):
    

    #@pytest.mark.skip(reason="mute")
    def test_move_head(self):
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

        """ 
        word = 'v100.0' 
        name = 'devel.verb'
        droot = "/disk/scratch1/s1847450/data/conll09/morph.word/"
        ifile = droot + '{}/{}'.format(word, name)
        ofile = ifile + '.moved' 
        """

        fname = 'devel.noun.morph.only'
        fname = 'train.verb'
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"
        droot = "/disk/scratch1/s1847450/data/conll09/morph.stem/"

        #fname = 'verb.bit'
        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"

        ifile = droot + fname
        ofile = droot + fname + '.moved'

        
        """
        fname = 'CoNLL2009-ST-English-train.txt'
        droot = "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/"
        ifile = droot + fname
        ofile = droot + fname + '.moved'
        """

        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'verb.bit'
        
        conll_reader.flatten_number = True
        with open(ofile, 'w') as fw:
            for sentence in conll_reader._sentences(ifile):
                for head in conll_reader.moved_preposition_head:
                    sentence.move_preposition_head(moved_label=head)
                for _, _, frame in sentence.srl_frames:
                    for i, x in enumerate(frame):
                        if x == 'O':
                            frame[i] = '_'
                #print(sentence.format())
                sentence.lemmas = conll_reader.filter_lemmas(sentence.lemmas, sentence)
                fw.write(sentence.format() + '\n')
            
        


    @pytest.mark.skip(reason="mute")
    def test_read_from_file(self):
        valid_srl_labels = ["A0", "A1", "A2", "A3", "A4", "A5", "AM-ADV", "AM-CAU", "AM-DIR", 
                            "AM-EXT", "AM-LOC", "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"]
        
        valid_srl_labels = ["A0", "A1", "A2", "A3", "A4", "A5", "AA", 
                            "AM-ADV", "AM-CAU", "AM-DIR", "AM-DIS", "AM-EXT", "AM-LOC", "AM-MNR", 
                            "AM-MOD", "AM-NEG", "AM-PNC", "AM-PRD", "AM-PRT", "AM-REC", "AM-TMP"],

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

        
        #word = 'v20.0' 
        #name = 'train.verb'
        #droot = "/disk/scratch1/s1847450/data/conll09/morph.word/"
        #ifile = droot + '{}/{}'.format(word, name)

        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'noun.bit'

        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"
        ifile = droot + 'noun.bit'

        #ifile = droot + 'vocab.src'
        instances = conll_reader.read(ifile)

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
