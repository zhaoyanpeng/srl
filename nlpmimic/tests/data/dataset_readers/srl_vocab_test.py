# pylint: disable=no-self-use,invalid-name
import pytest, re, itertools, json

from tqdm import tqdm
from collections import Counter
from collections import defaultdict 

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

    
class TestConll2009Reader():

    
    @pytest.mark.skip(reason="mute")
    @pytest.mark.parametrize("lazy", (True,))
    @pytest.mark.parametrize("move", (True,))
    def test_label_file(self, lazy, move):
        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"
        ofile = droot + 'verb.all.moved.arg.vocab' 
        pfile = droot + "verb.all.predicate.vocab"
        
        mlen = 80
        marg = 7

        #ofile = pfile = None 
        #mlen = 800
        #marg = 70


        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              predicate_file = pfile,
                                              predicate_use_firstk = 20,
                                              maximum_length = mlen,
                                              valid_srl_labels = ["A0", "A1", "A2", "A3", "A4", "A5", "AA", 
                                                "AM-ADV", "AM-CAU", "AM-DIR", "AM-DIS", "AM-EXT", "AM-LOC", "AM-MNR", 
                                                "AM-MOD", "AM-NEG", "AM-PNC", "AM-PRD", "AM-PRT", "AM-REC", "AM-TMP"],
                                              feature_labels=['pos', 'dep'], 
                                              max_num_argument = marg,
                                              move_preposition_head=move,
                                              instance_type='srl_graph',
                                              allow_null_predicate = False)

        
        ignored_labels = set(["AM-TMP", "AM-MNR", "AM-LOC", "AM-DIR"])

        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"
        ifile = droot + 'train.verb'

        instances = conll_reader.read(ifile)

        label_dict = defaultdict(int) 

        for instance in tqdm(instances):
            labels = instance['srl_frames'].labels
            for i, label in enumerate(labels):
                label_dict[label] += 1   
             
        print()
        for k, v in label_dict.items():
            print('{}\t{}'.format(k, v))

        s = 0
        for k in ignored_labels:
            s += label_dict[k]
        print('{} = {}'.format(ignored_labels, s))

    @pytest.mark.skip(reason="mute")
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

    @pytest.mark.skip(reason="mute")
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



    @pytest.mark.skip(reason="mute")
    @pytest.mark.parametrize("lazy", (True,))
    @pytest.mark.parametrize("move", (True,))
    def test_vocab_file(self, lazy, move):
        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"
        ofile = droot + 'verb.all.moved.arg.vocab' 
        pfile = droot + "verb.all.predicate.vocab"

        conll_reader = Conll2009DatasetReader(lazy=lazy, 
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              predicate_file = pfile,
                                              predicate_use_firstk = 1500,
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=move,
                                              instance_type='srl_graph',
                                              allow_null_predicate = False)

        

        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'noun.bit'

        ifile = droot + 'verb.vocab.src'
        instances = conll_reader.read(ifile)
        instances = ensure_list(instances)

        predicate_dict = Counter()

        for instance in tqdm(instances):
            predicate = instance['metadata']['predicate']
            predicate_dict[predicate] += 1
        
        print('\n|vocab of predicates| is {}'.format(len(conll_reader.predicate_set)))
        print('|vocab of predicates| is {}'.format(len(predicate_dict)))

    #@pytest.mark.skip(reason="mute")
    @pytest.mark.parametrize("lazy", (True,))
    @pytest.mark.parametrize("move", (True,))
    def test_predicate_file(self, lazy, move):
        conll_reader = Conll2009DatasetReader(lazy=True, 
                                              feature_labels=['pos', 'dep'], 
                                              moved_preposition_head=['IN', 'TO'],
                                              instance_type='srl_graph',
                                              flatten_number=True,
                                              allow_null_predicate = False)

        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'noun.bit'

        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"
        droot = "/disk/scratch1/s1847450/data/conll09/lemma.only/"
        #ifile = droot + 'verb.vocab.src'
        #ofile = droot + 'verb.all.predicate.vocab'

        #ifile = droot + 'vocab.src'

        ifile = droot + 'all.morph.only.vocab.src'

        ifile = droot + 'all.morph.only.moved.vocab.src'
        ofile = droot + 'all.morph.only.predicate.vocab.json' 


        instances = conll_reader.read(ifile)

        predicate_dict = defaultdict(dict) 
        for instance in tqdm(instances):
            predicate = instance['metadata']['predicate']
            predicate_sense = instance['metadata']['predicate_sense']

            if predicate not in predicate_dict:
                predicate_dict[predicate] = Counter()
            predicate_dict[predicate][predicate_sense] += 1
        
        print('|vocab of predicate| is {}'.format(len(predicate_dict)))
        
        freq_dict = Counter()
        for k, v in predicate_dict.items():
            freq_dict[k] += sum(v.values())
        
        with open(ofile, 'w') as fw:
            for k, v in freq_dict.most_common():
                data = (k, v, predicate_dict[k])
                json.dump(data, fw)
                fw.write('\n')


    @pytest.mark.skip(reason="mute")
    @pytest.mark.parametrize("lazy", (True,))
    @pytest.mark.parametrize("move", (True,))
    def test_vocabulary_file(self, lazy, move):
        conll_reader = Conll2009DatasetReader(lazy=True, 
                                              feature_labels=['pos', 'dep'], 
                                              moved_preposition_head=['IN', 'TO'],
                                              instance_type='srl_graph',
                                              flatten_number=True,
                                              allow_null_predicate = False)
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ifile = droot + 'vocab.src'
        
        #droot = "/disk/scratch1/s1847450/data/conll09/bitgan/"
        #ifile = droot + 'noun.bit'

        droot = "/disk/scratch1/s1847450/data/conll09/morph.only/"

        droot = "/disk/scratch1/s1847450/data/conll09/lemma.only/"
        #ifile = droot + 'noun.vocab.src'
        ifile = droot + 'all.morph.only.vocab.src'

        ifile = droot + 'all.morph.only.moved.vocab.src'
        
        #ofile = droot + 'noun.all.moved.arg.vocab' 
        ofile = droot + 'all.morph.only.moved.arg.vocab.json' 
        instances = conll_reader.read(ifile)
        
        arg_dict = Counter()

        for instance in tqdm(instances):
            lemmas = instance['metadata']['lemmas']
            labels = instance['srl_frames'].labels

            #print(labels)
            #print(lemmas)
            for idx, (label, lemma) in enumerate(zip(labels, lemmas)):
                if label == Conll2009DatasetReader._EMPTY_LABEL:
                    continue 
                """
                if lemma == 'be':
                    for a, b in zip(lemmas, labels):
                        print(a, b)
                    import sys
                    sys.exit(0)
                """
                arg_dict[lemma] += 1 
        
        #return

        with open(ofile, 'w') as fw:
            for k, v in arg_dict.most_common():
                fw.write('{}\t{}\n'.format(k, v))

