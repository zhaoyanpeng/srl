# pylint: disable=no-self-use,invalid-name
import pytest, sys
import itertools, json

from tqdm import tqdm
from collections import Counter
from collections import defaultdict 

from allennlp.common.util import ensure_list
from nlpmimic.common.testing import NlpMimicTestCase
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

def arg_lemma(instances, use_sense = True):
    cnt, ipred, stats = 0, 0, defaultdict(dict)
    for sent in instances:
        if not sent.srl_frames:    
            continue
        cnt += 1
        for pid, pred, frame in sent.srl_frames:
            pargs = frame 
            ipred += 1

            if not use_sense:
                pred = pred.split('.')[0]
            
            if pred in stats: # allocate mem
                this_pred = stats[pred]
            else:
                this_pred = defaultdict(dict)
                stats[pred] = this_pred

            for idx, parg in enumerate(pargs): # syntax-arguments
                if parg == 'O':
                    continue
                syntx = sent.predicted_lemmas[idx]

                if parg in this_pred: # allocate mem
                    this_parg = this_pred[parg]
                else:
                    this_parg = defaultdict(int)
                    this_pred[parg] = this_parg
                this_parg[syntx] += 1
    return stats

def write_arg_syntax(stats, ofile):
    with open(ofile, 'w') as fw:
        for k, v in stats.items():
            data = {k: v}
            json.dump(data, fw)
            fw.write('\n') 

def main_arg_lemma():
    ofile = min_valid_lemmas = valid_srl_labels = None
    conll_reader = ConllxUnlabeledDatasetReader(lazy = True,
                                          lemma_file = ofile,
                                          lemma_use_firstk = 5,
                                          feature_labels=['pos', 'dep'], 
                                          instance_type='srl_graph',
                                          maximum_length = 2019,
                                          min_valid_lemmas = min_valid_lemmas,
                                          max_num_argument = 7, 
                                          valid_srl_labels = valid_srl_labels,
                                          allow_null_predicate = False)

    word = "v100.0" 
    droot = "/disk/scratch1/s1847450/data/nytimes/"

    ctx_name = 'nytimes.45.only'
    apx_name = 'nytimes.45.verb.only'

    #ctx_name = 'nytimes.45.only.small'
    #apx_name = 'nytimes.45.verb.only.small'

    context_file =  droot + "morph.only/{}".format(ctx_name)
    appendix_file = droot + "morph.only/{}".format(apx_name)
    
    instances = conll_reader._sentences(context_file, appendix_path=appendix_file, 
                                        appendix_type='nyt_learn')

    ifile = instances
    ofile = droot + 'lemmasign/{}.model'.format(ctx_name) 

    use_sense = False    
    stats = arg_lemma(ifile, use_sense = use_sense)
    write_arg_syntax(stats, ofile)

def main_arg_lemma_new():
    ofile = None
    firstk = sys.maxsize
    valid_srl_labels = None
    min_valid_lemmas = None
    
    conll_reader = ConllxUnlabeledDatasetReader(lazy = False,
                                          lemma_file = ofile,
                                          lemma_use_firstk = 5,
                                          feature_labels=['pos', 'dep'], 
                                          instance_type='srl_graph',
                                          maximum_length = 80,
                                          min_valid_lemmas = min_valid_lemmas,
                                          max_num_argument = 7, 
                                          valid_srl_labels = valid_srl_labels,
                                          allow_null_predicate = False)

    droot = "/disk/scratch1/s1847450/data/nytimes.new/morph.stem/"
    ctx_name = "nyt.verb.20.10.1000"
    context_file =  droot + ctx_name 


    print(context_file)
    instances = conll_reader._sentences(context_file, appendix_path=None, 
                                   appendix_type='nyt_learn')
    ifile = instances
    ofile = droot + 'lemma.stem/{}.model'.format(ctx_name) 

    use_sense = False    
    stats = arg_lemma(ifile, use_sense = use_sense)
    write_arg_syntax(stats, ofile)

class TestConll2003Reader(NlpMimicTestCase):

    #@pytest.mark.skip(reason="mute")
    def test_move_head(self):
        #main_arg_lemma()
        main_arg_lemma_new()

    @pytest.mark.skip(reason="mute")
    def test_conllx_reader(self):
        valid_srl_labels = ["A0", "A1", "A2", "A3", "A4", "A5", "AM-ADV", "AM-CAU", "AM-DIR", 
                            "AM-EXT", "AM-LOC", "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"]
        valid_srl_labels = set(valid_srl_labels)
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ofile = droot + 'all.moved.arg.vocab' 
        firstk = 100000
        min_valid_lemmas = 0.5 

        ofile = None
        valid_srl_labels = None
        min_valid_lemmas = None
        
        conll_reader = ConllxUnlabeledDatasetReader(lazy = False,
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              feature_labels=['pos', 'dep'], 
                                              instance_type='srl_graph',
                                              maximum_length = 80,
                                              min_valid_lemmas = min_valid_lemmas,
                                              max_num_argument = 7, 
                                              valid_srl_labels = valid_srl_labels,
                                              allow_null_predicate = False)

        droot = "/disk/scratch1/s1847450/data/nytimes.new/morph.word/"
        context_file =  droot + "nyt.verb.20.10.1000.s"
        
        print(context_file)
        instances = conll_reader._read(context_file, appendix_path=None, 
                                       appendix_type='nyt_learn',
                                       firstk = firstk)
        for inst in instances:
            print(inst)
         

    @pytest.mark.skip(reason="mute")
    def test_read_from_conllx_file(self):
        valid_srl_labels = ["A0", "A1", "A2", "A3", "A4", "A5", "AM-ADV", "AM-CAU", "AM-DIR", 
                            "AM-EXT", "AM-LOC", "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"]
        valid_srl_labels = set(valid_srl_labels)
        droot = "/disk/scratch1/s1847450/data/conll09/separated/"
        ofile = droot + 'all.moved.arg.vocab' 
        
        firstk = 100000
        min_valid_lemmas = 0.5 
        conll_reader = ConllxUnlabeledDatasetReader(lazy = False,
                                              lemma_file = ofile,
                                              lemma_use_firstk = 5,
                                              feature_labels=['pos', 'dep'], 
                                              instance_type='srl_graph',
                                              maximum_length = 80,
                                              min_valid_lemmas = min_valid_lemmas,
                                              max_num_argument = 7, 
                                              valid_srl_labels = valid_srl_labels,
                                              allow_null_predicate = False)

        word = "v100.0" 
        droot = "/disk/scratch1/s1847450/data/nytimes/morph.word/"
        context_file =  droot + "{}/nytimes.verb.ctx".format(word)
        appendix_file = droot + "{}/nytimes.verb.sel".format(word)
        
        #droot = '/disk/scratch1/s1847450/data/nyt_annotated/xchen/' 
        #context_file = droot + 'nytimes.45.lemma.small'
        #appendix_file = droot + 'nytimes.verb.small.picked'
        
        instances = conll_reader._read(context_file, appendix_path=appendix_file, 
                                       appendix_type='nyt_learn',
                                       firstk = firstk)
    
        instances = ensure_list(instances)
        
        """
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
