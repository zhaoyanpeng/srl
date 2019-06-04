# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

from nlpmimic.data.dataset_readers.conllontonotes import OntonotesDatasetReader

class TestOntonotesSrlReader:
    @pytest.mark.parametrize("lazy", (False,))
    def test_read_from_file(self, lazy):
        conll_reader = OntonotesDatasetReader(lazy=lazy)

        droot = '/disk/scratch1/s1847450/data/conllon/conll-formatted-ontonotes-5.0/data/' 
        #portion = 'development' # 2082/noun & 33215/verb & 0/adj 
        #portion = 'test'        # 1865/noun & 24850/verb & 0/adj 
        #portion = 'conll-2012-test' # 1803/noun & 22659/verb & 0/adj 
        portion = 'train'        # 14555/noun & 238515/verb & 0/adj 

        #droot = '/afs/inf.ed.ac.uk/user/s18/s1847450/Code/allennlp_shit/allennlp/tests/fixtures/'
        #portion = 'conll_2012/subdomain/'

        froot = droot + portion
        instances = conll_reader.read(froot)

        cnt_noun, cnt_verb, cnt_adj = 0, 0, 0
        for instance in instances:
            pos_tags = instance['metadata']['pos_tags']
            pis = instance['verb_indicator'].labels
            if 1 not in pis:
                #print(instance)
                continue
            idx = pis.index(1)
            pos = pos_tags[idx] 
            #print(idx, pis)
            #print(pos, pos_tags)
            
            if 'NN' in pos: 
                cnt_noun += 1 
            elif 'JJ' in pos:
                cnt_adj  += 1
            else:
                cnt_verb += 1
        print('{}/noun & {}/verb & {}/adj'.format(cnt_noun, cnt_verb, cnt_adj))

