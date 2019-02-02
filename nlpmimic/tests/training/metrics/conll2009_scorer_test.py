# pylint: disable=no-self-use,invalid-name,protected-access
import numpy
from nltk import Tree

from allennlp.common.testing import AllenNlpTestCase
from nlpmimic.data.dataset_readers import Conll2009DatasetReader
from nlpmimic.data.dataset_readers import Conll2009Sentence
from nlpmimic.training.metrics import Conll2009Scorer


class Conll2009ScorerTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
    
    def test_eval09_correctly_scores_identical_trees(self):
        srl_tags = ["O", "A0", "A1", "O", "A5", "O", "O"]
        sentence = ["Mark", "and", "Matt", "were", "running", "fast", "."]

        sentence = Conll2009Sentence.initialize(4, sentence, srl_tags, sentence)


        conll2009_scorer = Conll2009Scorer()
        conll2009_scorer([sentence], [sentence])
        metrics = conll2009_scorer.get_metric()
        
        #assert metrics["eval09_recall"] == 1.0
        #assert metrics["eval09_precision"] == 1.0
        #assert metrics["eval09_f1_measure"] == 1.0
        numpy.testing.assert_almost_equal(metrics["eval09_recall"], 1.0)
        numpy.testing.assert_almost_equal(metrics["eval09_precision"], 1.0)
        numpy.testing.assert_almost_equal(metrics["eval09_f1_measure"], 1.0)
    
    def test_output_reader(self):
        conll_reader = Conll2009DatasetReader(lazy=True, 
                                              feature_labels=['pos', 'dep'], 
                                              move_preposition_head=True,
                                              allow_null_predicate = True)
        gold = '/disk/scratch1/s1847450/data/conll09/yy'
        pred = '/disk/scratch1/s1847450/data/conll09/xx'
        gold_instances = conll_reader.read(gold)
        pred_instances = conll_reader.read(pred)

        gold_sentences = []
        pred_sentences = []
        for gg, pp in zip(gold_instances, pred_instances):
            g = Conll2009Sentence.instance(gg)
            p = Conll2009Sentence.instance(pp)
            gold_sentences.append(g)
            pred_sentences.append(p)

        conll2009_scorer = Conll2009Scorer()
        conll2009_scorer(pred_sentences, gold_sentences)
        
        metrics = conll2009_scorer.get_metric()
        print()
        print(metrics)
       
        
        conll2009_scorer(pred_sentences, gold_sentences)
        metrics = conll2009_scorer.get_metric()
        print(metrics)


