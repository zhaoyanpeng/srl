# pylint: disable=no-self-use,invalid-name,protected-access
import os
import subprocess

import torch
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.training.metrics import Metric
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from nlpmimic.data.dataset_readers import Conll2009Sentence
from nlpmimic.training.metrics import DependencyBasedF1Measure

class DependencyBasedF1Test(AllenNlpTestCase):

    def setUp(self):
        super(DependencyBasedF1Test, self).setUp()
        namespace = "srl_tags"
        vocab = Vocabulary()
        vocab.add_token_to_namespace("O", namespace)  # 0
        vocab.add_token_to_namespace("A0", namespace) # 1
        vocab.add_token_to_namespace("A1", namespace) # 2
        vocab.add_token_to_namespace("A2", namespace) # 3
        vocab.add_token_to_namespace("A3", namespace) # 4
        vocab.add_token_to_namespace("A4", namespace) # 5
        vocab.add_token_to_namespace("A5", namespace) # 6
        vocab.add_token_to_namespace("AM-ADV", namespace) # 7
        vocab.add_token_to_namespace("AM-MNR", namespace) # 8
        vocab.add_token_to_namespace("AM-TMP", namespace) # 9
        vocab.add_token_to_namespace("AM-LOC", namespace) #10
        vocab.add_token_to_namespace("AM-MOD", namespace) #11
        vocab.add_token_to_namespace("AM-DIS", namespace) #12
        vocab.add_token_to_namespace("AM-NEG", namespace) #13
        vocab.add_token_to_namespace("AM-PNC", namespace) #14
        vocab.add_token_to_namespace("AM-EXT", namespace) #15

        self.vocab = vocab

    def test_span_metrics_are_computed_correcly_with_prediction_map(self):
        # In this example, datapoint1 only has access to ARG1 and V labels,
        # whereas datapoint2 only has access to ARG2 and V labels.

        # gold_labels = [["O", "A0", "A1", "O", "A5", "O"],
        #               ["A2", "A4", "O", "A5", "A3", "O"]]
        gold_indices = [[0, 1, 2, 0, 4, 0], # 0, 1, 2, 0, 6, 0
                        [1, 3, 0, 4, 2, 0]] # 3, 5, 0, 6, 4, 0
        prediction_map_indices = [[0, 1, 2, 5, 6],
                                  [0, 3, 4, 5, 6]]

        gold_tensor = torch.Tensor(gold_indices)
        prediction_map_tensor = torch.Tensor(prediction_map_indices)
        
        # 0, 1, 4, 0, 4, 0 -> 0, 1, 6, 0, 6, 0 -> O, A0, A5, O,  A5, O
        # 0, 1, 0, 1, 4, 4 -> 0, 3, 0, 3, 6, 6 -> O, A2,  O, A2, A5, A5
        prediction_tensor = torch.rand([2, 6, 5])
        prediction_tensor[0, 0, 0] = 1
        prediction_tensor[0, 1, 1] = 1 # (True Positive - A0 
        prediction_tensor[0, 2, 4] = 1 # (False Positive - A5 # fn - A1
        prediction_tensor[0, 3, 0] = 1
        prediction_tensor[0, 4, 4] = 1 # (True Positive - A5 
        prediction_tensor[0, 5, 0] = 1  
        prediction_tensor[1, 0, 0] = 1 # (False Negative - A2
        prediction_tensor[1, 1, 1] = 1 # (False Positive - A2 # fn - A4
        prediction_tensor[1, 2, 0] = 1
        prediction_tensor[1, 3, 1] = 1 # (False Positive - A2 # fn - A5
        prediction_tensor[1, 4, 4] = 1 # (False Positive - A5 # fn - A3
        prediction_tensor[1, 5, 4] = 1 # (False Positive - A5 

        metric = DependencyBasedF1Measure(self.vocab, "srl_tags")
        metric(prediction_tensor, gold_tensor, prediction_map=prediction_map_tensor)

        assert metric._true_positives["A0"] == 1
        assert metric._true_positives["A5"] == 1
        assert "O" not in metric._true_positives.keys()
        assert metric._false_negatives["A1"] == 1
        assert metric._false_negatives["A2"] == 1
        assert metric._false_negatives["A3"] == 1 
        assert metric._false_negatives["A4"] == 1
        assert metric._false_negatives["A5"] == 1
        assert "O" not in metric._false_negatives.keys()
        assert metric._false_positives["A2"] == 2
        assert metric._false_positives["A5"] == 3 
        assert "O" not in metric._false_positives.keys()
        assert metric._uniqueness_err['A2'] == 1 
        assert metric._uniqueness_err['A5'] == 2
        
        # Check things are accumulating correctly.
        metric(prediction_tensor, gold_tensor, prediction_map=prediction_map_tensor)
        assert metric._true_positives["A0"] == 2
        assert metric._true_positives["A5"] == 2 
        assert "O" not in metric._true_positives.keys() # 4
        assert metric._false_negatives["A1"] == 2
        assert metric._false_negatives["A2"] == 2
        assert metric._false_negatives["A3"] == 2 
        assert metric._false_negatives["A4"] == 2
        assert metric._false_negatives["A5"] == 2
        assert "O" not in metric._false_negatives.keys() # 10
        assert metric._false_positives["A2"] == 4
        assert metric._false_positives["A5"] == 6 
        assert "O" not in metric._false_positives.keys() # 10
        assert metric._uniqueness_err['A2'] == 2 
        assert metric._uniqueness_err['A5'] == 4

        metric_dict = metric.get_metric()
        # tp + fn: for recall; tp + fp: for precision
        # total: 4 + 10; 4 + 10;
        # A5: 2 + 2; 2 + 6;
        numpy.testing.assert_almost_equal(metric_dict["recall-A2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-A2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-A2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-A5"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["precision-A5"], 0.25)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-A5"], 0.3333333333)
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 0.2857142857)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 0.2857142857)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 0.2857142857)
        
    def test_span_metrics_are_computed_correctly(self):
        gold_labels = ["O", "A0", "A1", "O", "A5", "O"]
        gold_indices = [self.vocab.get_token_index(x, "srl_tags") for x in gold_labels]

        gold_tensor = torch.Tensor([gold_indices])

        prediction_tensor = torch.rand([2, 6, self.vocab.get_vocab_size("srl_tags")])

        # Test that the span measure ignores completely masked sequences by
        # passing a mask with a fully masked row.
        mask = torch.LongTensor([[1, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0]])

        # 0, 1, 6, 0, 6, 0 -> O, A0, A5, O,  A5, O
        prediction_tensor[:, 0, 0] = 1
        prediction_tensor[:, 1, 1] = 1  # (True positive - A0
        prediction_tensor[:, 2, 6] = 1  # (False positive - A5 # fn - A1
        prediction_tensor[:, 3, 0] = 1
        prediction_tensor[:, 4, 6] = 1  # (True positive - A5
        prediction_tensor[:, 5, 0] = 1  

        metric = DependencyBasedF1Measure(self.vocab, "srl_tags")
        metric(prediction_tensor, gold_tensor, mask)

        assert metric._true_positives["A5"] == 1
        assert metric._true_positives["A1"] == 0
        assert metric._true_positives["A0"] == 1 
        assert "O" not in metric._true_positives.keys()
        assert metric._false_negatives["A5"] == 0
        assert metric._false_negatives["A1"] == 1
        assert "O" not in metric._false_negatives.keys()
        assert metric._false_positives["A5"] == 1
        assert metric._false_positives["A1"] == 0
        assert "O" not in metric._false_positives.keys()
        assert metric._uniqueness_err['A5'] == 1

        # Check things are accumulating correctly.
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives["A5"] == 2
        assert metric._true_positives["A1"] == 0
        assert metric._true_positives["A0"] == 2 
        assert "O" not in metric._true_positives.keys() # 4 
        assert metric._false_negatives["A5"] == 0
        assert metric._false_negatives["A1"] == 2
        assert "O" not in metric._false_negatives.keys() # 2
        assert metric._false_positives["A5"] == 2
        assert metric._false_positives["A1"] == 0
        assert "O" not in metric._false_positives.keys() # 2
        assert metric._uniqueness_err['A5'] == 2

        metric_dict = metric.get_metric()
        numpy.testing.assert_almost_equal(metric_dict["recall-A1"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-A1"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-A1"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-A5"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-A5"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-A5"], 0.666666666)
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 0.6666666666)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 0.666666666)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 0.6666666666)


    def test_span_f1_matches_perl_script_for_continued_arguments(self):
        srl_tags = ["O", "A0", "A1", "O", "A5", "O", "O"]
        sentence = ["Mark", "and", "Matt", "were", "running", "fast", "."]

        gold_indices = [self.vocab.get_token_index(x, "srl_tags") for x in srl_tags]
        gold_tensor = torch.Tensor([gold_indices])
        prediction_tensor = torch.rand([1, len(srl_tags), self.vocab.get_vocab_size("srl_tags")])
        mask = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])

        # Make prediction so that it is exactly correct.
        for i, tag_index in enumerate(gold_indices):
            prediction_tensor[0, i, tag_index] = 1

        metric = DependencyBasedF1Measure(self.vocab, "srl_tags")
        metric(prediction_tensor, gold_tensor, mask)
        metric_dict = metric.get_metric()

        assert metric._true_positives["A0"] == 1
        assert metric._true_positives["A1"] == 1
        assert metric._true_positives["A5"] == 1

        numpy.testing.assert_almost_equal(metric_dict["recall-A0"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-A0"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-A0"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-A1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-A1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-A1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-A5"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-A5"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-A5"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 1.0)

        # Check that the number of true positive ARG1 labels is the same as the perl script's output:
        gold_file_path = os.path.join(self.TEST_DIR, "gold_conll_eval.txt")
        prediction_file_path = os.path.join(self.TEST_DIR, "prediction_conll_eval.txt")
        with open(gold_file_path, "a+") as gold_file, open(prediction_file_path, "a+") as prediction_file:
            # Use the same bio tags as prediction vs gold to make it obvious by looking
            # at the perl script output if something is wrong.    
            sentence = Conll2009Sentence.initialize(4, sentence, srl_tags, sentence)
            text = sentence.format()
            print('\n{}\n'.format(text))
            gold_file.write(text)
            prediction_file.write(text)
            
        # Run the official perl script and collect stdout.
        perl_script_command = ["perl", str(self.TOOLS_ROOT / "eval09.pl"), '-s', prediction_file_path, 
                               '-g', gold_file_path, '2> /dev/null']
        # print('\n{}\n{}'.format(gold_file_path, prediction_file_path))
        print(perl_script_command)
        
        stdout = subprocess.check_output(perl_script_command, universal_newlines=True)
        stdout_lines = stdout.split("\n")
        
        #for line in stdout_lines[7:10]:
        #    print(line)
        
        precision = float(stdout_lines[7].split()[-2]) / 100.
        recall = float(stdout_lines[8].split()[-2]) / 100.
        F1 = float(stdout_lines[9].split()[-1]) / 100.
        
        assert F1 == 1.
         
