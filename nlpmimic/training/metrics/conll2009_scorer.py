from typing import List
import logging
import os
import tempfile
import subprocess
import shutil
import numpy
import re

from collections import defaultdict
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_EVAL_DIR = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "tools"))

@Metric.register("conll2009")
class Conll2009Scorer(Metric):
    """
    This class uses the external eval09.pl software for computing a broad range of metrics
    on syntactic and semantic dependencies. Here, we use it to compute the Precision, Recall,
    and F1 metrics. You can download the source for eval09.py from here: 
    <https://ufal.mff.cuni.cz/conll2009-st/scorer.html>.

    Note that this metric reads and writes from disk quite a bit. You probably don't
    want to include it in your training loop; instead, you should calculate this on
    a validation set only.

    Parameters
    ----------
    eval09_directory_path : ``str``, required.
        The directory containing the eval09.pl.
    """
    def __init__(self,
                 eval09_directory_path: str = DEFAULT_EVAL_DIR) -> None:
        self._eval09_directory_path = eval09_directory_path
        self._eval09_program_path = os.path.join(eval09_directory_path, "eval09.pl")

        self._recall = 0.0
        self._precision = 0.0
        self._f1_measure = 0.0

        self._regx = "\s*?(\d+\.?\d*?)\s*?\|"
        self._header_sd_root = "(i.e., predicate identification and classification)"
        self._header_sd_nonroot = "(i.e., identification and classification of arguments)"
        
        self._gold_check: Dict[str, int] = defaultdict(int)
        self._system_check: Dict[str, int] = defaultdict(int)
        self._correct_check: Dict[str, int] = defaultdict(int)
        
        self._gold_args: Dict[str, int] = defaultdict(int)
        self._system_args: Dict[str, int] = defaultdict(int)
        self._correct_args: Dict[str, int] = defaultdict(int)
        
        self._gold_root: Dict[str, int] = defaultdict(int)
        self._system_root: Dict[str, int] = defaultdict(int)
        self._correct_root: Dict[str, int] = defaultdict(int)

    @overrides
    def __call__(self, 
                 predicted_samples: List[Conll2009Sentence], 
                 gold_samples: List[Conll2009Sentence]) -> None: # type: ignore
        """
        Parameters
        ----------
        predicted_samples : ``List[Conll2009Sentence]``
            A list of predicated Conll 2009 sentences to compute score for.
        gold_samples : ``List[Conll2009Sentence]``
            A list of gold Conll 2009 sentences to use as a reference.
        """
        if not os.path.exists(self._eval09_program_path):
            logger.warning(f"eval09.pl not found at {self._eval09_program_path}, now trying downloading it.")
            try:
                uri = 'https://ufal.mff.cuni.cz/conll2009-st/eval09.pl'
                command = ['wget', uri, '-P', self._eval09_directory_path]
                subprocess.run(command, shell=True, check=True)
            except Exception as e: 
                pass
            if not os.path.exists(self._evalb_program_path):
                raise ConfigurationError(f"eval09.pl not found at {self._eval09_program_path}. "
                                "And downloading failed. You must download it and put it there before using it.")

        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, "gold.txt")
        predicted_path = os.path.join(tempdir, "predicted.txt")
        
        # write into file 
        with open(gold_path, "w") as gold_file:
            for sentence in gold_samples:
                gold_file.write(sentence.format())
        with open(predicted_path, "w") as predicted_file:
            for sentence in predicted_samples:
                predicted_file.write(sentence.format())
        
        command = ["perl", "-X", self._eval09_program_path, '-s', predicted_path, '-g', gold_path, '2> /dev/null']
        
        self.reset_check_dict() # reset checking tool

        #try:
        stdout = subprocess.check_output(command, universal_newlines=True)
        stdout_lines = stdout.split("\n")

        self._f1_measure = float(stdout_lines[9].split()[-1]) / 100. 
        self._precision = float(stdout_lines[7].split()[-2]) / 100.
        self._recall = float(stdout_lines[8].split()[-2]) / 100.
        #print(self._f1_measurei, self._precision, self._recall)
        
        sd_root, sd_nonroot = False, False
        
        for line in stdout_lines:
            stripped = line.strip()
            
            if sd_root:
                if stripped == '':
                    sd_root = False
                    #print('------------  end sd_root')
                    continue
                
                columns = stripped.split("|", 1)
                numbers = re.findall(self._regx, columns[1])
                
                if len(numbers) != 6: 
                    #print('---------------------------->> {}'.format(stripped))
                    continue
                else:
                    numbers = list(map(float, numbers))
                    label = columns[0].strip()
                    #print(label)
                    self._gold_root[label] += numbers[0]
                    self._system_root[label] += numbers[2]
                    self._correct_root[label] += numbers[1]
                    
                    self._gold_check[label] += numbers[0]
                    self._system_check[label] += numbers[2]
                    self._correct_check[label] += numbers[1]
                    #print(stripped)

            if sd_nonroot:
                if stripped == '':
                    sd_nonroot = False
                    #print('------------  end sd_nonroot')
                    continue
                
                columns = stripped.split("|", 1)
                numbers = re.findall(self._regx, columns[1])
                
                if len(numbers) != 6: 
                    #print('---------------------------->> {}'.format(stripped))
                    continue
                else:
                    numbers = list(map(float, numbers))
                    label = columns[0].strip().split("+")[1].strip()
                    #print(label)
                    self._gold_args[label] += numbers[0]
                    self._system_args[label] += numbers[2]
                    self._correct_args[label] += numbers[1]
                    
                    self._gold_check[label] += numbers[0]
                    self._system_check[label] += numbers[2]
                    self._correct_check[label] += numbers[1]
                    #print(stripped)

            if stripped == self._header_sd_root:
                sd_root = True
                #print('\n------------begin sd_root')
            if stripped == self._header_sd_nonroot:
                sd_nonroot = True
                #print('\n------------begin sd_nonroot')
        #except Exception as e:
        #    self.reset()
        #    logger.info("errors encountered")

        shutil.rmtree(tempdir)
        
    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average precision, recall and f1.
        """
        tp = sum(self._correct_check.values())
        tp_and_fp = sum(self._system_check.values())
        tp_and_fn = sum(self._gold_check.values())
        
        #print(tp, tp_and_fp, tp_and_fn)
        precision, recall, f1_measure = self._compute_metrics(tp, tp_and_fp, tp_and_fn)
        
        numpy.testing.assert_almost_equal(precision, self._precision, decimal=3)
        numpy.testing.assert_almost_equal(recall, self._recall, decimal=3)
        numpy.testing.assert_almost_equal(f1_measure, self._f1_measure, decimal=3)
        
        tp = sum(self._correct_args.values()) + sum(self._correct_root.values())
        tp_and_fp = sum(self._system_args.values()) + sum(self._system_root.values())
        tp_and_fn = sum(self._gold_args.values()) + sum(self._gold_root.values())

        #print(tp, tp_and_fp, tp_and_fn)
        self._precision, self._recall, self._f1_measure = self._compute_metrics(tp, tp_and_fp, tp_and_fn)
        precision, recall, f1_measure = self._precision, self._recall, self._f1_measure 

        if reset:
            self.reset()
        return {"eval09_recall": recall, "eval09_precision": precision, "eval09_f1_measure": f1_measure}

    @staticmethod
    def _compute_metrics(true_positives: int, tp_and_fp: int, tp_and_fn: int):
        precision = float(true_positives) / float(tp_and_fp + 1e-13)
        recall = float(true_positives) / float(tp_and_fn + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure
    
    @overrides
    def reset(self):
        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0
        
        self._gold_args: Dict[str, int] = defaultdict(int)
        self._system_args: Dict[str, int] = defaultdict(int)
        self._correct_args: Dict[str, int] = defaultdict(int)
        
        self._gold_root: Dict[str, int] = defaultdict(int)
        self._system_root: Dict[str, int] = defaultdict(int)
        self._correct_root: Dict[str, int] = defaultdict(int)
    
    def reset_check_dict(self):
        self._recall = 0.0
        self._precision = 0.0
        self._f1_measure = 0.0
        
        self._gold_check: Dict[str, int] = defaultdict(int)
        self._system_check: Dict[str, int] = defaultdict(int)
        self._correct_check: Dict[str, int] = defaultdict(int)

    @staticmethod
    def clean_eval(eval09_directory_path: str = DEFAULT_EVAL_DIR):
        os.system("rm {}".format(os.path.join(eval09_directory_path, "eval09.pl")))
