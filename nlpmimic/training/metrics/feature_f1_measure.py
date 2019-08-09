from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric

@Metric.register("feature_f1")
class FeatureBasedF1Measure(Metric):
    """
    Clustering F1 measure requires gold clusters and induced clusters.
    We need to build mapping from each gold cluster to a induced one that results a maximum overlap.
    and build mapping from each induced one to gold one that results a maximum overlap. 
    These two corresponds are similar to recall and precision measures, respectively.
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "srl_tags",
                 is_a_sentence: bool = False,
                 per_predicate: bool = False,  # metrics applied to each predicate
                 unlabeled_vals: bool = False, # considering the non-argument role
                 ignore_classes: List[str] = None) -> None:
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []
        self._unlabeled_vals = unlabeled_vals

        # recording gold and induced labels for each argument 
        self._gold_clusters: Dict[str, Set[int]] = defaultdict(set)
        self._induced_clusters: Dict[str, Set[int]] = defaultdict(set) 

        self._iargument: int = 0 # each argument has an unique index

        self._uniqueness_err: Dict[str, int] = defaultdict(int)
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

        self._ignored_labels = set(["AM-TMP", "AM-MNR", "AM-LOC", "AM-DIR"])
        self._ignored_labels = set() #

        ### below is the implementation of Ivan's evaluator
        self._is_a_sentence = is_a_sentence
        self._per_predicate = per_predicate
        self._clusters: Dict[str, Dict[str, [Dict, int]]] = defaultdict(dict)
        self._true_pos = self._true_neg = self._matched = 0
        self._one_cluster: Dict[str, Dict[str, int]] = defaultdict(dict)
    

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 predicates: torch.Tensor = None,
                 prediction_map: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask, prediction_map = \
            self.unwrap_to_tensors(predictions, gold_labels, mask, prediction_map)

        num_classes = predictions.size(-1)
        if not self._unlabeled_vals and (gold_labels >= num_classes).any():
            print(gold_labels)
            raise ConfigurationError("A gold label passed to SpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(num_classes))

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1]

        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())


        if self._per_predicate and not self._is_a_sentence:
            return self.eval_per_predicate(gold_labels, argmax_predictions, sequence_lengths, predicates)


        argmax_predictions = argmax_predictions.float()
        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            predicate = predicates[i].tolist()[0]
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            gold_string_labels = [self._label_vocabulary[label_id]
                                  for label_id in sequence_gold_label[:length].tolist()]
            gold_spans = [(string_tag, (index, index + 1)) for index, string_tag 
                          in enumerate(gold_string_labels) if string_tag != 'O']

            if not self._unlabeled_vals:  
                predicted_string_labels = [self._label_vocabulary[label_id]
                                       for label_id in sequence_prediction[:length].tolist()]
                predicted_spans = [(string_tag, (index, index + 1)) for index, string_tag 
                                   in enumerate(predicted_string_labels) if string_tag != "O"]
            else:
                predicted_string_labels = [self._label_vocabulary[label_id + 1]
                                       for label_id in sequence_prediction[:length].tolist()]
                predicted_spans = []
                for _, (index, _) in gold_spans:
                    string_tag = predicted_string_labels[index] 
                    predicted_spans.append((string_tag, (index, index + 1)))
            
            for gold, induced in zip(gold_spans, predicted_spans):
                self._iargument += 1 
                self._gold_clusters[gold[0]].add(self._iargument)

                induced_label = induced[0]
                if gold[0] in self._ignored_labels:
                    induced_label = gold[0] 
                self._induced_clusters[induced_label].add(self._iargument)

                self._matched += 1
                self._clusters[predicate].setdefault(gold, defaultdict(int))
                self._clusters[predicate][gold][induced] += 1
                self._one_cluster[gold].setdefault(induced, 0)
                self._one_cluster[gold][induced] += 1

    def eval_per_predicate(self, gold_labels, predictions, sequence_lengths, predicates):
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            length = sequence_lengths[i]
            predicate = predicates[i].tolist()[0]
            prediction = predictions[i, :length].tolist()
            gold_label = gold_labels[i, :length].tolist()

            if length == 0: continue

            # unnecessary to map id roles the str roles as in self.__call__() 
            for gold, induced in zip(gold_label, prediction):
                self._matched += 1

                self._clusters[predicate].setdefault(gold, defaultdict(int))
                self._clusters[predicate][gold][induced] += 1

                self._one_cluster[gold].setdefault(induced, 0)
                self._one_cluster[gold][induced] += 1

    def cal_pu(self, rets):
        predictions = set() 
        for vals in rets.values():
            predictions = predictions | set(vals.keys())

        pu = 0.
        for prediction in predictions:
            this_pu = 0.
            for vals in rets.values():
                cnt = vals.get(prediction, 0)
                if cnt > this_pu:
                    this_pu = cnt
            pu += this_pu
        return pu

    def cal_co(self, rets):
        predictions = set() 
        for vals in rets.values():
            predictions = predictions | set(vals.keys())
        
        co = 0.
        for vals in rets.values():
            this_co = 0.
            for prediction in predictions:
                cnt = vals.get(prediction, 0)
                if cnt > this_co:
                    this_co = cnt
            co += this_co
        return co 
    
    def cal_f1(self, pu, co, c = 1e-15):
        pu += c
        pu /= (float(self._matched) + c)

        co += c
        co /= (float(self._matched) + c)

        f1 = 2 * pu * co / (pu + co)
        return pu, co, f1

    def get_per_metric(self, reset: bool = False):
        c = 1e-15 
        pu = co = 0
        for _, v in self._clusters.items():
            pu += self.cal_pu(v)
            co += self.cal_co(v)
        pu, co, f1 = self.cal_f1(pu, co)

        all_metrics = {}
        all_metrics["pu"] = pu 
        all_metrics["co"] = co 
        all_metrics["f1"] = f1 
        
        pu = self.cal_pu(self._one_cluster)
        co = self.cal_co(self._one_cluster)
        pu, co, f1 = self.cal_f1(pu, co)

        all_metrics["f1-overall"] = f1 

        if reset: self.reset()
        return all_metrics

    def get_metric(self, reset: bool = False):
        if self._per_predicate or self._is_a_sentence:
            return self.get_per_metric(reset = reset)

        c  = 0 
        pu = 0.
        for _, induced in self._induced_clusters.items():
            this_pu = 0 
            for _, gold in self._gold_clusters.items():
                overlap = induced & gold
                this_pu = len(overlap) if len(overlap) > this_pu else this_pu
            pu += this_pu
        pu += c
        pu /= (float(self._iargument) + c)
        
        co = 0.
        for _, gold in self._gold_clusters.items():
            this_co = 0
            for _, induced in self._induced_clusters.items():
                overlap = induced & gold
                this_co = len(overlap) if len(overlap) > this_co else this_co
            co += this_co
        co += c
        co /= (float(self._iargument) + c)

        all_metrics = {}
        all_metrics["pu-overall"] = pu 
        all_metrics["co-overall"] = co 
        all_metrics["f1-overall"] = 2 * pu * co / (pu + co) 

        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision, recall, f1_measure = -float('inf')
        return precision, recall, f1_measure

    def reset(self):
        self._gold_clusters = defaultdict(set)
        self._induced_clusters = defaultdict(set) 
        self._iargument: int = 0 # each argument has an unique index


        self._clusters = defaultdict(dict)
        self._true_pos = self._true_neg = self._matched = 0
        self._one_cluster = defaultdict(dict)
