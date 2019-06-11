from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric


@Metric.register("dependency_f1")
class DependencyBasedF1Measure(Metric):
    """
    The Conll 2008/2009 SRL metrics are based on exact label matching. This metric
    implements dependency-based precision and recall metrics for a normal sequence
    tagging scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2008/2009 
    data - particularly, (I did NOT read the perl code, thus I do not know what exact
    difference between the perl script and this implementation could be). However, 
    it is a close proxy, which can be helpful for judging model performance during 
    training. 

    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "srl_tags",
                 unlabeled_vals: bool = False,
                 ignore_classes: List[str] = None) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a normal sequence tagging format is used 
            in which the labels are of the format: ["A0", "A1", "O"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             ``ignore_classes=["V"]``
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        """

        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []
        self._unlabeled_vals = unlabeled_vals

        # These will hold per label span counts.
        self._uniqueness_err: Dict[str, int] = defaultdict(int)
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 prediction_map: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        prediction_map: ``torch.Tensor``, optional (default = None).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            ``vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))``,
            rather than simply ``vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))``.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask, prediction_map = self.unwrap_to_tensors(predictions,
                                                                                gold_labels,
                                                                                mask, prediction_map)

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

        argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
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
            """ 
            print('\n{}\n{}\n'.format(gold_string_labels, predicted_string_labels))
            print('\n{}\n{}\n'.format(gold_spans, predicted_spans))
            print(sequence_prediction[:length]) 
            print(sequence_gold_label[:length])
            """

            duplicate_labels = defaultdict(int)
            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1

                duplicate_labels[span[0]] += 1 
            
            for k, v in duplicate_labels.items():
                if v > 1:
                    self._uniqueness_err[k] += 1

            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1
        
        # output the last one
        #print('\n{}\n{}\n'.format(gold_string_labels, predicted_string_labels))

    def arg_metric(self,
                   predictions: torch.Tensor,
                   gold_labels: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   prediction_map: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask, prediction_map = self.unwrap_to_tensors(predictions,
                                                                                gold_labels,
                                                                                mask, prediction_map)

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

        argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            gold_string_labels = [self._label_vocabulary[label_id]
                                  for label_id in sequence_gold_label[:length].tolist()]
            gold_spans = [(string_tag, (index, index + 1)) for index, string_tag in enumerate(gold_string_labels)]

            if not self._unlabeled_vals:  
                predicted_string_labels = [self._label_vocabulary[label_id]
                                       for label_id in sequence_prediction[:length].tolist()]
                predicted_spans = [(string_tag, (index, index + 1)) for index, string_tag 
                                   in enumerate(predicted_string_labels) if string_tag != "O"]
            else:
                predicted_string_labels = [self._label_vocabulary[label_id]
                                       for label_id in sequence_prediction[:length].tolist()]
                predicted_spans = []
                for _, (index, _) in gold_spans:
                    string_tag = predicted_string_labels[index] 
                    predicted_spans.append((string_tag, (index, index + 1)))
            """
            print('\n{}\n{}\n'.format(gold_string_labels, predicted_string_labels))
            print('\n{}\n{}\n'.format(gold_spans, predicted_spans))
            print(sequence_prediction[:length]) 
            print(sequence_gold_label[:length])
            """

            duplicate_labels = defaultdict(int)
            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1

                duplicate_labels[span[0]] += 1 
            
            for k, v in duplicate_labels.items():
                if v > 1:
                    self._uniqueness_err[k] += 1

            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1
        
        # output the last one
        #print('\n{}\n{}\n'.format(gold_string_labels, predicted_string_labels))

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        uniqueness_err = 0
        for label, count in self._uniqueness_err.items():
            if len(label) == 2 and label[0] == 'A':
                uniqueness_err += count 
        
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        all_metrics["uniqueness_err_overall"] = uniqueness_err 
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._uniqueness_err = defaultdict(int)
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
