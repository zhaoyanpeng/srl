"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""
from typing import Tuple, Set, Dict, List, TextIO, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Linear, Dropout

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

from nlpmimic.nn.util import gumbel_softmax, gumbel_sinkhorn
from nlpmimic.training.metrics import FeatureBasedF1Measure 
from nlpmimic.training.metrics import ClusteringBasedF1Measure 
from nlpmimic.training.metrics import DependencyBasedF1Measure

@Model.register("srl_inf_feate")
class SrlVaeFeateClassifier(Model):
    """ A srl classifier make predictions of p(y|x).
    """
    def __init__(self, vocab: Vocabulary,
                 feate_embedder: TextFieldEmbedder = None,
                 argmt_embedder: TextFieldEmbedder = None,
                 predt_embedder: TextFieldEmbedder = None,
                 label_embedder: Embedding = None,
                 seq_encoder: Seq2SeqEncoder = None,
                 psign_dim: int = None, # psign: predicate sign bit (0/1) 
                 tau: float = None,
                 tunable_tau: bool = False,
                 suppress_nonarg: bool = False,
                 seq_projection_dim: int = None, 
                 feate_dropout: float = 0.,
                 argmt_dropout: float = 0.,
                 label_dropout: float = 0.,
                 predt_dropout: float = 0.,
                 label_smoothing: float = None,
                 embed_lemma_ctx: bool = False,
                 metric_type: str = 'dependency',
                 ignore_span_metric: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SrlVaeFeateClassifier, self).__init__(vocab, regularizer)
        self.signature = 'classifier'
        
        self.seq_encoder = seq_encoder
        self.feate_embedder = feate_embedder
        self.argmt_embedder = argmt_embedder
        self.label_embedder = label_embedder
        self.predt_embedder = predt_embedder
        if psign_dim is not None:
            self.psign_embedder = Embedding(2, psign_dim)

        self.feate_dropout = Dropout(p=feate_dropout)
        self.argmt_dropout = Dropout(p=argmt_dropout)
        self.label_dropout = Dropout(p=label_dropout)
        self.predt_dropout = Dropout(p=predt_dropout)

        self.tau = None
        self.minimum_tau = 1e-5 
        if tau is not None:
            self.tau = torch.nn.Parameter(torch.tensor(tau), requires_grad=tunable_tau) 
        # another representation of lemmas
        if embed_lemma_ctx:
            embedder = getattr(self.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
            nlemma, dim = embedder.weight.size() 
            # similar to in/out embeddings
            self.lectx_embedder = torch.nn.Embedding(nlemma, dim) 
            torch.nn.init.xavier_normal_(self.lectx_embedder.weight)
        
        # feature space to label space
        self.suppress_nonarg = suppress_nonarg
        if not self.suppress_nonarg:
            self.nclass = self.vocab.get_vocab_size("srl_tags")
        else: # FIXME: assumming non-argument label has id 0, should ensure this in the configuration file
            self.nclass = self.vocab.get_vocab_size("srl_tags") - 1

        self.label_projection_layer = None
        if self.seq_encoder is not None:
            self.label_projection_layer = TimeDistributed(
                Linear(self.seq_encoder.get_output_dim(), self.nclass))
        
        # feature space transformation
        self.seq_projection_dim = seq_projection_dim
        self.seq_projection_layer = None
        if self.seq_projection_dim is not None:
            self.seq_projection_layer = \
                Linear(self.seq_encoder.get_output_dim(), self.seq_projection_dim)
        
        self.ignore_span_metric = ignore_span_metric

        params = {'unlabeled_vals': self.suppress_nonarg, 'tag_namespace': "srl_tags", 'per_predicate': True}
        self.span_metric = FeatureBasedF1Measure(vocab, **params) 

        self._label_smoothing = label_smoothing

        self.init()

    def init(self):
        pass 

    def forward(self) -> Dict[str, torch.Tensor]:
        return None  

    def encode_global_predt(self, token: str = 'GLOBAL_PREDT', device=None):
        predt_idx = self.vocab.get_token_index(token, namespace="predts")
        predt_idx = {'predts': torch.tensor(predt_idx, device=device)}
        embedded_predt = self.predt_embedder(predt_idx)
        embedded_predt = self.predt_dropout(embedded_predt)
        return embedded_predt

    def encode_lemma(self, lemmas: Dict[str, torch.LongTensor]):
        embedded_lemmas = self.argmt_embedder(lemmas)
        embedded_lemmas = self.argmt_dropout(embedded_lemmas)  
        return embedded_lemmas
    
    def encode_all(self, 
                   predicate: Dict[str, torch.LongTensor] = None,
                   arguments: Dict[str, torch.LongTensor] = None,
                   feate_ids: Dict[str, torch.LongTensor] = None):
        e_p = m_p = e_a = m_a = e_f = m_f = None
        if predicate is not None:
            e_p = self.predt_embedder(predicate)
            e_p = self.predt_dropout(e_p)
            m_p = get_text_field_mask(predicate)
        if arguments is not None:
            e_a = self.argmt_embedder(arguments)
            e_a = self.argmt_dropout(e_a)
            m_a = get_text_field_mask(arguments)
        if feate_ids is not None:
            e_f = self.feate_embedder(feate_ids)
            e_f = self.feate_dropout(e_f)
            m_f = get_text_field_mask(feate_ids)
        return e_p, m_p, e_a, m_a, e_f, m_f

    def labeled_loss(self,
                     mask: torch.Tensor,
                     logits: torch.Tensor,
                     labels: torch.LongTensor,
                     average: str = 'batch'):
        
        loss_ce = sequence_cross_entropy_with_logits(
            logits, labels, mask, average=average, label_smoothing=self._label_smoothing)
        return loss_ce 

    def add_outputs(self, 
                    mask: torch.Tensor,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    output_dict: Dict[str, torch.Tensor], 
                    predicates: torch.Tensor = None, 
                    metadata: List[Dict[str, Any]] = None) -> None:
        if labels is None: 
            raise ConfigurationError("Prediction loss required but gold labels `labels` is None.")
        output_dict["gold_srl"] = labels # decoded in `decode` for the purpose of debugging

        if not self.ignore_span_metric:
            self.span_metric(logits, labels, mask=mask, predicates=predicates)

        if metadata is not None: 
            list_predts, list_argmts, list_labels = \
                zip(*[(x["p_lemma"], x["a_lemmas"], x["a_roles"]) for x in metadata])
            output_dict["argmts"] = list(list_argmts)
            output_dict["labels"] = list(list_labels)
            output_dict["predts"] = list(list_predts)
        return None

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        namespace = 'srl_tags'
        if 'logits' in output_dict:
            all_predictions = output_dict['logits']
        elif 'logits_softmax' in output_dict:
            all_predictions = output_dict['logits_softmax']
        else:
            raise ConfigurationError("unavailable logits for decoding predictions")
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
        # discard useless stuff in the output dict 
        returned_dict = {"argmts": output_dict["argmts"], 
                         "labels": output_dict["labels"],
                         "predts": output_dict["predts"]}
        batch_size = all_predictions.size(0)
        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(batch_size)]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        #print(self.vocab._token_to_index['srl_tags'])
        isent = 0
        for predictions, length in zip(predictions_list, sequence_lengths):
            scores, max_likelihood_sequence = torch.max(predictions[:length], 1) 
            max_likelihood_sequence = max_likelihood_sequence.tolist() 
            
            #tags = [self.vocab.get_token_from_index(x, namespace=namespace) for x in max_likelihood_sequence]
            tags = [str(x) for x in max_likelihood_sequence]

            isent += 1

            all_tags.append(tags)
        returned_dict["srl_tags"] = all_tags

        # gold srl labels
        gold_srl = []
        srl_frames = output_dict["gold_srl"]
        srl_frames = [srl_frames[i].detach().cpu() for i in range(srl_frames.size(0))]
        for srls, length in zip(srl_frames, sequence_lengths):
            tags = [self.vocab.get_token_from_index(x, namespace="srl_tags") for x in srls.tolist()[:length]]
            gold_srl.append(tags)
        returned_dict["gold_srl"] = gold_srl 
        return returned_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            return {} # Return an empty dictionary if ignoring the span metric
        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "f1" in x or "co" in x or "pu" in x}

