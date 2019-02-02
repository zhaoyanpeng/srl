from typing import Dict, Tuple, List, Optional, NamedTuple, Any
from overrides import overrides

import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from nltk import Tree

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules import TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.models.model import Model
from allennlp.models.constituency_parser import SpanInformation
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import EvalbBracketingScorer, DEFAULT_EVALB_DIR
from allennlp.common.checks import ConfigurationError


@Model.register("dio_rae")
class DioRaeParser(Model):
    """
    This ``SpanConstituencyParser`` simply encodes a sequence of text
    with a stacked ``Seq2SeqEncoder``, extracts span representations using a
    ``SpanExtractor``, and then predicts a label for each span in the sequence.
    These labels are non-terminal nodes in a constituency parse tree, which we then
    greedily reconstruct.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    span_extractor : ``SpanExtractor``, required.
        The method used to extract the spans from the encoded sequence.
    encoder : ``Seq2SeqEncoder``, required.
        The encoder that we will use in between embedding tokens and
        generating span representations.
    feedforward : ``FeedForward``, required.
        The FeedForward layer that we will use in between the encoder and the linear
        projection to a distribution over span labels.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    evalb_directory_path : ``str``, optional (default=``DEFAULT_EVALB_DIR``)
        The path to the directory containing the EVALB executable used to score
        bracketed parses. By default, will use the EVALB included with allennlp,
        which is located at allennlp/tools/EVALB . If ``None``, EVALB scoring
        is not used.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 evalb_directory_path: str = DEFAULT_EVALB_DIR) -> None:
        super(DioRaeParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.feedforward_layer = TimeDistributed(feedforward) if feedforward else None
        self.pos_tag_embedding = pos_tag_embedding or None 

        self.token_table = list(self.vocab._retained_counter['tokens'].items())
        self.token_distr = F.normalize(torch.Tensor([f for _, f in self.token_table]).pow(0.75), dim=0) 
        self.token_index = [self.vocab.get_token_index(t, namespace='tokens') for t, _ in self.token_table]

        representation_dim = text_field_embedder.get_output_dim()

        if evalb_directory_path is not None:
            self._evalb_score = EvalbBracketingScorer(evalb_directory_path)
        else:
            self._evalb_score = None
        initializer(self)

    
    def sample(self, nsample: int, tokens: torch.LongTensor):
        tokens = tokens.data.tolist()
        nword, words = len(tokens), []
        
        for i in range(nword):
            neg_ws = {tokens[i]} 
            while len(neg_ws) < nsample + 1:
                w = torch.multinomial(self.token_distr, 1).tolist()[0]
                neg_ws.add(self.token_index[w])
            neg_ws.remove(tokens[i])
            neg_ws = list(neg_ws)
            words.append(neg_ws)
            
        '''
        for i, token in enumerate(tokens):
            print(token)
            print(words[i])
            print('--------------------')
        
        neg_ws = torch.tensor(words, dtype=torch.long)
        neg_ws = {'tokens': neg_ws}
        print(neg_ws)
        
        vec_neg_ws = self.text_field_embedder(neg_ws)
        print(vec_neg_ws.size())
        print(vec_neg_ws)

        sum_neg_ws = torch.sum(vects_neg_ws, dim=1)
        print(sum_neg_ws.size())
        print(sum_neg_ws)
        '''

        neg_ws = torch.tensor(words, dtype=torch.long)
        neg_ws = {'tokens': neg_ws}
           
        vec_neg_ws = self.text_field_embedder(neg_ws)
        vec_neg_ws = self.feedforward_layer(vec_neg_ws)
        #sum_neg_ws = torch.sum(vec_neg_ws, dim=1)
        
        return vec_neg_ws

    
    def loss(self, v_outs, neg_vs, pos_vs, nsample, bsize=15,  average='batch'):
        diff = neg_vs - pos_vs
        diff.transpose_(1, 2) # word * dim_o * nsample
        v_outs = torch.unsqueeze(v_outs, 1) # word * 1 * dim_i
        
        rets = torch.bmm(v_outs, diff) + torch.tensor(1.)
        rets = torch.clamp(rets, min=0)
        if average == 'token':
            nword = rets.size()[0]
            loss = rets.sum() / (nsample * nword * bsize)
        else:
            loss = rets.sum() / (nsample * bsize)
        return loss


    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                spans: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                pos_tags: Dict[str, torch.LongTensor] = None,
                span_labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        spans : ``torch.LongTensor``, required.
            A tensor of shape ``(batch_size, num_spans, 2)`` representing the
            inclusive start and end indices of all possible spans in the sentence.
        metadata : List[Dict[str, Any]], required.
            A dictionary of metadata for each batch element which has keys:
                tokens : ``List[str]``, required.
                    The original string tokens in the sentence.
                gold_tree : ``nltk.Tree``, optional (default = None)
                    Gold NLTK trees for use in evaluation.
                pos_tags : ``List[str]``, optional.
                    The POS tags for the sentence. These can be used in the
                    model as embedded features, but they are passed here
                    in addition for use in constructing the tree.
        pos_tags : ``torch.LongTensor``, optional (default = None)
            The output of a ``SequenceLabelField`` containing POS tags.
        span_labels : ``torch.LongTensor``, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
            spans, of shape ``(batch_size, num_spans)``.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
            representing a distribution over the label classes per span.
        spans : ``torch.LongTensor``
            The original spans tensor.
        tokens : ``List[List[str]]``, required.
            A list of tokens in the sentence for each element in the batch.
        pos_tags : ``List[List[str]]``, required.
            A list of POS tags in the sentence for each element in the batch.
        num_spans : ``torch.LongTensor``, required.
            A tensor of shape (batch_size), representing the lengths of non-padded spans
            in ``enumerated_spans``.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        projected_input = self.feedforward_layer(embedded_text_input)
        
        mask = get_text_field_mask(tokens)
        
        if self.training:
            v_outs = self.encoder(projected_input, mask)
            v_outs = torch.squeeze(v_outs, 0)
            if not self.training:
                print('v_out: ', v_outs.size())

            neg_vs = self.sample(3, tokens['tokens'][0]) 
            if not self.training:
                print('n_out: ', neg_vs.size())
                print(neg_vs)

            pos_vs = projected_input[0]
            pos_vs = torch.unsqueeze(pos_vs, 1)
            if not self.training:
                print('p_out: ', pos_vs.size())
                print(pos_vs)

            loss = self.loss(v_outs, neg_vs, pos_vs, nsample=3, average='batch') 
            if not self.training: 
                import sys
                sys.exit(0)

            output_dict = {"spans": spans, "loss": loss}
        else:
            v_outs = self.encoder(projected_input, mask)
            # loss = self.get_metric(gold_spans, predicted_spans)
            loss = torch.tensor(1.0)
            output_dict = {"spans": spans, "predicted_spans:": v_outs, "loss": loss}    
        return output_dict
    

    @overrides
    def decode(self, chart: Dict[str, List[Dict]]):
        embedded_text_input = self.text_field_embedder(tokens)
        projected_input = self.feedforward_layer(embedded_text_input)
        
        mask = get_text_field_mask(tokens)
        
        self.encoder.set_test(True)
        spans = self.encoder(projected_input, mask)
        self.encoder.set_test(False)
        
        return spans
        
