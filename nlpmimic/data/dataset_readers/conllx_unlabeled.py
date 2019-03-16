from typing import Any, Type, TypeVar, Dict, List, Sequence, Iterable, Optional, Tuple
import itertools 
import inspect 
import logging
import re

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.from_params import remove_optional
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("conllx_unlabeled")
class ConllxUnlabeledDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format ():
    
    TOKEN_ID TOKEN LEMMA PLEMMA POS PPOS ...
    
    Return only gold features by default
    
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """
    _DUMMY = '_'
    _EMPTY_LABEL = 'O'
    _RE_SENSE_ID = '(^.*?)\.(\d+\.?\d*?)$'
    _VALID_LABELS = {'dep', 'pos'}
    _DEFAULT_INSTANCE_TYPE = 'basic' # srl_gan
    _DEFAULT_APPENDIX_TYPE = 'nyt_infer' # nyt_learn

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lemma_indexers: Dict[str, TokenIndexer] = None,
                 feature_labels: Sequence[str] = (),
                 move_preposition_head: bool = False,
                 allow_null_predicate: bool = False,
                 instance_type: str = _DEFAULT_INSTANCE_TYPE,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._lemma_indexers = lemma_indexers or {'lemmas': SingleIdTokenIndexer(namespace='lemmas')} 
        for label in feature_labels:
            if label not in self._VALID_LABELS: 
                raise ConfigurationError("unknown feature label type: {}".format(label))
        
        self.feature_labels = set(feature_labels)
        self.move_preposition_head = move_preposition_head
        self.allow_null_predicate = allow_null_predicate
        self.instance_type = instance_type

    @overrides
    def _read(self, context_path: str, appendix_path: str, appendix_type: str = _DEFAULT_APPENDIX_TYPE) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        context_path = cached_path(context_path)
        appendix_path = cached_path(appendix_path)
        cnt: int = 0
        
        for sentence in self._sentences(context_path, appendix_path, appendix_type): 
            lemmas = sentence.predicted_lemmas
            tokens = [Token(t) for t in sentence.tokens]
            pos_tags = sentence.predicted_pos_tags
            
            head_ids = sentence.head_ids
            dep_rels = sentence.dep_rels
            #cnt += 1
            #print('\n{}\n{}\n'.format(cnt, sentence.format()))
            if self.move_preposition_head:
                sentence.move_preposition_head()
            
            #if cnt == 168:
            #    import sys
            #    sys.exit(0)

            if not sentence.srl_frames:    
                if not self.allow_null_predicate:
                    continue  

                labels = [self._EMPTY_LABEL for _ in tokens]
                predicate_indicators = [0 for _ in tokens]
                yield self.text_to_instance(tokens, 
                                            lemmas,
                                            labels, 
                                            predicate_indicators, 
                                            None, 
                                            None, 
                                            None,
                                            pos_tags,
                                            head_ids,
                                            dep_rels) 
            else:
                for (predicate_index, predicate, labels) in sentence.srl_frames:
                    srl_labels = list(set(labels))
                    if not self.allow_null_predicate and \
                        len(srl_labels) == 1 and srl_labels[0] == self._EMPTY_LABEL:
                        continue
                    predicate_indicators = [0 for _ in labels]
                    predicate_indicators[predicate_index] = 1
                    predicate_sense = sentence.predicate_senses[predicate_index]
                    yield self.text_to_instance(tokens, 
                                                lemmas, 
                                                labels,
                                                predicate_indicators, 
                                                predicate, 
                                                predicate_index,
                                                predicate_sense,
                                                pos_tags,
                                                head_ids,
                                                dep_rels)

    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
                         lemmas: List[str],
                         labels: List[str],
                         predicate_indicators: List[int],
                         predicate: Optional[str] = None,
                         predicate_index: Optional[int] = None,
                         predicate_sense: Optional[float] = None,
                         pos_tags: List[str] = None,
                         head_ids: List[int] = None,
                         dep_rels: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        if self.instance_type == self._DEFAULT_INSTANCE_TYPE:
            # pylint: disable=arguments-differ
            fields: Dict[str, Field] = {}
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            fields['tokens'] = text_field
            fields['srl_frames'] = SequenceLabelField(labels, text_field, 'srl_tags')
            fields['predicate_indicators'] = SequenceLabelField(predicate_indicators, text_field)
            
            metadata = {'tokens': [x.text for x in tokens],
                        'lemmas': lemmas,
                        'predicate': predicate,
                        'predicate_index': predicate_index,
                        'predicate_sense': predicate_sense}
            
            if 'pos' in self.feature_labels and not pos_tags:
                raise ConfigurationError("Dataset reader was specified to use pos_tags as "
                                         "features. Pass them to text_to_instance.")
            if 'dep' in self.feature_labels and (not head_ids or not dep_rels):
                raise ConfigurationError("Dataset reader was specified to use dep_rels as "
                                         "features. Pass head_ids and dep_rels to text_to_instance.")

            if 'pos' in self.feature_labels and pos_tags:
                metadata['pos_tags'] = pos_tags
            if 'dep' in self.feature_labels and head_ids and dep_rels:
                metadata['head_ids'] = head_ids
                metadata['dep_rels'] = dep_rels
            
            fields['metadata'] = MetadataField(metadata)
            return Instance(fields)
        elif self.instance_type == "srl_gan":
            # pylint: disable=arguments-differ
            fields: Dict[str, Field] = {}
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            fields['tokens'] = text_field
            
            lemma_tokens = [Token(t) for t in lemmas]
            lemma_field = TextField(lemma_tokens, token_indexers=self._lemma_indexers)
            fields['lemmas'] = lemma_field
            
            predicate_tokens = [self._DUMMY if i == 0 else predicate for i in predicate_indicators]
            fields['predicates'] = SequenceLabelField(predicate_tokens, text_field, 'predicates')
            
            fields['srl_frames'] = SequenceLabelField(labels, text_field, 'srl_tags')
            fields['predicate_indicators'] = SequenceLabelField(predicate_indicators, text_field)

            metadata = {'tokens': [x.text for x in tokens],
                        'lemmas': lemmas,
                        'predicate': predicate,
                        'predicate_index': predicate_index,
                        'predicate_sense': predicate_sense}
            
            if 'pos' in self.feature_labels and not pos_tags:
                raise ConfigurationError("Dataset reader was specified to use pos_tags as "
                                         "features. Pass them to text_to_instance.")
            if 'dep' in self.feature_labels and (not head_ids or not dep_rels):
                raise ConfigurationError("Dataset reader was specified to use dep_rels as "
                                         "features. Pass head_ids and dep_rels to text_to_instance.")

            if 'pos' in self.feature_labels and pos_tags:
                metadata['pos_tags'] = pos_tags
            if 'dep' in self.feature_labels and head_ids and dep_rels:
                metadata['head_ids'] = head_ids
                metadata['dep_rels'] = dep_rels
            
            fields['metadata'] = MetadataField(metadata)
            return Instance(fields)
        else:
            raise ConfigurationError("unknown appendix type: {}".format(label))
            
    def _sentences(self, context_path: str, appendix_path: str, appendix_type: str) -> Iterable[Conll2009Sentence]:
        """
        """
        with open(context_path, "r", encoding='utf8') as context_file, \
            open(appendix_path, "r", encoding='utf8') as appendix_file:
            logger.info("Reading contexts from lines in file at: %s", context_path)
            logger.info("Reading appendix from lines in file at: %s", appendix_path)
            conll_rows = []
            for context, appendix in zip(context_file, appendix_file):
                context = context.strip()
                appendix = appendix.strip()

                line = ' '.join([context, appendix]).strip()

                if not line and conll_rows:
                    yield self._conll_rows_to_sentence(conll_rows, appendix_type)
                    conll_rows = []
                    continue
                if line: conll_rows.append(line)
            if conll_rows:
                yield self._conll_rows_to_sentence(conll_rows, appendix_type)

    def _conll_rows_to_sentence(self, conll_rows: List[str], appendix_type: str) -> Conll2009Sentence:
        # Token counter, starting at 1 for each new sentence
        token_ids: List[int] = []
        # Form or pubctuation symbol
        tokens: List[str] = []
        # Gold-standard lemmas of the words
        lemmas: List[str] = []
        # Automatically predicted lemmas of the words
        predicted_lemmas: List[str] = []
        # Gold-standard pos tags
        pos_tags: List[str] = []
        # Predicted pos tags
        predicted_pos_tags: List[str] = []
        # Gold-standard morphological features
        features: List[Optional[str]] = []
        # Predicted morphological features
        predicted_features: List[Optional[str]] = []
        # Gold-standard syntactic head of the current word
        head_ids: List[int] = []
        # Predicted syntactic head
        predicted_head_ids: List[int] = []
        # Gold-standard syntactic dependency relation to head
        dep_rels: List[str] = []
        # Predicted syntax deendency relation to predicted head
        predicted_dep_rels: List[str] = []
        # With 'Y' indicating a argument-bearing token (predicate)
        predicate_indicators: List[Optional[str]] = []
        # Lemma of the predicate coming from a current token, and sense id
        predicate_lemmas: List[Optional[str]] = []
        # Sense ids extracted from the predicate lemmas
        predicate_senses: List[Optional[float]] = []
        
        # Place holder for srl frames
        span_labels: List[List[str]] = []
        predicates: List[str] = []
        
        if appendix_type == self._DEFAULT_APPENDIX_TYPE:
            for index, row in enumerate(conll_rows):
                columns = row.split()
                 
                if len(columns) < 9:
                    hint = '\n{}\n\n'.format(row.strip())
                    for row in conll_rows:
                        hint += row.strip() + '\n'
                    raise ValueError('see hints (the incorrect data piece) below: {}'.format(hint)) 

                token_ids.append(int(columns[0]))
                tokens.append(columns[1])
                lemmas.append(columns[2])
                predicted_lemmas.append(columns[3])
                pos_tags.append(columns[4])
                predicted_pos_tags.append(columns[5])

                # optional
                features.append(columns[6] if columns[6] != self._DUMMY else None)
                predicted_features.append(columns[7] if columns[7] != self._DUMMY else None)
                
                head_ids.append(None)
                predicted_head_ids.append(None)
                dep_rels.append(None)
                predicted_dep_rels.append(None)
                
                # optional
                predicate = columns[8]
                predicate_indicators.append('Y' if predicate != self._DUMMY else None)
                if predicate != self._DUMMY:
                    lemma_sense = re.match(self._RE_SENSE_ID, predicate).groups()
                    plemma = lemma_sense[0] 
                    psense = None 
                    predicates.append((index, plemma))
                else:
                    plemma, psense = None, None
                predicate_lemmas.append(plemma)
                predicate_senses.append(psense)
                
                if not span_labels:
                    span_labels = [[] for _ in columns[9:]]
                for column, item in zip(span_labels, columns[9:]):
                    column.append(item if item != self._DUMMY else self._EMPTY_LABEL)
            
            span_labels = [[self._EMPTY_LABEL] * len(tokens) for _ in predicates]
            srl_frames = [(idx, predicate, labels) for (idx, predicate), labels 
                                     in zip(predicates, span_labels)]
        elif appendix_type == 'nyt_learn':
            for index, row in enumerate(conll_rows):
                columns = row.split()
                 
                if len(columns) < 9:
                    hint = '\n{}\n\n'.format(row.strip())
                    for row in conll_rows:
                        hint += row.strip() + '\n'
                    raise ValueError('see hints (the incorrect data piece) below: {}'.format(hint)) 

                token_ids.append(int(columns[0]))
                tokens.append(columns[1])
                lemmas.append(columns[2])
                predicted_lemmas.append(columns[3])
                pos_tags.append(columns[4])
                predicted_pos_tags.append(columns[5])

                # optional
                features.append(columns[6] if columns[6] != self._DUMMY else None)
                predicted_features.append(columns[7] if columns[7] != self._DUMMY else None)
                
                head_ids.append(None)
                predicted_head_ids.append(None)
                dep_rels.append(None)
                predicted_dep_rels.append(None)
                
                # optional
                predicate = columns[8]
                predicate_indicators.append('Y' if predicate != self._DUMMY else None)
                if predicate != self._DUMMY:
                    lemma_sense = re.match(self._RE_SENSE_ID, predicate).groups()
                    plemma = lemma_sense[0] 
                    psense = None 
                    predicates.append((index, plemma))
                else:
                    plemma, psense = None, None
                predicate_lemmas.append(plemma)
                predicate_senses.append(psense)
                
                if not span_labels:
                    span_labels = [[] for _ in columns[9:]]
                for column, item in zip(span_labels, columns[9:]):
                    column.append(item if item != self._DUMMY else self._EMPTY_LABEL)
            
            srl_frames = [(idx, predicate, labels) for (idx, predicate), labels 
                                     in zip(predicates, span_labels)]
        else:
            raise ConfigurationError("unknown appendix type: {}".format(appendix_type))
            
        return Conll2009Sentence(token_ids,
                                 tokens,
                                 lemmas,
                                 predicted_lemmas,
                                 pos_tags,
                                 predicted_pos_tags,
                                 features,
                                 predicted_features,
                                 head_ids,
                                 predicted_head_ids,
                                 dep_rels,
                                 predicted_dep_rels,
                                 predicate_indicators,
                                 predicate_lemmas,
                                 predicate_senses,
                                 srl_frames)
