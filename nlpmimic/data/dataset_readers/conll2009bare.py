from typing import Any, Type, TypeVar, Set, Dict, List, Sequence, Iterable, Optional, Tuple
import itertools 
import inspect 
import logging
import re

from overrides import overrides
from collections import Counter

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.from_params import remove_optional
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from nlpmimic.data.fields import IndexField
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("conll2009_bare")
class Conll2009BareDatasetReader(DatasetReader):
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
    _EMPTY_LEMMA = 'M' # masked lemma
    _RE_SENSE_ID = '(^.*?)\.(\d+\.?\d*?)$'
    _VALID_LABELS = {'dep', 'pos'}
    _DEFAULT_INSTANCE_TYPE = 'basic'
    _MAX_NUM_ARGUMENT = 7 

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lemma_indexers: Dict[str, TokenIndexer] = None,
                 lemma_file: str = None,
                 lemma_use_firstk: int = 5000, # used as frequency when it is smaller than 100
                 feature_labels: Sequence[str] = (),
                 maximum_length: float = float('inf'),
                 valid_srl_labels: Sequence[str] = (),
                 move_preposition_head: bool = False,
                 allow_null_predicate: bool = False,
                 max_num_argument: int = 100,
                 instance_type: str = _DEFAULT_INSTANCE_TYPE,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._MAX_NUM_ARGUMENT = max_num_argument

        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._lemma_indexers = lemma_indexers or {'lemmas': SingleIdTokenIndexer(namespace='lemmas')} 
        for label in feature_labels:
            if label not in self._VALID_LABELS: 
                raise ConfigurationError("unknown feature label type: {}".format(label))
        
        self.maximum_length = maximum_length
        self.valid_srl_labels = valid_srl_labels

        self.feature_labels = set(feature_labels)
        self.move_preposition_head = move_preposition_head
        self.allow_null_predicate = allow_null_predicate
        self.instance_type = instance_type
        
        try:
            self.lemma_set = None
            if lemma_file is not None:
                lemma_dict = Counter() 
                with open(lemma_file, 'r') as lemmas:
                    for line in lemmas:
                        k, v = line.strip().split()
                        lemma_dict[k] += int(v)
                # construct vocab
                lemma_set = set()
                if lemma_use_firstk > 100: # first k most common lemmas
                    for idx, (k, v) in enumerate(lemma_dict.most_common()):
                        if idx >= lemma_use_firstk: break
                        lemma_set.add(k)
                else: # lemmas with a frequency above 'lemma_use_firstk'
                    for k, v in lemma_dict.most_common():
                        if v < lemma_use_firstk: continue
                        lemma_set.add(k)
                self.lemma_set = lemma_set
        except Exception as e:
            logger.info("Reading vocabulary of lemmas failed: %s", lemma_file)
            self.lemma_set = None
    
    def filter_lemmas(self, lemmas: List[str]) -> List[str]:
        if self.lemma_set is not None:
            lemmas = [lemma if lemma in self.lemma_set else self._EMPTY_LEMMA for lemma in lemmas] 
        return lemmas

    @overrides
    def _read(self, file_path: str) -> Iterable[Conll2009Sentence]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        cnt: int = 0
        for sentence in self._sentences(file_path): 
            lemmas = self.filter_lemmas(sentence.lemmas)
            tokens = [Token(t) for t in sentence.tokens]
            pos_tags = sentence.pos_tags
            head_ids = sentence.head_ids
            dep_rels = sentence.dep_rels

            if len(tokens) > self.maximum_length:
                continue
            if self.move_preposition_head:
                sentence.move_preposition_head()

            if not sentence.srl_frames and not self.allow_null_predicate:
                continue  
            yield sentence

    def _sentences(self, file_path: str) -> Iterable[Conll2009Sentence]:
        """
        """
        with open(file_path, "r", encoding='utf8') as open_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            conll_rows = []
            for line in open_file:
                line = line.strip()
                if not line and conll_rows:
                    yield self._conll_rows_to_sentence(conll_rows)
                    conll_rows = []
                    continue
                if line: conll_rows.append(line)
            if conll_rows:
                yield self._conll_rows_to_sentence(conll_rows)

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> Conll2009Sentence:
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
        
        for index, row in enumerate(conll_rows):
            columns = row.split()
             
            if len(columns) < 14:
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
            
            head_ids.append(int(columns[8]))
            predicted_head_ids.append(int(columns[9]))
            dep_rels.append(columns[10])
            predicted_dep_rels.append(columns[11])
            
            # optional
            predicate_indicators.append('Y' if columns[12] != self._DUMMY else None)
            if columns[13] != self._DUMMY:
                lemma_sense = re.match(self._RE_SENSE_ID, columns[13]).groups()
                plemma = lemma_sense[0] 
                psense = float(lemma_sense[1])
                predicates.append((index, plemma))
            else:
                plemma, psense = None, None
            predicate_lemmas.append(plemma)
            predicate_senses.append(psense)
            
            if not span_labels:
                span_labels = [[] for _ in columns[14:]]
            for column, item in zip(span_labels, columns[14:]):
                item = item if item != self._DUMMY else self._EMPTY_LABEL
                if self.valid_srl_labels:
                    item = item if item in self.valid_srl_labels else self._EMPTY_LABEL
                column.append(item)
        
        srl_frames = [(idx, predicate, labels) for (idx, predicate), labels 
                                 in zip(predicates, span_labels)]
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
