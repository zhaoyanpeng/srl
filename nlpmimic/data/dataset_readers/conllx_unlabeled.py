from typing import Callable, Iterator, Any, Type, TypeVar, Dict, List, Sequence, Iterable, Optional, Tuple
import itertools 
import inspect 
import logging
import re, sys

from overrides import overrides
from collections import Counter

from allennlp.common import Tqdm
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

class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk for generating instances and calls it for
    each call to ``__iter__``.
    """
    def __init__(self, instance_generator: Callable[[], Iterator[Instance]]) -> None:
        super().__init__()
        self.instance_generator = instance_generator

    def __iter__(self) -> Iterator[Instance]:
        instances = self.instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator")
        return instances

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
    _EMPTY_LEMMA = '@@UNKNOWN@@' # masked lemma
    _EMPTY_PREDICATE = '@@UNKNOWN@@'
    _WILD_NUMBER = 'NNN'
    _RE_SENSE_ID = '(^.*?)\.(\d+\.?\d*?)$'
    #_RE_IS_A_NUM = '^\d+(?:[,.]\d*)?$'
    _RE_IS_A_NUM = '^\d+(?:([,]|[.]|[:]|[-]|[\\]|[\/]|\\\/)\d*){0,5}$'
    _VALID_LABELS = {'dep', 'pos'}
    _DEFAULT_INSTANCE_TYPE = 'basic' # srl_gan
    _DEFAULT_APPENDIX_TYPE = 'nyt_infer' # nyt_learn
    _MAX_NUM_ARGUMENT = 6 

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lemma_indexers: Dict[str, TokenIndexer] = None,
                 lemma_file: str = None,
                 lemma_use_firstk: int = 5000, # used as frequency when it is smaller than 100
                 predicate_file: str = None,
                 predicate_use_firstk: int = 1500,
                 feature_labels: Sequence[str] = (),
                 maximum_length: float = float('inf'),
                 flatten_number: bool = False,
                 valid_srl_labels: Sequence[str] = (),
                 moved_preposition_head: List[str] = [],
                 allow_null_predicate: bool = False,
                 max_num_argument: int = 7,
                 min_valid_lemmas: float = None,
                 instance_type: str = _DEFAULT_INSTANCE_TYPE,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._MAX_NUM_ARGUMENT = max_num_argument

        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._lemma_indexers = lemma_indexers or {'lemmas': SingleIdTokenIndexer(namespace='lemmas')} 
        for label in feature_labels:
            if label not in self._VALID_LABELS: 
                raise ConfigurationError("unknown feature label type: {}".format(label))
        
        self.flatten_number = flatten_number
        self.maximum_length = maximum_length
        self.valid_srl_labels = valid_srl_labels
        self.min_valid_lemmas = min_valid_lemmas

        self.feature_labels = set(feature_labels)
        self.moved_preposition_head = moved_preposition_head
        self.allow_null_predicate = allow_null_predicate
        self.instance_type = instance_type

        try:
            self.lemma_set = None
            if lemma_file is not None:
                lemma_dict = Counter() 
                with open(lemma_file, 'r') as lemmas:
                    for line in lemmas:
                        if not line.strip():
                            continue
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
    
        try:
            self.predicate_set = None
            if predicate_file is not None:
                use_json = True if predicate_file.endswith('.json') else False
                predicate_dict = Counter()
                with open(predicate_file, 'r') as predicates:
                    for line in predicates:
                        if not line.strip():
                            continue
                        if use_json:
                            line = json.loads(line)
                        else:
                            line = line.strip().split()
                        k, v = line[0], int(line[1])
                        predicate_dict[k] += int(v)
                # construct vocab
                predicate_set = set()
                if predicate_use_firstk > 100:
                    for idx, (k, v) in enumerate(predicate_dict.most_common()):
                        if idx >= predicate_use_firstk: break
                        predicate_set.add(k)
                else:
                    for k, v in predicate_dict.most_common():
                        if v < predicate_use_firstk: continue
                        predicate_set.add(k)
                self.predicate_set = predicate_set
        except Exception as e:
            logger.info("Reading vocabulary of predicates failed: %s", predicate_file)
            self.predicate_set = None

    def filter_lemmas(self, lemmas: List[str], sentence: Conll2009Sentence) -> List[str]:
        gold_lemmas = list(set(lemmas))  
        gold_isnull = len(gold_lemmas) == 1 and gold_lemmas[0] == self._DUMMY
        
        if self.flatten_number: # replace numbers with ...
            pos_tags = list(set(sentence.pos_tags))  
            if len(pos_tags) == 1 and pos_tags[0] == self._DUMMY:
                pos_tags = sentence.predicted_pos_tags 
            else:
                pos_tags = sentence.pos_tags
            for idx, lemma in enumerate(lemmas):
                #if pos_tags[idx] == 'CD' and re.match(self._RE_IS_A_NUM, lemma):
                if re.match(self._RE_IS_A_NUM, lemma):
                    if len(sentence.tokens) > 0:
                        sentence.tokens[idx] = self._WILD_NUMBER 
                    if len(sentence.lemmas) > 0:
                        sentence.lemmas[idx] = self._WILD_NUMBER 
                    if len(sentence.predicted_lemmas) > 0:
                        sentence.predicted_lemmas[idx] = self._WILD_NUMBER 

        lemmas = sentence.lemmas if not gold_isnull else sentence.predicted_lemmas
        if self.lemma_set is not None and lemmas is not None:
            lemmas = [lemma if lemma in self.lemma_set else self._EMPTY_LEMMA for lemma in lemmas] 
        return lemmas

    def is_valid_lemmas(self, lemmas: List[str], labels: List[str]) -> bool:
        # in case there are not valid arguments, e.g., all are reset to 'M'
        n_arg, n_valid_arg = 0, 0
        for idx, label in enumerate(labels):
            if label != self._EMPTY_LABEL:
                n_arg += 1
                if lemmas[idx] != self._EMPTY_LEMMA:
                    n_valid_arg += 1
        ratio = n_valid_arg / n_arg 
        if ratio < self.min_valid_lemmas:
            return False
        else:
            return True
    
    def read(self, 
             context_path: str, 
             appendix_path: str, 
             appendix_type: str = _DEFAULT_APPENDIX_TYPE, 
             firstk: int = sys.maxsize) -> Iterable[Instance]:
        
        lazy = getattr(self, 'lazy', None)
        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if lazy:
            return _LazyInstances(lambda: iter(self._read(
                context_path, appendix_path, appendix_type, firstk)))
        else:
            instances = self._read(context_path, appendix_path, appendix_type, firstk)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))
            return instances

    @overrides
    def _read(self, 
              context_path: str, 
              appendix_path: str = None, 
              appendix_type: str = _DEFAULT_APPENDIX_TYPE, 
              firstk: int = sys.maxsize) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        context_path = cached_path(context_path)
        if appendix_path is not None:
            appendix_path = cached_path(appendix_path)
        cnt: int = 0
        xll: int = 0
        xxl: int = 0 
        llx: int = 0
        isample: int = 0
        for sentence in self._sentences(context_path, appendix_path=appendix_path, appendix_type=appendix_type): 
            lemmas = self.filter_lemmas(sentence.lemmas, sentence)
            tokens = [Token(t) for t in sentence.tokens]
            pos_tags = sentence.predicted_pos_tags
            
            head_ids = sentence.head_ids
            dep_rels = sentence.dep_rels
            cnt += 1
            #print('\n{}\n{}\n{}\n'.format(cnt, xxl, sentence.format()))
            if len(tokens) > self.maximum_length:
                continue
            if False and self.moved_preposition_head: # can't move without head ids
                sentence.move_preposition_head()

            if not sentence.srl_frames:    
                #print('\n{}\n{}\n{}\n'.format(cnt, xxl, sentence.format()))
                if not self.allow_null_predicate:
                    continue  

                labels = [self._EMPTY_LABEL for _ in tokens]
                predicate_indicators = [0 for _ in tokens]
                isample += 1
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
                if isample >= firstk:
                    break
            else:
                for (predicate_index, predicate, labels) in sentence.srl_frames:
                    srl_labels = list(set(labels))
                    if not self.allow_null_predicate and \
                        len(srl_labels) == 1 and srl_labels[0] == self._EMPTY_LABEL:
                        #print('\n{}\n{}\n{}\n'.format(cnt, xxl, sentence.format()))
                        continue

                    if self.predicate_set is not None and \
                        predicate not in self.predicate_set:
                        xll += 1
                        predicate = self._EMPTY_PREDICATE
                        pass
                        #continue

                    if self.min_valid_lemmas and not self.is_valid_lemmas(lemmas, labels):
                        llx += 1                 
                        continue

                    predicate_indicators = [0 for _ in labels]
                    predicate_indicators[predicate_index] = 1
                    predicate_sense = sentence.predicate_senses[predicate_index]
                    xxl += 1
                    isample += 1 
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
                    if isample >= firstk:
                        break
                if isample >= firstk:
                    break
        logger.info("{} sentences, {} instances, {} skipped instances, {} modified predicates".format(cnt, isample, llx, xll))

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
        if self.instance_type == "srl_graph":
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

            argument_indices = [i for i, label in enumerate(labels) if label != self._EMPTY_LABEL]
            num_argument = len(argument_indices)
            if num_argument >= self._MAX_NUM_ARGUMENT:
                argument_indices = argument_indices[:self._MAX_NUM_ARGUMENT]
                argument_mask = [1] * self._MAX_NUM_ARGUMENT
            else:
                non_argument_idx = 0
                #non_argument_idx = labels.index(self._EMPTY_LABEL)
                num_pad = self._MAX_NUM_ARGUMENT - num_argument
                argument_indices += [non_argument_idx] * num_pad 
                argument_mask = [1] * num_argument + [0] * num_pad 

            fields['argument_indices'] = IndexField(argument_indices, fields['srl_frames'])
            fields['predicate_index'] = IndexField([predicate_index], fields['predicates']) 
            fields['argument_mask'] = IndexField(argument_mask, fields['argument_indices'])

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
            
    def _sentences(self, context_path: str, appendix_path: str = None, 
        appendix_type: str = _DEFAULT_APPENDIX_TYPE) -> Iterable[Conll2009Sentence]:
        """ The appendix file could have been merged into the context file
        """
        if appendix_path is None:
            with open(context_path, "r", encoding='utf8') as context_file:
                logger.info("Reading contexts from lines in file at: %s", context_path)
                conll_rows = []
                for context in context_file:
                    line = context.strip()
                    if not line and conll_rows:
                        yield self._conll_rows_to_sentence(conll_rows, appendix_type)
                        conll_rows = []
                        continue
                    if line: conll_rows.append(line)
                if conll_rows:
                    yield self._conll_rows_to_sentence(conll_rows, appendix_type)
        else:
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
                    item = item if item != self._DUMMY else self._EMPTY_LABEL
                    if self.valid_srl_labels:
                        item = item if item in self.valid_srl_labels else self._EMPTY_LABEL
                    column.append(item)
            
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
                    item = item if item != self._DUMMY else self._EMPTY_LABEL
                    if self.valid_srl_labels:
                        item = item if item in self.valid_srl_labels else self._EMPTY_LABEL
                    column.append(item)
            
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

