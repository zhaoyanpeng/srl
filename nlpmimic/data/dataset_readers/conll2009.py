from typing import Any, Type, TypeVar, Set, Dict, List, Sequence, Iterable, Optional, Tuple
import itertools, copy 
import inspect 
import logging
import re, json

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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T = TypeVar('T')
class Conll2009Sentence:
    def __init__(self,
                 token_ids: List[int],
                 tokens: List[str],
                 lemmas: List[str],
                 predicted_lemmas: List[str],
                 pos_tags: List[str],
                 predicted_pos_tags: List[str],
                 features: List[Optional[str]],
                 predicted_features: List[Optional[str]],
                 head_ids: List[int],
                 predicted_head_ids: List[int],
                 dep_rels: List[str],
                 predicted_dep_rels: List[str],
                 predicate_indicators: List[Optional[str]],
                 predicate_lemmas: List[Optional[str]],
                 predicate_senses: List[Optional[float]],
                 srl_frames: List[Tuple[str, List[str]]]) -> None:

        self.token_ids = token_ids
        self.tokens = tokens
        self.lemmas = lemmas
        self.predicted_lemmas = predicted_lemmas
        self.pos_tags = pos_tags
        self.predicted_pos_tags = predicted_pos_tags
        self.features = features
        self.predicted_features = predicted_features
        self.head_ids = head_ids
        self.predicted_head_ids = predicted_head_ids
        self.dep_rels = dep_rels
        self.predicted_dep_rels = predicted_dep_rels
        self.predicate_indicators = predicate_indicators
        self.predicate_lemmas = predicate_lemmas
        self.predicate_senses = predicate_senses
        self.srl_frames = srl_frames

        self.all_fields_but_srl = [
                self.token_ids, self.tokens, 
                self.lemmas, self.predicted_lemmas,
                self.pos_tags, self.predicted_pos_tags,
                self.features, self.predicted_features,
                self.head_ids, self.predicted_head_ids,
                self.dep_rels, self.predicted_dep_rels,
                self.predicate_indicators, self.predicate_lemmas]
    
    def concat(self, list_a: List[str], list_b: List[str]):
        return [e1 + e2 for e1, e2 in zip(list_a, list_b)]

    def padding_with_space(self, 
                           items: List[Any], 
                           width: int = 0,
                           align_left: bool = True) -> List[str]: 
        return ['{: <{}}'.format(x, width) for x in items] if align_left \
            else ['{: >{}}'.format(x, width) for x in items]
    
    def padding_with_tab(self, 
                         items: List[Any], 
                         align_left: bool = True) -> List[str]:
        return ['{}\t'.format(x) for x in items] if align_left \
            else ['\t{}'.format(x) for x in items]
    
    def format(self, read_friendly: bool = True, space_width: int = 2) -> str:
        """ FIXME: Here assumming a valid Conll 2009 sentence, add sanity check.
        """
        all_fields_but_srl = copy.deepcopy(self.all_fields_but_srl)
        for idx, sign in enumerate(self.predicate_indicators):
            if not sign: continue
            all_fields_but_srl[-1][idx] = '{}.{:02.0f}'.format(
            self.predicate_lemmas[idx], self.predicate_senses[idx])
        # format string items
        columns = []
        srl_frames = [frame for _, _, frame in self.srl_frames]
        if not read_friendly:
            for idx, field in enumerate(all_fields_but_srl + srl_frames):
                field = ['_' if x is None else x for x in field] # remove None
                columns.append(self.padding_with_tab(field))
        else:
            for idx, field in enumerate(all_fields_but_srl + srl_frames):
                field = ['_' if x is None else x for x in field] # remove None
                width = max([0] + [len(str(x)) for x in field]) + space_width
                columns.append(self.padding_with_space(field, width, True))
        # concatenate string items
        rows = self.concat(columns[0], columns[1])
        for field in columns[2:]:
            rows = self.concat(rows, field) 
        return '\n'.join(rows) + '\n' 
    
    @classmethod 
    def instance(cls: Type[T], instance: Instance) -> Type[T]:
        """ Convert the Instance into the Conll 2009 sentence. Assume only one or zero predicate.
        """
        fields = instance.fields
        if 'tokens' not in fields:
            raise ValueError(f'expected key \'tokens\' for {self.__name__}') 

        kwargs: Dict[str, Any] = {}
        kwargs['tokens'] = [t.text for t in fields['tokens']]
        kwargs['lemmas'] = fields['metadata']['lemmas']
        kwargs['predicate_indicators'] = fields['predicate_indicators'].labels
        kwargs['srl_frames'] = []
        
        length = len(kwargs['lemmas'])
        kwargs['token_ids'] = list(range(1, length + 1))

        signature = inspect.signature(cls.__init__)
        for name, param in signature.parameters.items():
            if name == 'self' or name in kwargs:
                continue
             
            annotation = remove_optional(param.annotation)
            args = getattr(annotation, '__args__', [str])

            if name in fields['metadata']:
                kwargs[name] = fields['metadata'][name]
            elif args[0] == int or args[0] == float:
                kwargs[name] = [0 for _ in range(length)]
            else:
                kwargs[name] = ['_' for _ in range(length)]

        if fields['metadata']['predicate']:
            idx = kwargs['predicate_indicators'].index(1) 
            kwargs['predicate_lemmas'][idx] = fields['metadata']['predicate']
            kwargs['srl_frames'] = [(idx, 
                                    kwargs['lemmas'][idx], 
                                    ['_' if x == 'O' else x for x in fields['srl_frames'].labels])]
        predicate_indicators = ['_' if x == 0 else 'Y' for x in kwargs['predicate_indicators']] 
        kwargs['predicate_indicators'] = predicate_indicators 
        return cls(**kwargs)
    
    @classmethod
    def initialize(cls: Type[T],
                   predicate_index: int,
                   tokens: List[str],
                   labels: List[str],
                   lemmas: List[str] = None,
                   pos_tags: List[str] = None,
                   head_ids: List[int] = None,
                   predicate: Optional[str] = None) -> Type[T]:
        length = len(tokens)
        kwargs: Dict[str, Any] = {}
        kwargs['tokens'] = tokens
        kwargs['lemmas'] = lemmas or ['_'] * length
        kwargs['pos_tags'] = pos_tags or ['_'] * length
        kwargs['head_ids'] = head_ids or ['_'] * length
        kwargs['token_ids'] = list(range(1, length + 1))
        kwargs['predicate_indicators'] = ['_'] * length
        kwargs['predicate_indicators'][predicate_index] = 'Y'
        kwargs['srl_frames'] = [(predicate_index,
                                kwargs['lemmas'][predicate_index],
                                ['_' if x == 'O' else x for x in labels])]
        signature = inspect.signature(cls.__init__)
        for name, param in signature.parameters.items():
            if name == 'self' or name in kwargs:
                continue
             
            annotation = remove_optional(param.annotation)
            args = getattr(annotation, '__args__', [str])

            if args[0] == int or args[0] == float:
                kwargs[name] = [0 for _ in range(length)]
            else:
                kwargs[name] = ['_' for _ in range(length)]
        
        predicate = predicate if predicate is not None else kwargs['lemmas'][predicate_index]
        kwargs['predicate_lemmas'][predicate_index] = predicate 
        return cls(**kwargs)
    
    def move_preposition_head(self, empty_label: str = 'O', moved_label: str = 'IN') -> None:
        """ Hard-coding, moving the head of a prepositional phrase from the preposition 
            to the nearest noun following the preposition.
        """ 
        for _, _, frames in self.srl_frames:
            for idx, label in reversed(list(enumerate(frames))):
                if label != empty_label and self.pos_tags[idx] == moved_label:
                    if idx + 1 in self.head_ids[idx:]:
                        offset = self.head_ids[idx:].index(idx + 1)
                        head_idx = idx + offset 
                        frames[idx] = empty_label
                    else:
                        head_idx = idx
                else:
                    head_idx = idx
                frames[head_idx] = label 

    def restore_preposition_head(self, empty_label: str = 'O', preposition: str = 'IN') -> None:
        """ Restoring implies this Conll 2009 sentence has been moved before.
        """
        for _, _, frames in self.srl_frames:
            for idx, label in enumerate(frames):
                head_idx = self.head_ids[idx] - 1
                if label != empty_label and self.pos_tags[head_idx] == preposition:
                    frames[idx] = empty_label
                else:
                    head_idx = idx
                frames[head_idx] = label
    
    def restore_preposition_head_cycle(self, empty_label: str = 'O') -> None:
        """ Restoring implies this Conll 2009 sentence has been moved before.
            This only considers the case where the predicate is also an argument.
            In this case, we do not move back the argument.
        """
        for _, _, frames in self.srl_frames:
            idx_predicate = self.predicate_indicators.index('Y')
            for idx, label in enumerate(frames):
                head_idx = self.head_ids[idx] - 1
                same_idx = idx == idx_predicate

                if label != empty_label and self.pos_tags[head_idx] == 'IN' and not same_idx:
                    frames[idx] = empty_label
                else:
                    head_idx = idx
                frames[head_idx] = label

    def restore_preposition_head_general(self, empty_label: str = 'O', preposition: str = 'IN') -> None:
        """ Restoring implies this Conll 2009 sentence has been moved before.
            We traverse all possible `IN` ancestors of the predicate and do not
            allow its arguments to move back to any of the `IN` ancestors.
        """
        for _, _, frames in self.srl_frames:
            idx_predicate = self.predicate_indicators.index('Y')
            head_predicate = self.head_ids[idx_predicate]
            
            father_in_idxes = set()
            nearest_in_root_idx = -1 
            while head_predicate != 0:
                head_idx = head_predicate - 1
                if self.pos_tags[head_idx] == preposition:
                    nearest_in_root_idx = head_idx
                    father_in_idxes.add(head_idx) 
                head_predicate = self.head_ids[head_idx]

            for idx, label in enumerate(frames):
                head_idx = self.head_ids[idx] - 1
                
                if label != empty_label and \
                    self.pos_tags[head_idx] == preposition and \
                    head_idx not in father_in_idxes and \
                    head_idx <= idx:
                    frames[idx] = empty_label
                else:
                    head_idx = idx
                frames[head_idx] = label

    def restore_preposition_head_with_conditions(self, 
                                                 start_pos: int, 
                                                 constraints: Set[int], 
                                                 empty_label: str = 'O') -> None:
        """ Restoring implies this Conll 2009 sentence has been moved before.
            Run `move_preposition_head` and then `restore_preposition_head`.
            After that, we `diff` the original input and the restored file .
            We parse the outputs of `diff` and record lines that do not need 
            to be restored. This is stupid hard-coding. We still do not know
            how to restore if the gold inputs are not available.
        """
        for _, _, frames in self.srl_frames:
            start_line_num = start_pos + 1
            for idx, label in enumerate(frames):
                line_num = start_line_num + idx
                head_idx = self.head_ids[idx] - 1
                if label != empty_label and \
                    self.pos_tags[head_idx] == 'IN' and \
                    line_num not in constraints:
                    frames[idx] = empty_label
                else:
                    head_idx = idx
                frames[head_idx] = label
        return start_pos + len(self.token_ids) + 1 

@DatasetReader.register("conll2009")
class Conll2009DatasetReader(DatasetReader):
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
    _DEFAULT_INSTANCE_TYPE = 'basic'
    _MAX_NUM_ARGUMENT = 7 

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
                 moved_preposition_head: List[str] = ["IN"],
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
            self.predicate_set = None

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
                
        if self.lemma_set is not None:
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

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        cnt: int = 0
        xll: int = 0
        xxl: int = 0 
        isample: int = 0
        for sentence in self._sentences(file_path): 
            lemmas = self.filter_lemmas(sentence.lemmas, sentence)
            tokens = [Token(t) for t in sentence.tokens]
            pos_tags = sentence.pos_tags
            head_ids = sentence.head_ids
            dep_rels = sentence.dep_rels
            cnt += 1
            if len(tokens) > self.maximum_length:
                continue
            for head in self.moved_preposition_head:
                sentence.move_preposition_head(moved_label=head)

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
                                            dep_rels,
                                            self.instance_type)
            else:
                for (predicate_index, predicate, labels) in sentence.srl_frames:
                    srl_labels = list(set(labels))
                    if not self.allow_null_predicate and \
                        len(srl_labels) == 1 and srl_labels[0] == self._EMPTY_LABEL:
                        continue

                    if self.predicate_set is not None and \
                        predicate not in self.predicate_set:
                        #continue
                        xll += 1
                        predicate = self._EMPTY_PREDICATE

                    if self.min_valid_lemmas and not self.is_valid_lemmas(lemmas, labels):
                        xxl += 1                 
                        continue

                    isample += 1    
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
                                                dep_rels,
                                                self.instance_type) 
        logger.info("{} sentences, {} instances, {} skipped instances, {} modified predicates".format(cnt, isample, xxl, xll))

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
                         dep_rels: List[str] = None,
                         instance_type: str = _DEFAULT_INSTANCE_TYPE) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        if instance_type == "srl_graph":
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
        
        if instance_type == "srl_gan":
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
                #psense = lemma_sense[1]
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

