from typing import Any, Type, TypeVar, Set, Dict, List, Sequence, Iterable, Optional, Tuple
import itertools 
import inspect 
import logging
import re, json

from overrides import overrides
from collections import Counter

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.from_params import remove_optional
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from nlpmimic.data.fields import IndexField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("feature")
class FeatureDatasetReader(DatasetReader):

    def __init__(self,
                 predt_indexers: Dict[str, TokenIndexer] = None,
                 lemma_indexers: Dict[str, TokenIndexer] = None,
                 feate_indexers: Dict[str, TokenIndexer] = None,
                 max_len_feate: int = 16,
                 max_num_argument: int = 7,
                 lazy: bool = False) -> None:
        self._predt_indexers = predt_indexers or {'predts': SingleIdTokenIndexer(namespace='predts')}
        self._argmt_indexers = lemma_indexers or {'argmts': SingleIdTokenIndexer(namespace='argmts')} 
        self._feate_indexers = feate_indexers or {'feates': SingleIdTokenIndexer(namespace='feates')} 
        
        self.max_len_feate = max_len_feate
        self.max_num_argument = max_num_argument

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        cnt: int = 0
        xxx: int = 0
        xxl: int = 0
        for instance in self._sentences(file_path): 
            cnt += 1
            if len(instance['a_lemmas']) <= 0:
                xxx += 1
                continue
            #print(instance)
            #if cnt >= 1: break
            xxl += 1 
            inst = self.text_to_instance(instance)
            yield self.text_to_instance(instance)

        logger.info("{} sentences, {} instances, {} skipped instances".format(cnt, xxl, xxx))

    def text_to_instance(self, inst) -> Instance:
        p_fvec, p_name, p_lemma = inst['p_fvec'], inst['p_name'], inst['p_lemma']
        a_fvecs, a_roles, a_lemmas = inst['a_fvecs'], inst['a_roles'], inst['a_lemmas']        
        
        fields: Dict[str, Field] = {}
        narg = min(len(a_lemmas), self.max_num_argument)
        fields['predicate'] = TextField([Token(p_lemma)], token_indexers=self._predt_indexers)

        afield = TextField([Token(t) for t in a_lemmas[:narg]], token_indexers=self._argmt_indexers)
        fields['arguments'] = afield
        fields['srl_frames'] = SequenceLabelField(a_roles[:narg], afield, 'srl_tags')
        fields['feate_lens'] = SequenceLabelField([len(vec) for vec in a_fvecs[:narg]], afield, 'len_tags')
        
        features = []
        for vec in a_fvecs[:narg]:
            vec = vec[:]
            if len(vec) < self.max_len_feate:
                vec += ['O'] * (self.max_len_feate - len(vec))
            vec = map(str, vec)
            features += vec
        fields['feate_ids'] = TextField([Token(t) for t in features], token_indexers=self._feate_indexers)

        metadata = {'p_lemma': p_lemma, 'p_fvec': p_fvec,
                    'a_roles': a_roles, 'a_fvecs': a_fvecs, 'a_lemmas': a_lemmas} 
        fields['metadata'] = MetadataField(metadata)
        
        return Instance(fields)

    def _sentences(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding='utf8') as open_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in open_file:
                instance = json.loads(line) 
                yield instance 
