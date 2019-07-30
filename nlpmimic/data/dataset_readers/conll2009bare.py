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
from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class SrlInstance:
    
    def __init__(self, ptoken, plemma, pfeature):
        self.ptoken = ptoken
        self.plemma = plemma
        self.pfeature = pfeature

        self.arg_labels = []
        self.arg_lemmas = []
        self.arg_features = []

    def add_argument(self, arg_label, arg_lemma, arg_feature):
        self.arg_labels.append(arg_label)
        self.arg_lemmas.append(arg_lemma)
        self.arg_features.append(arg_feature)

# feature definitions

def pword(srl, aid=None):
    return srl.tokens[srl.predicate_idx]

def plemma(srl, aid=None):
    return srl.predicate

def pdep(srl, aid=None):
    return srl.dep_rels[srl.predicate_idx]

def ppos(srl, aid=None):
    return srl.pos_tags[srl.predicate_idx]

def pparent_word(srl, aid=None):
    pid = srl.head_ids[srl.predicate_idx]
    return 'ROOT_TOKEN' if pid == 0 else srl.tokens[pid - 1]

def pparent_pos(srl, aid=None):
    pid = srl.head_ids[srl.predicate_idx]
    return 'ROOT_POS' if pid == 0 else srl.pos_tags[pid - 1] 

def pchild_dep_set(srl, aid=None):
    child_descs = []
    for cid in srl.children_ids[srl.predicate_idx]:
        child_descs.append(srl.dep_rels[cid - 1])
    child_descs.sort()
    return '[' + ','.join(child_descs) + ']'

def pchild_pos_set(srl, aid=None):
    child_descs = []
    for cid in srl.children_ids[srl.predicate_idx]:
        child_descs.append(srl.pos_tags[cid - 1])
    child_descs.sort()
    return '[' + ','.join(child_descs) + ']'

def pchild_token_dep_set(srl, aid=None):
    child_descs = []
    for cid in srl.children_ids[srl.predicate_idx]:
        child_descs.append(srl.tokens[cid - 1] + ':' + srl.dep_rels[cid - 1])
    child_descs.sort()
    return '[' + ','.join(child_descs) + ']'

def pchild_pos_dep_set(srl, aid=None):
    child_descs = []
    for cid in srl.children_ids[srl.predicate_idx]:
        child_descs.append(srl.pos_tags[cid - 1] + ':' + srl.dep_rels[cid - 1])
    child_descs.sort()
    return '[' + ','.join(child_descs) + ']'

def pvoice(srl, aid=None):
    pid = srl.predicate_idx
    if not srl.pos_tags[pid].startswith("VB"):
        return None
    for cid in srl.children_ids[pid]:
        if srl.dep_rels[cid - 1] == "LGS":
            return "PASSIVE"
    if srl.tokens[pid].endswith("ing"):
        return "ACTIVE"
    if srl.dep_rels[pid] == "VC" and \
        srl.head_ids[pid] != 0 and srl.lemmas[srl.head_ids[pid] - 1] == "be":
            return "PASSIVE"
    else:
        return "ACTIVE"

def position(srl, aid):
    pid = srl.predicate_idx
    if pid < aid:
        return "RIGHT"
    elif pid > aid:
        return "LEFT"
    else:
        return "SAME"

def aword(srl, aid):
    return srl.tokens[aid]

def adep(srl, aid):
    return srl.dep_rels[aid]

def apos(srl, aid):
    return srl.pos_tags[aid]

def get_leftmost_cwid(srl, aid):
    if len(srl.children_ids[aid]) == 0 or srl.children_ids[aid][0] - 1 > aid:
        return None
    return srl.children_ids[aid][0] - 1

def get_rightmost_cwid(srl, aid):
    if len(srl.children_ids[aid]) == 0 or srl.children_ids[aid][-1] - 1 < aid:
        return None
    return srl.children_ids[aid][-1] - 1

def leftmost_cword(srl, aid):
    wid = get_leftmost_wid(srl, aid)
    return None if wid is None else srl.tokens[wid]

def rightmost_cword(srl, aid):
    wid = get_rightmost_wid(srl, aid)
    return None if wid is None else srl.tokens[wid]

def leftmost_cpos(srl, aid):
    wid = get_leftmost_wid(srl, aid)
    return None if wid is None else srl.pos_tags[wid]

def rightmost_cpos(srl, aid):
    wid = get_rightmost_wid(srl, aid)
    return None if wid is None else srl.pos_tags[wid]

def get_left_sibling_wid(srl, aid):
    siblings = srl.children_ids[srl.head_ids[aid] - 1] 
    left_sibling = None
    for sibling in siblings:
        if sibling - 1 == aid:
            break
        left_sibling = sibling - 1
    return left_sibling

def get_right_sibling_wid(srl, aid):
    siblings = srl.children_ids[srl.head_ids[aid] - 1]
    right_sibling = None
    for sibling in reversed(siblings):
        if sibling - 1 == aid:
            break
        right_sibling = sibling - 1
    return right_sibling

def left_sibling_word(srl, aid):
    wid = get_left_sibling_wid(srl, aid)
    return None if wid is None else srl.tokens[wid]

def right_sibling_word(srl, aid):
    wid = get_right_sibling_wid(srl, aid)
    return None if wid is None else srl.tokens[wid]

def left_sibling_pos(srl, aid):
    wid = get_left_sibling_wid(srl, aid)
    return None if wid is None else srl.pos_tags[wid]

def right_sibling_pos(srl, aid):
    wid = get_right_sibling_wid(srl, aid)
    return None if wid is None else srl.pos_tags[wid]

def get_path2root(srl, aid, include_root = False):
    path = []
    pid = srl.head_ids[aid]
    while pid != 0: # root
        path.append(pid - 1)
        pid = srl.head_ids[pid - 1]
    if include_root:
        path.append(-1)
    return path 

def get_path_a2b(srl, aid, bid):
    apath = get_path2root(srl, aid, include_root = True)
    bpath = get_path2root(srl, bid, include_root = True)

    common_idx = 0
    ia, ib = len(apath) - 1, len(bpath) - 1
    while apath[ia] == bpath[ib]:
        common_idx += 1 # longest common substr
        if ia == 0 or ib == 0:
            break
        ia -= 1
        ib -= 1
    
    assert common_idx > 0

    upath = apath[0 : -common_idx]    # upward path
    dpath = bpath[-common_idx - 1: : -1] # downward path 
    common = apath[-common_idx]
    return upath, common, dpath

def tok2str(srl, aid, include_pos, include_token, include_lemma):
    fields = []
    if include_pos:
        pos = srl.pos_tags[aid] if aid >= 0 else "ROOT_POS" 
        fields.append(pos)
    if include_token:
        token = srl.tokens[aid] if aid >= 0 else "ROOT_TOKEN"
        fields.append(token)
    if include_lemma:
        lemma = srl.lemmas[aid] if aid >= 0 else "ROOT_LEMMA" 
        fields.append(lemma)
    if len(fields) > 0:
        return '[{}]'.format('_'.join(map(str, fields)))
    else:
        return ''

def str_path(srl, aid, max_len=1000, include_pos=True, 
    include_dep=True, include_token=False, include_lemma=False):
    pid = srl.predicate_idx
    if aid == pid:
        return ""
    upath, common, dpath = get_path_a2b(pid, aid)

    str_tok = lambda aid: tok2str(srl, aid, include_pos, include_token, include_lemma) 
    str_rel = lambda cid: srl.dep_rels[cid] if include_dep else "" 

    s, l = '', 0 
    if upath:
        l += 1
        s += "-" + str_rel(pid) + "->"
        for idx in upath:
            if l == max_len:
                return s + "..."
            s += str_tok(idx) 
            s += "-" + str_rel(idx) + "->"
            l += 1
    
    if common != pid and common != aid:
        if l == max_len:
            return s + "..."
        s += str_tok(common) 
        
    if dpath:
        for idx in dpath:
            s += "<-" + str_rel(idx) + "-"
            l += 1
            if l == max_len:
                return s + "..."
            s += str_tok(idx)
        s += "<-" + str_rel(aid) + "-" 
    return s

def rel_path_p2a(srl, aid):
    return str_path(srl, aid, include_pos=False)  


class SrlFeature:

    @classmethod 
    def JohanssonPfeatures():
        features = [pchild_dep_set, plemma, pvoice, position, 
                    aword, apos, rightmost_cword, rightmost_cpos, ppos, rel_path_p2a, pdep]  
        return features

    @classmethod 
    def JohanssonAfeatures():
        features = [pchild_dep_set, plemma, pvoice, position, 
                    aword, apos, leftmost_cword, leftmost_cpos, rightmost_cword, rightmost_cpos,
                    left_sibling_word, left_sibling_pos, ppos, rel_path_p2a, pdep, adep]   
        return features 

    @classmethod 
    def pfeature():
        pass


class SrlStructure:
    
    def __init__(self, 
                tokens, 
                lemmas, 
                labels,
                predicate_indicators, 
                predicate, 
                predicate_index,
                predicate_sense,
                pos_tags,
                head_ids,
                dep_rels,
                dumy=None): 
        self.tokens = tokens 
        self.lemmas = lemmas 
        self.labels = labels
        self.predicate_indicators = predicate_indicators 
        self.predicate = predicate
        self.predicate_index = predicate_index
        self.predicate_sense = predicate_sense
        self.pos_tags = pos_tags
        self.head_ids = head_ids
        self.dep_rels = dep_rels 
        
        self.children_ids = [[] for _ in self.dep_rels]
        for idx, dep in enumerate(self.head_ids):
            if dep == 0: continue
            self.children_ids[dep - 1].append(idx + 1)
    
    def __repr__(self):
        return {
            "tokens" : self.tokens, 
            "lemmas" : self.lemmas,  
            "labels" : self.labels, 
            "predicate_indicators" : self.predicate_indicators,  
            "predicate" : self.predicate, 
            "predicate_index" : self.predicate_index, 
            "predicate_sense" : self.predicate_sense, 
            "pos_tags" : self.pos_tags, 
            "head_ids" : self.head_ids, 
            "dep_rels" : self.dep_rels,
            "children_ids" : self.children_ids}

    def __str__(self):
        ctx = self.__repr__()
        s = '\n'
        for k, v in ctx.items():
            ss = '{}: {}\n'.format(k, v) 
            s += ss 
        return s

@DatasetReader.register("conll2009_bare")
class Conll2009BareDatasetReader(Conll2009DatasetReader):

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
                 instance_type: str = Conll2009DatasetReader._DEFAULT_INSTANCE_TYPE,
                 lazy: bool = False) -> None:
        super(Conll2009BareDatasetReader, self).__init__(
                 token_indexers = token_indexers,
                 lemma_indexers = lemma_indexers,
                 lemma_file = lemma_file,
                 lemma_use_firstk = lemma_use_firstk, 
                 predicate_file = predicate_file,
                 predicate_use_firstk = predicate_use_firstk,
                 feature_labels = feature_labels,
                 maximum_length = maximum_length,
                 flatten_number = flatten_number,
                 valid_srl_labels = valid_srl_labels,
                 moved_preposition_head = moved_preposition_head,
                 allow_null_predicate = allow_null_predicate,
                 max_num_argument = max_num_argument,
                 min_valid_lemmas = min_valid_lemmas,
                 instance_type = instance_type,
                 lazy = lazy)

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


    def read_sentences(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        cnt: int = 0
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

                yield SrlStructure(tokens, 
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
                    if self.predicate_set is not None and \
                        predicate not in self.predicate_set:
                        xxl += 1
                        continue

                    srl_labels = list(set(labels))
                    if not self.allow_null_predicate and \
                        len(srl_labels) == 1 and srl_labels[0] == self._EMPTY_LABEL:
                        continue

                    if self.min_valid_lemmas and not self.is_valid_lemmas(lemmas, labels):
                        xxl += 1                 
                        continue

                    isample += 1    
                    predicate_indicators = [0 for _ in labels]
                    predicate_indicators[predicate_index] = 1
                    predicate_sense = sentence.predicate_senses[predicate_index]
                    yield SrlStructure(tokens, 
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
        logger.info("{} sentences, {} instances, {} skipped instances".format(cnt, isample, xxl))
