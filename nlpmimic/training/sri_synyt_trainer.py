import os, time, re, random, datetime, traceback, logging
from typing import cast, Dict, Optional, List, Tuple, Union, Iterable, Iterator, Any, NamedTuple

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, check_for_data_path, check_for_gpu
from allennlp.common.util import (ensure_list, dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb)
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util as nn_util
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util

from nlpmimic.training import util as mimic_training_util
from nlpmimic.training.util import DataSampler, DataLazyLoader 
from nlpmimic.training.tensorboard_writer import GanSrlTensorboardWriter
from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TrainerBase.register("sri_synyt")
class VaeSrlTrainer():
    def __init__(self, 
                 train_dx, 
                 train_dy, 
                 train_dataset,
                 model = None,
                 devel_dataset = None,
                 test_dataset = None) -> None:

        self.train_dataset = train_dataset
        self.devel_dataset = devel_dataset
        self.test_dataset = test_dataset

        self.train_dx = train_dx
        self.train_dy = train_dy

        self.model = model

    def batch_loss(self) -> torch.Tensor:
        pass

    def train(self) -> Dict[str, Any]:
        for i in range(3):  
            cnt = 0
            for sent in self.train_dx:
                cnt += 1
                #print(sent)
                if cnt > 1:
                    print('\n\n\n')
                    break

    def load_model(self) -> bool:
        pass

    def archive(self) -> Dict[str, Any]:
        pass

    def _validation_loss(self) -> Tuple[float, int]:
        pass

    @classmethod
    def from_params(cls, # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False):
        reader_mode = params.pop("reader_mode", mimic_training_util.DEFAULT_READER_MODE)
        all_datasets = mimic_training_util.datasets_from_params(params, reader_mode)
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        params.pop("vocabulary", {})
        params.pop("add_unlabeled_noun", False)

        trainer_params = params.pop("trainer")

        train_dx_data = all_datasets['train_dx']
        train_dy_data = all_datasets['train_dy']

        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')
        train_data = None

        # Special logic to keep old from_params behavior.
        return cls(train_dx_data, train_dy_data, train_data, devel_dataset = validation_data, test_dataset = test_data)

