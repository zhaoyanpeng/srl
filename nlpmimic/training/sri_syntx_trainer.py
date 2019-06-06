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

@TrainerBase.register("sri_syntx")
class VaeSrlTrainer():
    def __init__(self, train_dataset,
                 model = None,
                 devel_dataset = None,
                 test_dataset = None) -> None:

        self.train_dataset = train_dataset
        self.devel_dataset = devel_dataset
        self.test_dataset = test_dataset

        self.model = model

    def batch_loss(self) -> torch.Tensor:
        pass

    def train(self) -> Dict[str, Any]:
        cnt = 0
        for sent in self.train_dataset:
            cnt += 1
            print(sent.format())
            if cnt > 1:
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
        trainer_params = params.pop("trainer")
        reading_list = ["train_data_path", "validation_data_path", "test_data_path"]
        for data_name in reading_list:
            data_path = params.get(data_name, None)
            if data_path is not None:
                check_for_data_path(data_path, data_name)
        
        train_data = validation_data = test_data = None
        dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))

        train_data_path = params.pop('train_data_path')
        logger.info("Reading training data from %s", train_data_path)
        train_data = dataset_reader._read(train_data_path)

        validation_data_path = params.pop('validation_data_path', None)
        if validation_data_path is not None:
            logger.info("Reading validation data from %s", validation_data_path)
            validation_data = dataset_reader._read(validation_data_path)

        test_data_path = params.pop("test_data_path", None)
        if test_data_path is not None:
            logger.info("Reading test data from %s", test_data_path)
            test_data = dataset_reader._read(test_data_path)

        # Special logic to keep old from_params behavior.
        return cls(train_data, devel_dataset = validation_data, test_dataset = test_data)

