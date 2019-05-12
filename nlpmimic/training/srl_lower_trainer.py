
import logging
import os
import time
import re
import random
import datetime
import traceback
from typing import cast, Dict, Optional, List, Tuple, Union, Iterable, Iterator, Any, NamedTuple

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import (dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, add_noise_to_dict_values)
from allennlp.common.util import (ensure_list, lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.trainer import Trainer
from allennlp.training import util as training_util

from nlpmimic.training import util as mimic_training_util
from nlpmimic.training.tensorboard_writer import GanSrlTensorboardWriter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("srl_lower")
class VaeSrlTrainer(Trainer):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 parameter_names: List[str],
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 train_dx_dataset: Iterable[Instance],
                 train_dy_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 rec_in_training: bool =  False,
                 gen_skip_nepoch: int = -1, 
                 gen_pretraining: int = -1,
                 gen_loss_scalar: float = 1.,
                 kld_loss_scalar: float = 1.,
                 bpr_loss_scalar: float = 1.,
                 kld_update_rate: float = None,
                 kld_update_unit: int = None,
                 sort_by_length: bool = False,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None) -> None:
        
        super(VaeSrlTrainer, self).__init__(
                 model,
                 optimizer,
                 iterator,
                 train_dataset,
                 validation_dataset,
                 patience,
                 validation_metric,
                 validation_iterator,
                 shuffle,
                 num_epochs,
                 serialization_dir,
                 num_serialized_models_to_keep,
                 keep_serialized_model_every_num_seconds,
                 model_save_interval,
                 cuda_device,
                 grad_norm,
                 grad_clipping,
                 learning_rate_scheduler,
                 summary_interval,
                 histogram_interval,
                 should_log_parameter_statistics,
                 should_log_learning_rate,
                 log_batch_size_period)
        
        self.parameter_names = parameter_names
        self.gen_loss_scalar = gen_loss_scalar
        self.kld_loss_scalar = kld_loss_scalar
        self.bpr_loss_scalar = bpr_loss_scalar
        self.kld_update_rate = kld_update_rate
        self.kld_update_unit = kld_update_unit

        self.train_dx_data = train_dx_dataset 
        self.train_dy_data = train_dy_dataset
        
        self._tensorboard = GanSrlTensorboardWriter(
                get_batch_num_total=lambda: self._batch_num_total,
                serialization_dir=serialization_dir,
                summary_interval=summary_interval,
                histogram_interval=histogram_interval,
                should_log_parameter_statistics=should_log_parameter_statistics,
                should_log_learning_rate=should_log_learning_rate) 
        
        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self.model)
        
        self.rec_in_training = rec_in_training
        self.gen_skip_nepoch = gen_skip_nepoch
        self.gen_pretraining = gen_pretraining
        
        self.sort_by_length = sort_by_length
        
        self.data_y_pivot = 0 # used in sampling y (verb) data

    def batch_loss(self, 
                   batch: torch.Tensor, 
                   for_training: bool, 
                   reconstruction_loss: bool = True,
                   peep_prediction: bool = False) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            raise RuntimeError("Multiple-gpu training not supported.")
            #output_dict = training_util.data_parallel(batch, self.model, self._cuda_devices)
        else:
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch, 
                                     reconstruction_loss=reconstruction_loss)
        try:
            if for_training: # compulsory loss
                loss = rec_loss = kl_loss = bp_loss = None
                loss = output_dict['loss']
            else:
                loss = rec_loss = kl_loss = bp_loss = None
                
            if peep_prediction: #and not for_training:
                output_dict = self.model.decode(output_dict)
                tokens = output_dict['tokens'][:5]
                labels = output_dict['srl_tags'][:5]
                gold_srl = output_dict['gold_srl'][:5]
                predicates = output_dict["predicate"]
                pindexes = output_dict["predicate_index"]
                argidxes = output_dict["arg_idxes"]
                for token, label, glabel, p, pid, aid in zip(tokens, labels, gold_srl, predicates, pindexes, argidxes):
                    xx = ['({}|{}: {}_{})'.format(idx, t, g, l) for idx, (t, g, l) in enumerate(zip(token, glabel, label))]
                    print('{} {}.{} {}\n'.format(xx, p, pid, aid))
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " '*loss' key in the output of model.forward(inputs).")
            loss = rec_loss = kl_loss = bp_loss = None
        return loss, rec_loss, kl_loss, bp_loss 
    
    def rescale_gradients(self, parameter_names: List[str]) -> Optional[float]:
        return mimic_training_util.rescale_gradients(self.model, parameter_names, self._grad_norm)
    
    def _gradient(self, optimizer: torch.optim.Optimizer, batch_num_total: int) -> float:
        batch_grad_norm = self.rescale_gradients(self.parameter_names) 
            
        # This does nothing if batch_num_total is None or you are using an
        # LRScheduler which doesn't update per batch.
        if self._learning_rate_scheduler:
            self._learning_rate_scheduler.step_batch(batch_num_total)
       
        if self._tensorboard.should_log_histograms_this_batch():
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            param_updates = {name: param.detach().cpu().clone()
                             for name, param in self.model.named_parameters() if name in self.parameter_names}
            optimizer.step()
                
            for name, param in self.model.named_parameters():
                if name not in param_updates:
                    continue
                param_updates[name].sub_(param.detach().cpu())
                update_norm = torch.norm(param_updates[name].view(-1, ))
                param_norm = torch.norm(param.view(-1, )).cpu()
                self._tensorboard.add_train_scalar("gradient_update/" + name, update_norm / (param_norm + 1e-7))
        else:
            optimizer.step()

        return batch_grad_norm 

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        reconstruction_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()
        
        # Get tqdm for the training batches
        if self.train_data is not None:
            train_generator = self.iterator(self.train_data, 1, self.shuffle)
            num_training_batches = self.iterator.get_num_batches(self.train_data)
        else:
            train_generator = self._create_batches()
            num_training_batches = self._number_batches()
        
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        bsize, cnt = 15, 0
        gen_nbatch, dis_nbatch = 0, 0
        optimize_gen, optimize_dis = True, True

        cumulative_batch_size = 0
        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total
            
            #if batches_this_epoch > 1:
            #    break
            #print(batch) 
            #print()
            #for inst in batch['metadata']:
            #    print(inst['predicate'])
            #print()
            
            # update generator
            self.optimizer.zero_grad()
            gen_loss, rec_loss, kl_loss, bp_loss = self.batch_loss(
                                    batch, 
                                    for_training=True, 
                                    reconstruction_loss=False)
            if torch.isnan(gen_loss):
                raise ValueError("nan loss encountered")
            
            g_loss = gen_loss.item()
            gen_loss *= self.gen_loss_scalar 
                
            #gen_loss.backward()
            #gen_batch_grad_norm = self._gradient(self.optimizer, batch_num_total)
            train_loss += gen_loss.item()
            
            if rec_loss is not None:
                reconstruction_loss += rec_loss.item() 
            else:
                reconstruction_loss = None 

            # Update the description with the latest metrics
            metrics = mimic_training_util.get_metrics(self.model, 
                                                      train_loss, 
                                                      batches_this_epoch,
                                                      kl_loss = kl_loss,
                                                      bp_loss = bp_loss,
                                                      reconstruction_loss = reconstruction_loss)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        return metrics

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=False)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch in val_generator_tqdm:

            _, loss, _, _, = self.batch_loss(batch, 
                                      for_training=False,
                                      reconstruction_loss=True, # implying gold labels exist
                                      peep_prediction=True)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def train(self, boost: bool = False) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            boost_signal = False
            if boost:
                boost_signal = self.load_model() 
                epoch_counter = 0
            if not boost_signal: 
                patience = self._metric_tracker._patience
                epoch_counter = self._restore_checkpoint()
                logger.info('Patience {} -- {}'.format(patience, self._metric_tracker._patience))
                if self._metric_tracker._patience < patience:
                    self._metric_tracker._patience = patience 
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()
        
        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_"+key] = max(metrics.get("peak_"+key, 0), value)

            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self._tensorboard.log_metrics(train_metrics, val_metrics=val_metrics)
            
            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            if self._learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.model.close()

        return metrics

    def load_model(self) -> bool:
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            logger.info("Loading the best model as the initial point.")
            self.model.load_state_dict(best_model_state)
            return True
        else:
            logger.info("Could not load the best model. "
                        "Train the model from the latest checkpoint.")
            return False

    def archive(self) -> Dict[str, Any]:
        """ 
        Allennlp crashes when approaching to the phase of archiving the best model. 
        So I have to write an archive function to load and archive the best model.
        """
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        else:
            raise ConfigurationError("Could not load the best model. "
                                     "Could you please check your settings?")
        # let us recover what happened before the Allennlp crashed.
        if self._validation_data is not None:
            with torch.no_grad():
                # We have a validation set, so compute all the metrics on it.
                val_loss, num_batches = self._validation_loss()
                val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                # Check validation metric for early stopping
                this_epoch_val_metric = val_metrics[self._validation_metric]
                self._metric_tracker.add_metric(this_epoch_val_metric)
        return {}

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False):

        # Special logic to keep old from_params behavior.
        from nlpmimic.training.trainer import TrainerPieces

        pieces = TrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        return cls.instantiate(model=pieces.model,
                               serialization_dir = serialization_dir,
                               iterator = pieces.iterator,
                               train_data = pieces.train_dataset,
                               train_dx_data = pieces.train_dx_dataset,
                               train_dy_data = pieces.train_dy_dataset,
                               validation_data = pieces.validation_dataset,
                               rec_in_training = pieces.rec_in_training, 
                               params = pieces.params,
                               validation_iterator = pieces.validation_iterator,
                               discriminator_param_name=pieces.dis_param_name)

    
    # Requires custom from_params.
    @classmethod
    def instantiate(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    train_dx_data: Iterable[Instance],
                    train_dy_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    rec_in_training: bool,
                    params: Params,
                    validation_iterator: DataIterator = None,
                    discriminator_param_name: List[str] = []) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        
        # customized settings for training
        gen_skip_nepoch = params.pop("gen_skip_nepoch", -1)
        gen_pretraining = params.pop("gen_pretraining", -1)
        gen_loss_scalar = params.pop("gen_loss_scalar", 1.)
        kld_loss_scalar = params.pop("kld_loss_scalar", 1.)
        bpr_loss_scalar = params.pop("bpr_loss_scalar", 1.)
        kld_update_rate = params.pop("kld_update_rate", None)
        kld_update_unit = params.pop("kld_update_unit", None)

        # process input data
        sort_by_length = params.pop("sort_by_length", False)
        
        # parameters for wgan
        clip_val = params.pop("clip_val", 0.01)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)
        
        parameters = []
        parameter_names = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                parameter_names.append(n) 
                parameters.append([n, p])

        logger.info("Following parameters are being optimized (with gradient):")
        for x in parameters:
            logger.info('{} is leaf: {}'.format(x[0], x[1].is_leaf))
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, 
                   parameter_names, 
                   iterator, train_data, train_dx_data, train_dy_data,
                   validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   rec_in_training = rec_in_training,
                   gen_skip_nepoch = gen_skip_nepoch,
                   gen_pretraining = gen_pretraining,
                   gen_loss_scalar = gen_loss_scalar,
                   kld_loss_scalar = kld_loss_scalar,
                   bpr_loss_scalar = bpr_loss_scalar,
                   kld_update_rate = kld_update_rate,
                   kld_update_unit = kld_update_unit,
                   sort_by_length = sort_by_length,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period)

