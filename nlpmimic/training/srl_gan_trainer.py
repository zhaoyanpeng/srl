
import logging
import os
import time
import re
import random
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Iterator, Any, NamedTuple

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import (dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names)
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


@TrainerBase.register("srl_gan")
class GanSrlTrainer(Trainer):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 optimizer_dis: torch.optim.Optimizer,
                 parameter_names: List[str],
                 dis_param_names: List[str],
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
        
        super(GanSrlTrainer, self).__init__(
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
        
        self.optimizer_dis = optimizer_dis
        self.parameter_names = parameter_names
        self.dis_param_names = dis_param_names

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

    def _sample(self, instances: Iterable[Instance], batch_size: int) -> Iterable[Instance]:
        instances = ensure_list(instances)
        random.seed(0)
        samples = random.sample(population=instances, k = batch_size) 
        return samples
    
    def _number_batches(self) -> int:
        if self.train_data is not None:
            return self.iterator.get_num_batches(self.train_data)
        elif self.train_dx_data is not None and \
                self.train_dy_data is not None:
            return self.iterator.get_num_batches(self.train_dx_data)
        else:
            raise ConfigurationError("Invalid training data setting")

    def _create_batches(self, num_epochs: int = 1) -> Iterator[TensorDict]:
        def _iterate(instances: Iterable[Instance]):
            yield from instances
        # the above code mimics the valid input to `lazy_groups_of` 
        if self.train_data is not None:
            generator = self.iterator(self.train_data, num_epochs, self.shuffle)
            return generator
        elif self.train_dx_data is not None and \
                self.train_dy_data is not None:
            for dx_batch in lazy_groups_of(_iterate(self.train_dx_data), self.iterator._batch_size):
                dy_batch = self._sample(self.train_dy_data, len(dx_batch))
                dy_dx_batch = dy_batch + dx_batch
                batch = Batch(dy_dx_batch)
                batch.index_instances(self.model.vocab)
                batch = batch.as_tensor_dict()
                yield batch
        else:
            raise ConfigurationError("Invalid training data setting")
    
    def batch_loss(self, 
                   batch: torch.Tensor, 
                   for_training: bool, 
                   retrive_generator_loss: bool,
                   reconstruction_loss: bool = True) -> torch.Tensor:
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
                                     retrive_generator_loss=retrive_generator_loss, 
                                     reconstruction_loss=reconstruction_loss)
        try:
            if for_training: # compulsory loss
                if retrive_generator_loss:
                    loss = output_dict["gen_loss"] # loss of the generator
                else:
                    loss = output_dict["dis_loss"] # loss of the discriminator
                loss += self.model.get_regularization_penalty()
            else:
                loss = None
            if reconstruction_loss: # can be added into compulsory loss in semi-supervised setting
                rec_loss = output_dict["rec_loss"]
            else:
                rec_loss = None
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " '*loss' key in the output of model.forward(inputs).")
            loss = rec_loss = None
        return loss, rec_loss
    
    def rescale_gradients(self, parameter_names: List[str]) -> Optional[float]:
        return mimic_training_util.rescale_gradients(self.model, parameter_names, self._grad_norm)
    
    def _gradient(self, optimizer: torch.optim.Optimizer, for_generator: bool, batch_numb_total: int) -> float:
        # this rescales gradients of all the parameters of the model, it would be better to update 
        # gradients of generator's and discriminator's parameters respectively. To do this, we need
        # to retain parameter names of the generator and discriminator.
        
        if for_generator:
            batch_grad_norm = self.rescale_gradients(self.parameter_names) 
        else:
            batch_grad_norm = self.rescale_gradients(self.dis_param_names) 
            
        # This does nothing if batch_num_total is None or you are using an
        # LRScheduler which doesn't update per batch.
        if self._learning_rate_scheduler:
            self._learning_rate_scheduler.step_batch(batch_num_total)
       
        if self._tensorboard.should_log_histograms_this_batch():
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            if for_generator:
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters() if name in self.parameter_names}
            else:
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters() if name in self.dis_param_names}
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
        
        cumulative_batch_size = 0
        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total
            
            print(batch)

            # update generator
            self.optimizer.zero_grad()
            gen_loss, _ = self.batch_loss(batch, 
                                      for_training=True, 
                                      retrive_generator_loss=True,
                                      reconstruction_loss=False)
            if torch.isnan(gen_loss):
                raise ValueError("nan loss encountered")
            gen_loss.backward()
            gen_batch_grad_norm = self._gradient(self.optimizer, True, batch_num_total)
           
           
            # update discriminator
            self.optimizer_dis.zero_grad()
            dis_loss, rec_loss = self.batch_loss(batch, 
                                             for_training=True, 
                                             retrive_generator_loss=False,
                                             reconstruction_loss=True)
            if torch.isnan(dis_loss):
                raise ValueError("nan loss encountered")
            dis_loss.backward()
            dis_batch_grad_norm = self._gradient(self.optimizer, False, batch_num_total)
 
            #cnt += 1
            #if cnt >= 1:
            #    import sys
            #    sys.exit(0)
            
            #train_loss += loss.item()
            train_loss += gen_loss.item()
            train_loss += dis_loss.item()

            reconstruction_loss += rec_loss.item() 

            # Update the description with the latest metrics
            metrics = mimic_training_util.get_metrics(self.model, 
                                                      train_loss, 
                                                      batches_this_epoch,
                                                      reconstruction_loss=reconstruction_loss)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, 
                                                                        gen_batch_grad_norm,
                                                                        self.parameter_names)
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, 
                                                                        dis_batch_grad_norm,
                                                                        self.dis_param_names)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)
                self._tensorboard.log_learning_rates(self.model, self.optimizer_dis)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.add_train_scalar("loss/rec_loss_train", metrics["rec_loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = training_util.get_batch_size(batch)
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size/batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

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

            _, loss = self.batch_loss(batch, 
                                      for_training=False,
                                      retrive_generator_loss=False, # dead; active only during training
                                      reconstruction_loss=True) # implying gold labels exist
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

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
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
        
        cnt = 0
        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            
            cnt += 1
            #if cnt >= 2:
            #    import sys
            #    sys.exit(0)
            
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

        return metrics

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
                    params: Params,
                    validation_iterator: DataIterator = None,
                    discriminator_param_name: str = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)
        
        parameters = []
        dis_params = []
        parameter_names = []
        dis_param_names = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if discriminator_param_name in n:
                    dis_param_names.append(n) 
                    dis_params.append([n, p])
                else:
                    parameter_names.append(n) 
                    parameters.append([n, p])

        logger.info("Following parameters belong to the discriminator (with gradient):")
        for x in dis_param_names:
            logger.info(x)
        logger.info("Following parameters belong to the generator (with gradient):")
        for x in parameter_names:
            logger.info(x)
        
        #import sys
        #sys.exit(0)
        
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if dis_params:
            optimizer_dis = Optimizer.from_params(dis_params, params.pop("optimizer_dis"))
        else:
            optimizer_dis = None

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
        return cls(model, optimizer, optimizer_dis, 
                   parameter_names, dis_param_names, 
                   iterator, train_data, train_dx_data, train_dy_data,
                   validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
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

