import os, time, re, random, datetime, traceback, logging
from typing import cast, Dict, Optional, List, Tuple, Union, Iterable, Iterator, Any, NamedTuple

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import (dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb)
from allennlp.common.tqdm import Tqdm
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
from nlpmimic.training.util import DataSampler, DataLazyLoader 
from nlpmimic.training.tensorboard_writer import GanSrlTensorboardWriter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TrainerBase.register("sri_upper")
class VaeSrlTrainer(Trainer):
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
                 rec_in_training: bool =  False,
                 dis_min_loss: float = -1.,
                 dis_skip_nepoch: int = -1,
                 gen_skip_nepoch: int = -1, 
                 gen_pretraining: int = -1,
                 dis_loss_scalar: float = 1.,
                 gen_loss_scalar: float = 1.,
                 kld_loss_scalar: float = 1.,
                 bpr_loss_scalar: float = 1.,
                 kld_update_rate: float = None,
                 kld_update_unit: int = None,
                 feature_matching: bool = False,
                 use_wgan: bool = False,
                 clip_val: float = 5.0,
                 sort_by_length: bool = False,
                 consecutive_update: bool = False,
                 dis_max_nbatch: int = 0,
                 gen_max_nbatch: int = 0,
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
        
        self.optimizer_dis = optimizer_dis
        self.parameter_names = parameter_names
        self.dis_param_names = dis_param_names
        self.gen_loss_scalar = gen_loss_scalar
        self.dis_loss_scalar = dis_loss_scalar
        self.kld_loss_scalar = kld_loss_scalar
        self.bpr_loss_scalar = bpr_loss_scalar
        self.kld_update_rate = kld_update_rate
        self.kld_update_unit = kld_update_unit
        # wgan
        self.feature_matching = feature_matching
        self.use_wgan = use_wgan 
        self.clip_val = clip_val

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
        self.dis_min_loss = dis_min_loss
        self.dis_skip_nepoch = dis_skip_nepoch
        self.gen_skip_nepoch = gen_skip_nepoch
        self.gen_pretraining = gen_pretraining
        
        self.sort_by_length = sort_by_length
        
        self.consecutive_update = consecutive_update
        self.dis_max_nbatch = dis_max_nbatch
        self.gen_max_nbatch = gen_max_nbatch

        if self.train_data is not None:
            self.data_sampler = DataLazyLoader(self.train_data, self.iterator, self.sort_by_length)

        if self.train_dx_data is not None:
            self.noun_sampler = DataLazyLoader(self.train_dx_data, self.iterator, self.sort_by_length)
        if self.train_dy_data is not None:
            self.verb_sampler = DataSampler(self.train_dy_data, self.iterator, self.sort_by_length) 

    def batch_loss(self, batch, 
                   training: bool = False, 
                   optimizing_generator: bool = True,
                   relying_on_generator: bool = True, 
                   caching_feature_only: bool = False,
                   retrive_crossentropy: bool = False,
                   supervisely_training: bool = False,
                   peep_prediction: bool = False) -> torch.Tensor:
        if self._multiple_gpu:
            raise RuntimeError("Multiple-gpu training not supported.")
            #output_dict = training_util.data_parallel(batch, self.model, self._cuda_devices)
        else:
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            statistics = mimic_training_util.decode_metrics(
                                    self.model, batch, training, 
                                    optimizing_generator = optimizing_generator,
                                    relying_on_generator = relying_on_generator,
                                    caching_feature_only = caching_feature_only,
                                    retrive_crossentropy = retrive_crossentropy,
                                    supervisely_training = supervisely_training,
                                    peep_prediction = peep_prediction)
            return statistics 

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()
        
        # Get tqdm for the training batches
        if self.train_data is not None:
            train_generator = self.data_sampler.sample()
            num_training_batches = self.data_sampler.nbatch() 
        else:
            train_generator = self.noun_sampler.sample()
            num_training_batches = self.noun_sampler.nbatch() 
        
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0
        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        cumulative_batch_size = 0
        gen_nbatch, dis_nbatch = 0, 0
        optimize_gen, optimize_dis = True, True
        if self.kld_update_rate is not None and self.kld_update_unit is not None and \
            epoch != 0 and epoch % self.kld_update_unit == 0 and self.kld_loss_scalar < 1:
            self.kld_loss_scalar += self.kld_update_rate 
            print('\n----------------------epoch {}: self.kld_loss_scalar is {}'.format(epoch, self.kld_loss_scalar))

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)
        for noun_batch, batch_size in train_generator_tqdm:
            peep = False
            if batches_this_epoch % 50 == 0:
                peep = True 
                #print('\n----------------------0. model.tau is {}'.format(self.model.tau.item()))
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total
            
            #verb_batch = self.verb_sampler.sample(batch_size) 


            self.optimizer.zero_grad()

            gen_loss, ce_loss, kl_loss, _, _ = self.batch_loss(
                                    noun_batch, training = True, 
                                    optimizing_generator = True,
                                    relying_on_generator = True, 
                                    caching_feature_only = False,
                                    retrive_crossentropy = True,
                                    supervisely_training = True, # nil
                                    peep_prediction=peep)

            if self.gen_pretraining != 0 and torch.isnan(gen_loss):
                raise ValueError("nan loss encountered")

            if ce_loss is not None:
                ce_loss = ce_loss.item()
            if kl_loss is not None:
                gen_loss += self.kld_loss_scalar * kl_loss
                kl_loss = kl_loss.item()

            gen_loss.backward()

            train_loss += gen_loss.item()
            gen_batch_grad_norm = self.gradient(self.optimizer, True, batch_num_total)


            # Update the description with the latest metrics
            metrics = mimic_training_util.get_metrics(self.model, 
                                                      train_loss, 
                                                      batches_this_epoch,
                                                      ce_loss = ce_loss,
                                                      kl_loss = kl_loss)

            description = training_util.description_from_metrics(metrics)
            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, 
                                                                        gen_batch_grad_norm,
                                                                        self.parameter_names,
                                                                        model_signature = 'gen_')
                """
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, 
                                                                        dis_batch_grad_norm,
                                                                        self.dis_param_names,
                                                                        model_signature = 'dis_')
                """
                self._tensorboard.log_learning_rates(self.model, self.optimizer)
                # self._tensorboard.log_learning_rates(self.model, self.optimizer_dis)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                if "ce_loss" in metrics:
                    self._tensorboard.add_train_scalar("loss/ce_loss_train", metrics["ce_loss"])
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
                    time.time() - last_save_time > self._model_save_interval):
                last_save_time = time.time()
                self._save_checkpoint('{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time))))

        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        return metrics

    def train(self, boost: bool = False) -> Dict[str, Any]:
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
                            "a different serialization directory or delete the existing serialization directory?")
        training_util.enable_gradient_clipping(self.model, self._grad_clipping)
        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        training_start_time = time.time()
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        
        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()

            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0), train_metrics['cpu_memory_MB'])
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
                # Update all the best_ metrics. (Otherwise they just stay the same as they were.)
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

    def _validation_loss(self) -> Tuple[float, int]:
        logger.info("Validating")
        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=False)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)

        self.model.eval()
        val_loss, batches_this_epoch = 0, 0
        for batch in val_generator_tqdm:

            _, loss, _, _, _, = self.batch_loss(batch, training = False,
                                                retrive_crossentropy = True, # implying gold labels exist
                                                peep_prediction = True)
            if loss is not None:
                batches_this_epoch += 1 # only count batches which contain a valid loss
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)
        return val_loss, batches_this_epoch

    def rescale_gradients(self, parameter_names: List[str]) -> Optional[float]:
        return mimic_training_util.rescale_gradients(self.model, parameter_names, self._grad_norm)

    def gradient(self, optimizer: torch.optim.Optimizer, for_generator: bool, batch_num_total: int) -> float:
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
            # get the magnitude of parameter updates for logging. We need a copy of current parameters to 
            # compute magnitude of updates, and copy them to CPU so large models won't go OOM on the GPU.
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
        
        # wgan: clip discriminator's parameter values to a small range
        if self.use_wgan and not for_generator: 
            mimic_training_util.clip_parameters(self.model, self.dis_param_names, self.clip_val)
        return batch_grad_norm 

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
        dis_skip_nepoch = params.pop("dis_skip_nepoch", -1)
        gen_skip_nepoch = params.pop("gen_skip_nepoch", -1)
        gen_pretraining = params.pop("gen_pretraining", -1)
        dis_loss_scalar = params.pop("dis_loss_scalar", 1.)
        gen_loss_scalar = params.pop("gen_loss_scalar", 1.)
        kld_loss_scalar = params.pop("kld_loss_scalar", 1.)
        bpr_loss_scalar = params.pop("bpr_loss_scalar", 1.)
        kld_update_rate = params.pop("kld_update_rate", None)
        kld_update_unit = params.pop("kld_update_unit", None)
        dis_min_loss = params.pop("dis_min_loss", -1.)

        consecutive_update = params.pop("consecutive_update", False)
        dis_max_nbatch = params.pop("dis_max_nbatch", 1)
        gen_max_nbatch = params.pop("gen_max_nbatch", 1)

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
        dis_params = []
        parameter_names = []
        dis_param_names = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                found = False
                for key in discriminator_param_name:
                    if key in n:
                        found = True
                        break
                if found:
                    dis_param_names.append(n) 
                    dis_params.append([n, p])
                else:
                    parameter_names.append(n) 
                    parameters.append([n, p])

        logger.info("Following parameters belong to the discriminator (with gradient):")
        for x in dis_params:
            logger.info('{} is leaf: {}'.format(x[0], x[1].is_leaf))
        logger.info("Following parameters belong to the generator (with gradient):")
        for x in parameters:
            logger.info('{} is leaf: {}'.format(x[0], x[1].is_leaf))
        
        feature_matching = False # getattr(model.discriminator, 'feature_matching', False)
        use_wgan = False # getattr(model.discriminator, '_use_wgan', False) 
        if use_wgan:
            optimizer = Optimizer.from_params(parameters, params.pop("optimizer_wgan"))
            params.pop("optimizer", None)
        else:
            optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
            params.pop("optimizer_wgan", None)

        if dis_params:
            if use_wgan:
                optimizer_dis = Optimizer.from_params(dis_params, params.pop("optimizer_wgan_dis"))
                params.pop("optimizer_dis", None)
            else:
                optimizer_dis = Optimizer.from_params(dis_params, params.pop("optimizer_dis"))
                params.pop("optimizer_wgan_dis", None)
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
                   rec_in_training = rec_in_training,
                   dis_min_loss = dis_min_loss, 
                   dis_skip_nepoch = dis_skip_nepoch,
                   gen_skip_nepoch = gen_skip_nepoch,
                   gen_pretraining = gen_pretraining,
                   dis_loss_scalar = dis_loss_scalar,
                   gen_loss_scalar = gen_loss_scalar,
                   kld_loss_scalar = kld_loss_scalar,
                   bpr_loss_scalar = bpr_loss_scalar,
                   kld_update_rate = kld_update_rate,
                   kld_update_unit = kld_update_unit,
                   feature_matching = feature_matching,
                   use_wgan = use_wgan,
                   clip_val = clip_val,
                   sort_by_length = sort_by_length,
                   consecutive_update = consecutive_update,
                   dis_max_nbatch = dis_max_nbatch,
                   gen_max_nbatch = gen_max_nbatch,
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

