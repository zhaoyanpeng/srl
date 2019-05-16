"""
Helper functions for Trainers
"""
from typing import cast, Dict, Optional, List, Tuple, Union, Iterable, Iterator, Any, NamedTuple
import sys, logging, random

from allennlp.common.util import ensure_list, add_noise_to_dict_values, lazy_groups_of
from allennlp.common.checks import ConfigurationError, check_for_data_path, check_for_gpu
from allennlp.common.params import Params
from allennlp.training.util import sparse_clip_norm
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Instance
from allennlp.models.model import Model

logger = logging.getLogger(__name__)

NYT_READER_MODE = 'srl_nyt'
GAN_READER_MODE = 'srl_gan'
DEFAULT_READER_MODE = 'basic'

# We want to warn people that tqdm ignores metrics that start with underscores
# exactly once. This variable keeps track of whether we have.
class HasBeenWarned:
    tqdm_ignores_underscores = False

def datasets_from_params(params: Params, reader_mode: str = DEFAULT_READER_MODE) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    if reader_mode == GAN_READER_MODE or reader_mode == NYT_READER_MODE:
        reading_list = ["train_dx_path", 
                        "train_dy_path", 
                        "validation_data_path", 
                        "test_data_path", "vocab_src_path"]
        if reader_mode == GAN_READER_MODE:
            reading_list += ["train_dy_context_path", "train_dy_appendix_path"]
    else:
        reading_list = ["train_data_path", 
                        "validation_data_path", 
                        "test_data_path", "vocab_src_path"]

    for data_name in reading_list:
        data_path = params.get(data_name, None)
        if data_path is not None:
            check_for_data_path(data_path, data_name)

    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    
    if reader_mode == GAN_READER_MODE or reader_mode == NYT_READER_MODE:
        train_dx_path = params.pop('train_dx_path', None)
        logger.info("Reading training data of domain x from %s", train_dx_path)
        if train_dx_path is None and reader_mode == NYT_READER_MODE:
            train_dx_data = [] # empty dx is allowed in this mode 
        else:
            train_dx_data = dataset_reader.read(train_dx_path)

        train_dy_path = params.pop('train_dy_path', None)
        logger.info("Reading training data of domain y from %s", train_dy_path)
        if train_dy_path is None and reader_mode == NYT_READER_MODE:
            train_dy_data = [] # empty dx is allowed in this mode
        else:
            train_dy_data = dataset_reader.read(train_dy_path)

        if reader_mode == NYT_READER_MODE:
            train_dy_context_path = params.pop('train_dy_context_path') 
            train_dy_appendix_path = params.pop('train_dy_appendix_path')
            logger.info("Reading training (nytimes.context ) data of domain y from %s", train_dy_context_path)
            logger.info("Reading training (nytimes.appendix) data of domain y from %s", train_dy_appendix_path)

            train_dy_firstk = params.pop('train_dy_firstk', sys.maxsize)
            train_dx_firstk = params.pop('train_dx_firstk', sys.maxsize)
            
            nytimes_reader = DatasetReader.from_params(params.pop('nytimes_reader'))
            train_dy_nyt_data = nytimes_reader._read(train_dy_context_path, 
                                                     train_dy_appendix_path, 
                                                     appendix_type='nyt_learn',
                                                     firstk = train_dy_firstk)
            train_dy_nyt_data = ensure_list(train_dy_nyt_data)
            train_dy_data += train_dy_nyt_data # combine nytimes with gold
            
            add_unlabeled_noun = params.get('add_unlabeled_noun', False) 
            using_labeled_noun = params.pop('using_labeled_noun', True) 
            if add_unlabeled_noun:
                train_dx_context_path = params.pop('train_dx_context_path', None)
                train_dx_appendix_path = params.pop('train_dx_appendix_path')   
                
                if train_dx_context_path is None: # x domain has its own context file
                    train_dx_context_path = train_dy_context_path

                allow_null_predicate = nytimes_reader.allow_null_predicate

                if using_labeled_noun:
                    nytimes_reader.allow_null_predicate = False 
                    appendix_type = 'nyt_learn'
                else:
                    nytimes_reader.allow_null_predicate = True 
                    appendix_type = 'nyt_infer'
                train_dx_nyt_data = nytimes_reader._read(train_dx_context_path,
                                                         train_dx_appendix_path,
                                                         appendix_type=appendix_type,
                                                         firstk = train_dx_firstk)
                train_dx_nyt_data = ensure_list(train_dx_nyt_data)
                nytimes_reader.allow_null_predicate = allow_null_predicate
                train_dx_data += train_dx_nyt_data # combine nytimes with gold
            else:
                params.pop('train_dx_context_path', None)
                params.pop('train_dx_appendix_path', None)

        datasets: Dict[str, Iterable[Instance]] = {"train_dx": train_dx_data,
                                                   "train_dy": train_dy_data}
    else:
        train_data_path = params.pop('train_data_path')
        logger.info("Reading training data from %s", train_data_path)
        train_data = dataset_reader.read(train_data_path)

        datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    validation_ontraining_data = params.pop('validation_ontraining_data', False)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data
        if validation_ontraining_data:
            datasets["validation"] += train_dx_data 
            pass

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data
    
    vocab_src_path = params.pop("vocab_src_path", None)
    if vocab_src_path is not None:
        logger.info("Reading vocab source data from %s", vocab_src_path)
        vocab_data = validation_and_test_dataset_reader.read(vocab_src_path)
        
        if reader_mode == NYT_READER_MODE:
            vocab_data += train_dy_nyt_data

        datasets["vocab"] = vocab_data
    return datasets

def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0,
                    reverse: bool = False) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0], reverse=reverse)
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

def shuffle_argument_indices(instances: Iterable[Instance]):
    for instance in instances:
        arg_mask = instance['argument_mask'].sequence_index
        arg_indices = instance['argument_indices'].sequence_index 
        narg = sum(arg_mask)
        valid_arg_indices = arg_indices[:narg]
        random.shuffle(valid_arg_indices)
        arg_indices[:narg] = valid_arg_indices

class DataSampler(object):
    
    def __init__(self, 
                 instances: Iterable[Instance], 
                 iterator: DataIterator, 
                 sort_by_length: bool,
                 shuffle_arguments: bool = False):
        super().__init__()
        self.instances = ensure_list(instances)
        self.sort_by_length = sort_by_length
        self.iterator = iterator
        self.shuffle_arguments = shuffle_arguments

        self.data_pivot = 0 # initial pointer
        
        random.shuffle(self.instances)

    def sample(self, batch_size: int) -> Iterable[Instance]:
        samples = self.instances[self.data_pivot : self.data_pivot + batch_size]
        nsample = len(samples)
        if nsample < batch_size:
            samples += self.instances[0 : batch_size - nsample]
            self.data_pivot = batch_size - nsample 
            random.shuffle(self.instances)
            if self.sort_by_length: # sort
                self.instances[:] = sort_by_padding(
                                            self.instances, 
                                            self.iterator._sorting_keys,
                                            self.iterator.vocab,
                                            self.iterator._padding_noise,
                                            reverse=True)
                print('\n>>>>>>>|data| = {}, pivot = {}\n'.format(len(self.instances), self.data_pivot))
        else:
            self.data_pivot += batch_size

        if self.shuffle_arguments:
            shuffle_argument_indices(samples)
        batch = Batch(samples)
        batch.index_instances(self.iterator.vocab)
        batch = batch.as_tensor_dict()
        return batch 

class DataLazyLoader(object):

    def __init__(self, 
                 instances: Iterable[Instance], 
                 iterator: DataIterator, 
                 sort_by_length: bool,
                 shuffle_arguments: bool = False):
        super().__init__()
        self.instances = ensure_list(instances)
        self.sort_by_length = sort_by_length
        self.iterator = iterator
        self.shuffle_arguments = shuffle_arguments

        random.shuffle(self.instances)
    
    def nbatch(self):
        return self.iterator.get_num_batches(self.instances)

    def _iterate(self):
        random.shuffle(self.instances)
        if self.sort_by_length: 
            self.instances[:] = sort_by_padding(
                                        self.instances, 
                                        self.iterator._sorting_keys,
                                        self.iterator.vocab,
                                        self.iterator._padding_noise,
                                        reverse=True)
            print('\n<<<<<<<|data| = {}\n'.format(len(self.instances)))
        yield from self.instances

    def sample(self) -> Iterator[TensorDict]:
        for samples in lazy_groups_of(self._iterate(), self.iterator._batch_size):
            if self.shuffle_arguments:
                shuffle_argument_indices(samples)
            batch = Batch(samples)
            batch.index_instances(self.iterator.vocab)
            batch = batch.as_tensor_dict()
            yield batch, len(samples)

def rescale_gradients(model: Model, 
                      param_signatures: List[str], 
                      grad_norm: Optional[float] = None) -> Optional[float]:
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = []
        # invoked in model.parameters() 
        # see `https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.parameters`
        for n, p in model.named_parameters(): 
            if n in param_signatures and p.grad is not None:
                parameters_to_clip.append(p)
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None

def clip_parameters(model: Model,
                    param_signatures: List[str],
                    clip_value: float = 0.01) -> None:
    """ Used by WGAN which requires parameters to be clipped
    """
    for n, p in model.named_parameters(): 
        if n in param_signatures and p.grad is not None:
            p.data.clamp_(-clip_value, clip_value)
    return None

def decode_metrics(model, batch, training: bool, 
                   optimizing_generator: bool = True,
                   relying_on_generator: bool = True, 
                   caching_feature_only: bool = False,
                   retrive_crossentropy: bool = False,
                   supervisely_training: bool = False,
                   peep_prediction: bool = False):
    # forward pass
    output_dict = model(**batch, 
                optimizing_generator = optimizing_generator,
                relying_on_generator = relying_on_generator,
                caching_feature_only = caching_feature_only,
                retrive_crossentropy = retrive_crossentropy,
                supervisely_training = supervisely_training)
    try:
        loss = ce_loss = kl_loss = gp_loss = bp_loss = None 
        if training: # compulsory loss
            loss = output_dict["loss"] # loss of the generator or discriminator
            if loss is not None:
                loss += model.get_regularization_penalty()
            if 'kl_loss' in output_dict:
                kl_loss = output_dict["kl_loss"]
            if 'gp_loss' in output_dict:
                gp_loss = output_dict["gp_loss"]
            if 'bp_loss' in output_dict:
                bp_loss = output_dict["bp_loss"]
        
        # can be added into compulsory loss in semi-supervised setting
        if retrive_crossentropy: 
            ce_loss = output_dict["ce_loss"]

        if peep_prediction: #and not for_training:
            output_dict = model.decode(output_dict)
            tokens = output_dict['tokens'][:5]
            labels = output_dict['srl_tags'][:5]
            gold_srl = output_dict['gold_srl'][:5]
            predicates = output_dict["predicate"]
            pindexes = output_dict["predicate_index"]
            for token, label, glabel, p, pid in zip(tokens, labels, gold_srl, predicates, pindexes):
                xx = ['({}|{}: {}_{})'.format(idx, t, g, l) for idx, (t, g, l) in enumerate(zip(token, glabel, label))]
                print('{} {}.{}\n'.format(xx, p, pid))
    except KeyError:
        if training:
            raise RuntimeError("The model you are trying to optimize does not contain a '*loss' key"
                               "in the output of model.forward(inputs). output_dict: {}".format(output_dict))
        loss = ce_loss = kl_loss = bp_loss = gp_loss = None
    return loss, ce_loss, kl_loss, bp_loss, gp_loss

def decode_metrics_vae(model, batch, training: bool, 
                   retrive_crossentropy: bool = False,
                   supervisely_training: bool = False,
                   peep_prediction: bool = False):
    # forward pass
    output_dict = model(**batch, 
                retrive_crossentropy = retrive_crossentropy,
                supervisely_training = supervisely_training)
    try:
        loss = ce_loss = kl_loss = L = L_u = H = C = None 
        if training: # compulsory loss
            loss = output_dict["loss"] # loss of the generator or discriminator
            if loss is not None:
                loss += model.get_regularization_penalty()
            if 'kl_loss' in output_dict:
                kl_loss = output_dict["kl_loss"]
            if 'gp_loss' in output_dict:
                gp_loss = output_dict["gp_loss"]
            if 'bp_loss' in output_dict:
                bp_loss = output_dict["bp_loss"]
            if 'L' in output_dict:
                L = output_dict['L']
            if 'L_u' in output_dict:
                L_u = output_dict['L_u']
            if 'H' in output_dict:
                H = output_dict['H']
            if 'C' in output_dict:
                C = output_dict['C']
        
        # can be added into compulsory loss in semi-supervised setting
        if retrive_crossentropy: 
            ce_loss = output_dict["ce_loss"]

        if peep_prediction: #and not for_training:
            output_dict = model.decode(output_dict)
            peep_predictions(output_dict)
    except KeyError:
        if training:
            raise RuntimeError("The model you are trying to optimize does not contain a '*loss' key"
                               "in the output of model.forward(inputs). output_dict: {}".format(output_dict))
        loss = ce_loss = kl_loss = L = L_u = H = C = None 
    return loss, ce_loss, kl_loss, L, L_u, H, C 

def peep_predictions(output_dict: Dict[str, Any]):
    tokens = output_dict['tokens'][:5]
    labels = output_dict['srl_tags'][:5]
    gold_srl = output_dict['gold_srl'][:5]
    predicates = output_dict["predicate"]
    pindexes = output_dict["predicate_index"]
    argidxes = output_dict["arg_idxes"]
    for token, label, glabel, p, pid, aid in zip(tokens, labels, gold_srl, predicates, pindexes, argidxes):
        xx = ['({}|{}: {}_{})'.format(idx, t, g, l) for idx, (t, g, l) in enumerate(zip(token, glabel, label))]
        xx = ' '.join(xx)
        print('{} {}.{} {}\n'.format(xx, p, pid, aid))
    
def get_metrics(model: Model, 
                total_loss: float, 
                num_batches: int, 
                ce_loss: float = None,
                kl_loss: float = None,
                bp_loss: float = None,
                gp_loss: float = None,
                L: float = None,
                L_u: float = None,
                H: float = None,
                C: float = None,
                generator_loss: float = None,
                discriminator_loss: float = None,
                crossentropy_loss: float = None, 
                reset: bool = False) -> Dict[str, float]:
    """
    Gets the metrics but sets ``"loss"`` to
    the total loss divided by the ``num_batches`` so that
    the ``"loss"`` metric is "average loss per batch".
    """
    metrics = model.get_metrics(reset=reset)
    metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
    if generator_loss is not None:
        metrics["g_loss"] = generator_loss
    if discriminator_loss is not None:
        metrics["d_loss"] = discriminator_loss 

    if ce_loss is not None:
        metrics["ce_loss"] = ce_loss 
    if kl_loss is not None:
        metrics["kl_loss"] = kl_loss 
    if bp_loss is not None:
        metrics["bp_loss"] = bp_loss 
    if gp_loss is not None:
        metrics["gp_loss"] = gp_loss 
    
    if L is not None:
        metrics['L'] = L
    if L_u is not None:
        metrics['L_u'] = L_u
    if H is not None:
        metrics['H'] = H
    if C is not None:
        metrics['C'] = C 

    return metrics
