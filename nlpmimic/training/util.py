"""
Helper functions for Trainers
"""
from typing import Dict, List, Iterable, Optional 
import sys, logging

from allennlp.common.util import ensure_list
from allennlp.common.checks import ConfigurationError, check_for_data_path, check_for_gpu
from allennlp.common.params import Params
from allennlp.training.util import sparse_clip_norm
from allennlp.data.dataset_readers import DatasetReader
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

def get_metrics(model: Model, 
                total_loss: float, 
                num_batches: int, 
                kl_loss: float = None,
                bp_loss: float = None,
                gp_loss: float = None,
                generator_loss: float = None,
                discriminator_loss: float = None,
                reconstruction_loss: float = None, 
                reset: bool = False) -> Dict[str, float]:
    """
    Gets the metrics but sets ``"loss"`` to
    the total loss divided by the ``num_batches`` so that
    the ``"loss"`` metric is "average loss per batch".
    """
    metrics = model.get_metrics(reset=reset)
    metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
    if reconstruction_loss is not None:
        metrics["rec_loss"] = float(reconstruction_loss / num_batches) if num_batches > 0 else 0.0
    
    if kl_loss is not None:
        metrics["kl_loss"] = kl_loss 
    
    if bp_loss is not None:
        metrics["bp_loss"] = bp_loss 
    
    if gp_loss is not None:
        metrics["gp_loss"] = gp_loss 

    if generator_loss is not None:
        metrics["g_loss"] = generator_loss
    
    if discriminator_loss is not None:
        metrics["d_loss"] = discriminator_loss 
    return metrics
