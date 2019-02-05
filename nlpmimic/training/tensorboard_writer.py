from typing import List, Optional, Callable
import logging
import os

import torch

from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class GanSrlTensorboardWriter(TensorboardWriter):
    """
    Rewrites `log_parameter_and_gradient_statistics`.

    Parameters
    ----------
    Refer to the base class.
    """
    def __init__(self,
                 get_batch_num_total: Callable[[], int],
                 serialization_dir: Optional[str] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False) -> None:
        
        super(GanSrlTensorboardWriter, self).__init__(
                 get_batch_num_total,
                 serialization_dir,
                 summary_interval,
                 histogram_interval,
                 should_log_parameter_statistics,
                 should_log_learning_rate)

    def log_parameter_and_gradient_statistics(self, # pylint: disable=invalid-name
                                              model: Model,
                                              batch_grad_norm: float,
                                              param_signatures: List[str],
                                              model_signature: str = '') -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        if self._should_log_parameter_statistics:
            # Log parameter values to Tensorboard
            for name, param in model.named_parameters():
                if name not in param_signatures:
                    continue

                self.add_train_scalar("parameter_mean/" + name, param.data.mean())
                self.add_train_scalar("parameter_std/" + name, param.data.std())
                if param.grad is not None:
                    if param.grad.is_sparse:
                        # pylint: disable=protected-access
                        grad_data = param.grad.data._values()
                    else:
                        grad_data = param.grad.data

                    # skip empty gradients
                    if torch.prod(torch.tensor(grad_data.shape)).item() > 0: # pylint: disable=not-callable
                        self.add_train_scalar("gradient_mean/" + name, grad_data.mean())
                        self.add_train_scalar("gradient_std/" + name, grad_data.std())
                    else:
                        # no gradient for a parameter with sparse gradients
                        logger.info("No gradient for %s, skipping tensorboard logging.", name)
            # norm of gradients
            if batch_grad_norm is not None:
                name = model_signature + "gradient_norm"
                self.add_train_scalar(name, batch_grad_norm)


