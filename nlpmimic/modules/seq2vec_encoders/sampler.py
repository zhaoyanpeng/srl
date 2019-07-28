import numpy as np

import torch
import torch.nn.functional as F

from allennlp.modules import Seq2VecEncoder


@Seq2VecEncoder.register("uniform")
class SamplerUniform(Seq2VecEncoder):
    """ Uniform distribution over the space of exponentially many discrete structures
        The probability of each structure is nearly zero, implying -inf in log form.
        So why do we bother to include it as a part of losses? Just return zero! 
    """
    def __init__(self):
        super(SamplerUniform, self).__init__()
        pass

@Seq2VecEncoder.register("gumbel")
class SamplerGumbel(Seq2VecEncoder):
    def __init__(self, tau: float, tau_prior: float):
        super(SamplerGumbel, self).__init__()
        self.tau = tau
        self.tau_prior = tau_prior
        
        self.tau_ratio = tau_prior / tau 

@Seq2VecEncoder.register("gaussian")
class SamplerGaussian(Seq2VecEncoder):
    
    def __init__(self, input_dim: int, output_dim: int):
        super(SamplerGaussian, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # defaults
        self.mu = self.std = None
        self.def_mu = self.def_std = None

        self._dense_layer = torch.nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, z: torch.Tensor, nsample=1):
        mu_and_std = self._dense_layer(z).unsqueeze(1)
        self.mu, std = torch.chunk(mu_and_std, chunks=2, dim=-1)
        self.std = F.softplus(std) # ensure positiveness 
        z = self.sample(self.mu, self.std, nsample)
        return z 
    
    def sample(self, mu: torch.Tensor, std: torch.Tensor, nsample: int = 1):
        batch_size, _, dim = mu.size()
        eps = torch.randn([batch_size, nsample, dim], device=mu.device)
        z = mu + std * eps 
        return z

    def lprob(self, sample: torch.Tensor, mu: torch.Tensor = None, std: torch.Tensor = None):
        if mu is None and std is None: # using the default normal distribution
            if self.def_mu is None or self.def_std is None:
                self.def_mu = torch.zeros(self.output_dim, device=sample.device) 
                self.def_std = torch.ones(self.output_dim, device=sample.device) 
            mu, std = self.def_mu, self.def_std

        var = torch.pow(std, 2) + 1e-15 # sanity check
        lprob = -0.5 * torch.log(2 * np.pi * var) - torch.pow(sample - mu, 2) / (2 * var) 
        return lprob 
        
