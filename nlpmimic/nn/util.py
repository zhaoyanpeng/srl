import torch
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-15, dim=-1):
    """ Copied from Pytorch.
    Args:
        logits: `[..., num_features]` unnormalized log probabilities
    """
    # gumbel is to approximate the sampling process
    gumbels = -torch.empty_like(logits).exponential_().clamp_(min=eps).log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # softmax is to approximate the argmax operation
    y_soft = gumbels.softmax(dim) 

    # soft version
    ret_soft = y_soft
    ret_soft_log = F.log_softmax(gumbels, dim=dim)

    # straight through
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret_hard = y_hard - y_soft.detach() + y_soft

    return ret_hard, ret_soft, ret_soft_log

def smaple_gumbel(logits, eps=1e-15):
    gumbels = -torch.empty_like(logits).exponential_().clamp_(min=eps).log()  # ~Gumbel(0,1)
    return gumbels 

def gumbel_sinkhorn(logits, mask, tau=1, niter=10, noise_factor=1.0, eps=1e-15, dim=-1):
    """ Pseudo doubly stochastic matrice.
    Args:
        logits: `[batch_size, n, num_features]` unnormalized log probabilities
    """
    # gumbel is to approximate the sampling process
    gumbels = -torch.empty_like(logits).exponential_().clamp_(min=eps).log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels * noise_factor) / tau
    # sinkhorn is to approximate the argmax operation
    mask = (mask == 0).unsqueeze(-1)
    for _ in range(niter): 
        gumbels = gumbels.masked_fill_(mask, eps)
        gumbels = gumbels - torch.logsumexp(gumbels, dim=1, keepdim=True) 
        gumbels = gumbels - torch.logsumexp(gumbels, dim=2, keepdim=True) 
    y_soft = torch.exp(gumbels)

    # soft version
    ret_soft = y_soft
    ret_soft_log = gumbels 

    # straight through
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret_hard = y_hard - y_soft.detach() + y_soft

    return ret_hard, ret_soft, ret_soft_log

