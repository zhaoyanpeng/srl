import torch
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-15, dim=-1):
    """ Copied from Pytorch.
    Args:
        logits: `[..., num_features]` unnormalized log probabilities
    """
    gumbels = -torch.empty_like(logits).exponential_().clamp_(min=eps).log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim) 

    # soft version
    ret_soft = y_soft
    ret_soft_log = F.log_softmax(gumbels, dim=dim)

    # straight through
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret_hard = y_hard - y_soft.detach() + y_soft

    return ret_hard, ret_soft, ret_soft_log
    
