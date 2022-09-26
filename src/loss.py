import math
import torch
from torch.distributions.normal import Normal


def gaussian_loss(y_hat, y, log_std_min=-9.0):
    

    assert y_hat.shape[-1] == y.shape[-1] #(B, C, L)

    mean = y_hat[:, :1, :]
    log_std = torch.clamp(y_hat[:, 1:, :], min=log_std_min)

    log_probs = -0.5 * (- math.log(2.0 * math.pi) - 2. * log_std - torch.pow(y - mean, 2) * torch.exp((-2.0 * log_std)))
    
#     if size_average:
#             return losses.mean()
#         else:
#             return losses.mean(1).sum(0)
    
    return log_probs.squeeze().mean()



def KL_gaussians(mu_q, logs_q, mu_p, logs_p, log_std_min=-6.0, regularization=True):
    # KL (q || p)
    # q ~ N(mu_q, logs_q.exp_()), p ~ N(mu_p, logs_p.exp_())
    logs_q_org = logs_q
    logs_p_org = logs_p
    logs_q = torch.clamp(logs_q, min=log_std_min)
    logs_p = torch.clamp(logs_p, min=log_std_min)
    KL_loss = (logs_p - logs_q) + 0.5 * ((torch.exp(2. * logs_q) + torch.pow(mu_p - mu_q, 2)) * torch.exp(-2. * logs_p) - 1.)
    if regularization:
        reg_loss = torch.pow(logs_q_org - logs_p_org, 2)
    else:
        reg_loss = None
    return KL_loss, reg_loss
