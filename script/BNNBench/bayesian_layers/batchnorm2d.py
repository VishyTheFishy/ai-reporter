import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BayesianBase, default_priors, gaussian_kl


class BatchNorm2d(BayesianBase):
    def __init__(self, num_features, eps=1.0e-5, momentum=0.1, affine=True, track_running_stats=True, priors=None) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.prior_mu = priors['prior_mu']
            self.prior_sigma = priors['prior_sigma']
            self.posterior_mu_initial = priors['posterior_mu_initial']
            self.posterior_rho_initial = priors['posterior_rho_initial']

            self.W_mu = nn.Parameter(torch.empty(num_features))
            self.W_rho = nn.Parameter(torch.empty(num_features))
            self.bias_mu = nn.Parameter(torch.empty(num_features))
            self.bias_ro = nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter("W_mu", None)
            self.register_parameter("W_rho", None)
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
