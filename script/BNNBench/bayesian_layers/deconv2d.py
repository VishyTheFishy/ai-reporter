import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BayesianBase, default_priors, gaussian_kl


class ConvTranspose2d(BayesianBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        priors=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.stride = stride
        self.padding = padding

        if priors is None:
            priors = default_priors

        self.prior_mu = priors["prior_mu"]
        self.prior_sigma = priors["prior_sigma"]
        self.posterior_mu_initial = priors["posterior_mu_initial"]
        self.posterior_rho_initial = priors["posterior_rho_initial"]

        self.W_mu = nn.Parameter(
            torch.empty((in_channels, out_channels, kernel_size, kernel_size))
        )
        self.W_rho = nn.Parameter(
            torch.empty((in_channels, out_channels, kernel_size, kernel_size))
        )

        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(out_channels))
            self.bias_rho = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(input)
            W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * W_sigma
            if self.bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(input)
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu
        return F.conv_transpose2d(input, weight, bias, self.stride, self.padding)

    def kl_loss(self):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        kl = gaussian_kl(self.prior_mu, self.prior_sigma, self.W_mu, W_sigma)
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            kl += gaussian_kl(self.prior_mu, self.prior_sigma, self.bias_mu, bias_sigma)
        return kl
