import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BayesianBase, default_priors, gaussian_kl


class Linear(BayesianBase):
    def __init__(self, in_features, out_features, bias=True, priors=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        if priors is None:
            priors = default_priors

        self.prior_mu = priors["prior_mu"]
        self.prior_sigma = priors["prior_sigma"]
        self.posterior_mu_initial = priors["posterior_mu_initial"]
        self.posterior_rho_initial = priors["posterior_rho_initial"]

        self.W_mu = nn.Parameter(
            torch.empty((out_features, in_features), device=self.device)
        )
        self.W_rho = nn.Parameter(
            torch.empty((out_features, in_features), device=self.device)
        )

        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = nn.Parameter(
                torch.empty((out_features), device=self.device)
            )
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
            bias = self.bias_mu if self.bias else None

        return F.linear(input, weight, bias)

    def kl_loss(self):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        kl = gaussian_kl(self.prior_mu, self.prior_sigma, self.W_mu, W_sigma)
        if self.bias:
            kl += gaussian_kl(self.prior_mu, self.prior_sigma, self.bias_mu, bias_sigma)
        return kl
