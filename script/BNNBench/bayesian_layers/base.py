import torch
import torch.nn as nn

default_priors = {
    "prior_mu": 0,
    "prior_sigma": 0.02,
    "posterior_mu_initial": (0, 0.02),
    "posterior_rho_initial": (-5, 0.1),
}


class BayesianBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for module in self.children():
            x = module(x)
        kl = 0.0
        for module in self.children():
            if hasattr(module, "kl_loss"):
                kl = kl + module.kl_loss()
        return x, kl


def gaussian_kl(mu_q, sig_q, mu_p, sig_p):
    kl = (
        0.5
        * (
            2 * torch.log(sig_p / sig_q)
            - 1
            + (sig_q / sig_p).pow(2)
            + ((mu_p - mu_q) / sig_p).pow(2)
        ).sum()
    )
    return kl
