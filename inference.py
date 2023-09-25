import torch
import numpy as np
from hmc_torch import HMC


def gen_x(phi, ps_model, device=None):
    ps = ps_model(phi)
    return torch.fft.ifft2(torch.fft.fft2(torch.randn(ps.shape, device=device))*torch.sqrt(ps)).real

def sample_prior(n, device=None):
    """
    Sample from the (normalized) prior distribution.
    phi = (H0, Obh2) with H0 ~ U(0, 1), Obh2 ~ U(0, 1)
    (unnormalized prior is H0 ~ U(50, 90), Obh2 ~ U(0.0075, 0.0567))
    """
    phi = torch.rand(n, 2).to(device)
    return phi

def log_likelihood(phi, x, ps_model):
    """
    Compute the log likelihood of the Gaussian model.
    """
    x_dim = x.shape[-1]*x.shape[-2]
    ps = ps_model(phi)
    xf = torch.fft.fft2(x)

    term_pi = -(x_dim/2) * np.log(2*np.pi)
    term_logdet = -0.5 * torch.sum(torch.log(ps), dim=(-1, -2)) # The determinant is the product of the diagonal elements of the PS
    term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / ps, dim=(-1, -2))/x_dim # We divide by x_dim because of the normalization of the FFT
    return term_pi + term_logdet + term_x

def log_prior(phi):
    """
    Compute the log (normalized) prior of the parameters.
    """
    H0, Obh2 = phi[..., 0], phi[..., 1]
    term_H0 = torch.log(torch.logical_and(H0 >= 0.0, H0 <= 1.0).float()) #gives either 0 or -inf
    term_Obh2 = torch.log(torch.logical_and(Obh2 >= 0.0, Obh2 <= 1.0).float()) #gives either 0 or -inf
    return term_H0 + term_Obh2

def log_posterior(phi, x, ps_model):
    """
    Compute the log posterior of the parameters (not normalized by the evidence).
    """
    return log_likelihood(phi, x, ps_model) + log_prior(phi)

def infer(x, ps_model, nchains=20, nsamples=200, burnin=20, step_size=0.001, nleap=30, epsadapt=0, device=None):
    log_prob = lambda phi: log_posterior(phi, x, ps_model)
    def log_prob_grad(phi):
        """ Compute the log posterior and its gradient."""
        phi.requires_grad_(True)
        log_prob = log_posterior(phi, x, ps_model)
        grad_log_prob = torch.autograd.grad(log_prob, phi, grad_outputs=torch.ones_like(log_prob))[0]
        phi.requires_grad_(False)
        return log_prob.detach(), grad_log_prob
    hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)


    phi_0 = sample_prior(nchains, device=device)
    return hmc.sample(phi_0, nsamples=nsamples, burnin=burnin,step_size=step_size, nleap=nleap, epsadapt=epsadapt)