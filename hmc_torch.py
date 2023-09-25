""" Adapted from https://github.com/modichirag/VBS/blob/main/src/pyhmc.py """

import numpy as np
import torch
from tqdm import tqdm

class Sampler():

    def __init__(self):
        self.samples = []
        self.accepts = []
        self.Hs = []
        self.counts = []
        self.i = 0

    def to_tensor(self):
        for key in self.__dict__:
            if isinstance(self.__dict__[key], list):
                dim_per_key = {'samples': -2, 'accepts': -1, 'Hs': -2, 'counts': -2}
                self.__dict__[key] = torch.stack(self.__dict__[key], dim=dim_per_key[key])

    def to_list(self):
        for key in self.__dict__:
            if isinstance(self.__dict__[key], torch.Tensor):
                self.__dict__[key] = self.__dict__[key].tolist()

    def appends(self, q, acc, Hs, count):
        self.i += 1
        self.accepts.append(acc)
        self.samples.append(q)
        self.Hs.append(Hs)
        self.counts.append(count)
        
    def save(self, path):
        pass

class DualAveragingStepSize():
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75, nadapt=0):
        self.initial_step_size = initial_step_size 
        self.mu = torch.log(initial_step_size) #torch.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma * 2 #parameter to tune
        self.t = t0
        self.kappa = kappa
        self.error_sum = torch.zeros_like(self.initial_step_size).to(initial_step_size.device) #0
        self.log_averaged_step = torch.zeros_like(self.initial_step_size).to(initial_step_size.device) #0
        self.nadapt = nadapt
        
    def update(self, p_accept):
        p_accept[p_accept > 1] = 1.
        p_accept[torch.isnan(p_accept)] = 0.
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return torch.exp(log_step), torch.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n" % step_size)
        else:
            step_size = torch.exp(self.log_averaged_step)
        return step_size
    


class HMC():

    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None, precision=torch.float32):

        self.precision = precision

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.log_prob_and_grad = log_prob_and_grad

        # Convert to precision
        if self.log_prob is not None:
            self.log_prob = lambda x: log_prob(x).to(self.precision)
        if self.grad_log_prob is not None:
            self.grad_log_prob = lambda x: grad_log_prob(x).to(self.precision)
        if self.log_prob_and_grad is not None:
            self.log_prob_and_grad = lambda x: tuple([y.to(self.precision) for y in log_prob_and_grad(x)])

        if invmetric_diag is None:
            self.invmetric_diag = 1.
        else:
            self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        assert not((self.grad_log_prob is None) and (self.log_prob_and_grad is None))

        self.V = lambda x: -self.log_prob(x)
        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum(-1) #Sum across rows
        self.KE_g = lambda p: p * self.invmetric_diag

        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0

    def V_g(self, x):
        self.Vgcount += 1
        if self.grad_log_prob is not None:
            v_g = self.grad_log_prob(x)
        elif self.log_prob_and_grad is not None:
            v, v_g = self.log_prob_and_grad(x)
        return -v_g.detach()

    def V_vandg(self, x):
        if self.log_prob_and_grad is not None:
            self.Vgcount += 1
            v, v_g = self.log_prob_and_grad(x)
            return -v, -v_g
        else:
            raise NotImplementedError

    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum(-1) #sum across rows

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q, p, Vq=None):
        if Vq is None:
            self.Hcount += 1
            Vq = self.V(q)
        return Vq + self.KE(p)

    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1
        q0, p0 = q, p
        s = step_size.unsqueeze(-1)
        try:
            p = p - 0.5 * s * self.V_g(q)
            for i in range(N - 1):
                q = q + s * self.KE_g(p)
                p = p - s * self.V_g(q)
            q = q + s * self.KE_g(p)
            p = p - 0.5 * s * self.V_g(q)
            return q, p

        except Exception as e:
            print("exception : ", e)
            return q0, p0

    def leapfrog_Vgq(self, q, p, N, step_size, V_q=None, V_gq=None):
        self.leapcount += 1
        q0, p0, V_q0, V_gq0 = q, p, V_q, V_gq
        try:
            if V_gq is None:
                p = p - 0.5 * step_size * self.V_g(q)
            else:
                p = p - 0.5 * step_size * V_gq
            for _ in tqdm(range(N - 1)):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q)

            q = q + step_size * self.KE_g(p)
            if self.log_prob_and_grad is not None:
                V_q1, V_gq1 = self.V_vandg(q)
            else:
                V_q1, V_gq1 = None, self.V_g(q)
            p = p - 0.5 * step_size * V_gq1
            return q, p, V_q1, V_gq1

        except Exception as e:
            print("exception : ", e)
            return q0, p0, V_q0, V_gq0
        
    def metropolis(self, q0, p0, q1, p1, V_q0=None, V_q1=None, u=None):
        
        H0 = self.H(q0, p0, V_q0)
        H1 = self.H(q1, p1, V_q1)
        prob = torch.exp(H0 - H1)

        if u is None:
            u = torch.rand(prob.shape[0], device=prob.device)

        qq = q1.clone()
        pp = p1.clone()
        acc = torch.ones_like(prob)

        cond1 = torch.logical_or(torch.isnan(prob), torch.isinf(prob))
        cond1 = torch.logical_or(cond1, torch.sum(q0 - q1, dim=-1) == 0)

        qq[cond1] = q0[cond1]
        pp[cond1] = p0[cond1]
        acc[cond1] = -1.0

        cond2 = torch.logical_and(u > torch.min(torch.ones_like(u), prob), ~cond1)
        qq[cond2] = q0[cond2]
        pp[cond2] = p0[cond2]
        acc[cond2] = 0.0
        
        return qq, pp, acc, torch.stack([H0, H1], dim=-1)

    def step(self, q, nleap, step_size):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = torch.randn(q.shape, device=q.device, dtype=self.precision) * self.metricstd
        q1, p1 = self.leapfrog(q, p, nleap, step_size)
        q, p, accepted, Hs = self.metropolis(q, p, q1, p1)
        return q, p, accepted, Hs, torch.tensor([self.Hcount, self.Vgcount, self.leapcount])

    def adapt_stepsize(self, q, step_size, epsadapt, nleap):
        print("Adapting step size for %d iterations" % epsadapt)
        epsadapt_kernel = DualAveragingStepSize(step_size)

        for i in tqdm(range((epsadapt + 1))):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            q = q.detach()
            prob = torch.exp(Hs[...,0] - Hs[...,1])

            if i < epsadapt:
                step_size, avgstepsize = epsadapt_kernel.update(prob)
            elif i == epsadapt:
                _, step_size = epsadapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)
        return q, step_size
    
    def sample(self, q, step_size = 0.01, nsamples=20, burnin=10, nleap=30, p=None, callback=None, skipburn=True, epsadapt=0):
        if q.ndim == 1: q = q.unsqueeze(0) # Q must be at least 2D
        assert q.ndim == 2, "q must be 2D"

        q = q.to(self.precision)
        
        step_size = step_size * torch.ones((q.shape[0]), device=q.device, dtype=self.precision,requires_grad=False)
        state = Sampler()
        if epsadapt>0:
            q, step_size = self.adapt_stepsize(q, step_size, epsadapt, nleap)
        for i in tqdm(range(nsamples + burnin)):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            state.i += 1
            state.accepts.append(acc)
            if (skipburn and (i >= burnin)) or not skipburn:
                state.samples.append(q.cpu())
                state.Hs.append(Hs.cpu())
                state.counts.append(count)
                if callback is not None: callback(state)
        state.to_tensor()
        return state
