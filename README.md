# HMC for CMB analysis

For $\epsilon \sim \mathcal{N}(0, \Sigma(\phi))$ a realization of a Gaussian CMB process whose covariance $\Sigma(\phi)$ is parametrized by cosmological parameters $\phi$, we want to sample efficiently $p(\phi | \epsilon)$ for a given prior distribution $p(\phi)$.

For now, we only focus on the cosmological parameters $\phi = (H_0, \omega_b)$ and consider the following prior distribution:
$$p(H_0) \sim \mathcal{U}(50, 90),\text{ and } p(\omega_b) \sim \mathcal{U}(0.0075, 0.0567).$$

## Install

A Python environment with a reasonably recent version of PyTorch is needed. Additional packages have to be installed:

    pip install camb arviz
    pip install git+https://github.com/patrick-kidger/torchcubicspline.git
