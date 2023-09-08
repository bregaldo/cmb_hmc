import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchcubicspline


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for i in range(self.n_hidden_layers + 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

class CMBCov(nn.Module):

    def __init__(self):
        super().__init__()

        # MLP
        self.mlp = MLP(2, 100, 128, 2)
        CKPT_FOLDER = '/mnt/home/dheurtel/ceph/02_checkpoints/SIGMA_EMULATOR'
        MODEL_ID = 'Emulator_H0_ombh2_1'
        ckpt = torch.load(os.path.join(CKPT_FOLDER, MODEL_ID + '.pt'))
        self.mlp.load_state_dict(ckpt['network'])

        # Useful variables
        wn = (256*np.fft.fftfreq(256, d=1.0)).reshape((256,) + (1,) * (2 - 1))
        wn_iso = np.zeros((256,256))
        for i in range(2):
            wn_iso += np.moveaxis(wn, 0, i) ** 2
        wn_iso = np.sqrt(wn_iso)
        indices = np.fft.fftshift(wn_iso).diagonal()[128:] ## The value of the wavenumbers along which we have the power spectrum diagonal
        self.register_buffer("torch_indices", torch.tensor(indices))
        self.register_buffer("torch_wn_iso", torch.tensor(wn_iso, dtype=torch.float32))
    
    def forward(self, phi):
        phi = (phi - torch.tensor([70, 32e-3]).to(phi.device))/torch.tensor([20,25e-3]).to(phi.device)
        torch_diagonals = self.mlp(phi) ## Shape (batch_size, 128) (128 is the number of wavenumbers along which we have the power spectrum diagonal)
        torch_diagonals = torch_diagonals.reshape((128, -1)) ## Shape (128, batch_size) to be able to use torchcubicspline
        spline = torchcubicspline.NaturalCubicSpline(torchcubicspline.natural_cubic_spline_coeffs(self.torch_indices, torch_diagonals))
        return torch.moveaxis(spline.evaluate(self.torch_wn_iso), -1, 0)
