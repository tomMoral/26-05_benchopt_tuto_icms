from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    """Denoising with WaveletDictDenoiser (classical, non-iterative)."""

    name = "WaveletDict"

    requirements = ["pip::ptwt"]

    def set_objective(self, y, sigma):
        self.y = y
        self.sigma = sigma

    def run(self, _):
        denoiser = dinv.models.WaveletDictDenoiser()
        with torch.no_grad():
            self.x_hat = denoiser(self.y, self.sigma)

    def get_result(self):
        return dict(x_hat=self.x_hat)
