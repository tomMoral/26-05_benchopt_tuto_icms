from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    """Denoising with DnCNN (pretrained deep network, non-iterative)."""

    name = "DnCNN"

    def set_objective(self, y, sigma):
        self.y = y
        self.sigma = sigma
        self.device = y.device

    def run(self, _):
        denoiser = dinv.models.DnCNN(
            pretrained='download', device=self.device
        )
        with torch.no_grad():
            self.x_hat = denoiser(self.y, self.sigma)

    def get_result(self):
        return dict(x_hat=self.x_hat)
