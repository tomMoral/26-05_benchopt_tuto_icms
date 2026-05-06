from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):

    name = "DnCNN"

    def set_objective(self, y, sigma):
        self.y, self.sigma = y, sigma

    def run(self, _):
        denoiser = dinv.models.DnCNN(
            pretrained='download', device=self.y.device
        )
        with torch.no_grad():
            self.x_hat = denoiser(self.y, self.sigma)

    def get_result(self):
        return dict(x_hat=self.x_hat)
