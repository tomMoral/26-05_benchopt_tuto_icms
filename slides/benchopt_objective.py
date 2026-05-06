from benchopt import BaseObjective

import torch
import deepinv as dinv


class Objective(BaseObjective):

    name = "Denoising"
    requirements = ["pip::deepinv"]
    sampling_strategy = 'run_once'

    def set_data(self, x_true, sigma):
        self.x_true, self.sigma = x_true, sigma
        self.y = x_true + sigma * torch.randn(x_true.shape)

    def get_objective(self):
        return dict(y=self.y, sigma=self.sigma)

    def evaluate_result(self, x_hat):
        psnr = dinv.metric.PSNR()(x_hat, self.x_true).item()
        return dict(psnr=psnr)

    def get_one_result(self):
        return dict(x_hat=self.y)
