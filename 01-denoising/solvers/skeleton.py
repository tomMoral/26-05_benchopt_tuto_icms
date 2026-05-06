from benchopt import BaseSolver

import torch
import deepinv as dinv


class Solver(BaseSolver):
    """[PARTICIPANT SKELETON] — Add your denoiser here.

    Your task: pick a denoiser from deepinv and call it to produce x_hat.

    Useful reference:
      https://deepinv.github.io/deepinv/auto_examples/models/demo_denoiser_tour.html

    Examples of denoisers you can try:
      - dinv.models.DrUnet
      - dinv.models.SCUNet
      - dinv.models.SwinIR

    All denoisers share the same calling convention:
      x_hat = denoiser(noisy_image, sigma)
    """

    name = "MySolver"  # <-- change this to your solver name

    def set_objective(self, y, sigma):
        self.y = y
        self.sigma = sigma
        self.device = y.device

    def run(self, _):
        # TODO: 1. Instantiate your denoiser, e.g.: DRUNet

        # TODO: 2. Run it on the noisy image:

        raise NotImplementedError("Implement your denoiser in run()!")

    def get_result(self):
        # TODO: return your result as a dict compatible with
        # evaluate_result() in objective.py (i.e. containing 'x_hat')
