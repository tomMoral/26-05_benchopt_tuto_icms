"""Participant skeleton for the Blind Deblurring objective.

Your task: implement the three methods marked TODO below.

What is given to you (do NOT modify):
  - set_data()       — stores the ground-truth image, observation, and physics
  - get_one_result() — trivial baseline (delta kernel, return y as x_hat)

What you must implement:
  - get_objective()   — expose what solvers need (see hint below)
  - evaluate_result() — compute two metrics: image PSNR and kernel MSE
  - save_final_results() — persist both x_hat and k_hat between callbacks
"""

from benchopt import BaseObjective
from benchopt.stopping_criterion import SufficientProgressCriterion

import torch
import deepinv as dinv  # noqa: F401


class Objective(BaseObjective):

    name = "Blind Deblurring"
    url = "https://github.com/tommoral/26-05_benchopt_tuto_icms/02-blind_deblur"
    requirements = ["pip::deepinv"]
    min_benchopt_version = "1.8"

    stopping_criterion = SufficientProgressCriterion(
        strategy='callback', patience=10,
        key_to_monitor='psnr', minimize=False,
    )

    # -----------------------------------------------------------------
    # Given — do not modify
    # -----------------------------------------------------------------

    def set_data(self, x_true, y, physics, kernel_size):
        self.x_true = x_true
        self.physics = physics
        self.y = y
        self.kernel_size = kernel_size

    # -----------------------------------------------------------------
    # TODO — implement the three methods below
    # -----------------------------------------------------------------

    def get_objective(self):
        """Return a dict that will be unpacked as keyword arguments to
        each solver's set_objective().

        Hint: solvers need the blurred observation `y` and `kernel_size`
        to initialise their kernel estimate.  Do NOT expose the true kernel
        — that would defeat the purpose of blind deblurring!
        """
        return dict(y=self.y, kernel_size=self.kernel_size)

    def save_final_results(self, x_hat, k_hat):
        """Persist the image and kernel estimates between callbacks.

        Hint: return a dict with 'x_hat', 'k_hat', and 'k_true' moved to CPU.
        'k_true' is accessible as self.physics.filter.
        This is called after the run so the results can be saved to disk.
        """
        return dict(
            x_hat=x_hat.cpu().numpy(),
            x_true=self.x_true.cpu().numpy(),
            k_hat=k_hat.cpu().numpy(),
            k_true=self.physics.filter.cpu().numpy(),
        )

    def evaluate_result(self, x_hat, k_hat):
        """Compute image quality and kernel accuracy.

        Hint:
          - Image metric  : PSNR between x_hat and self.x_true
              → dinv.metric.PSNR()(x_hat, x_true)
          - Kernel metric : MSE between normalised kernels
              → normalise each kernel so that its values sum to 1
                before computing the squared difference
          - Move tensors to the same device before computing.

        Must return a dict with at least the key used in stopping_criterion
        (here: 'psnr').
        """
        psnr = dinv.metric.PSNR()(x_hat, self.x_true.to(x_hat.device)).item()
        k_hat_norm = k_hat / k_hat.sum()
        k_true_norm = self.physics.filter.to(k_hat.device) / self.physics.filter.sum()
        mse_kernel = torch.mean((k_hat_norm - k_true_norm) ** 2).item()
        return dict(psnr=psnr, mse_kernel=mse_kernel)

    # -----------------------------------------------------------------
    # Given — do not modify
    # -----------------------------------------------------------------

    def get_one_result(self):
        """Trivial baseline: identity kernel, return the blurred image."""
        k_delta = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        k_delta[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        return dict(x_hat=self.y.cpu(), k_hat=k_delta)
