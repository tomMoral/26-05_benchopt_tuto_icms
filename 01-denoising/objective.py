from benchopt import BaseObjective

import torch
import deepinv as dinv


class Objective(BaseObjective):

    name = "Denoising"
    url = "https://github.com/tommoral/26-05_benchopt_tuto_icms/01-denoising"

    requirements = ["pip::deepinv"]
    min_benchopt_version = "1.9"

    sampling_strategy = 'run_once'

    def set_data(self, x_true, sigma):
        self.x_true = x_true
        self.sigma = sigma

        # Change the noise seed if repeating with the same image/solver
        # but not the same repetition.
        seed = self.get_seed(use_repetition=True)

        # Add Gaussian noise with a fixed seed for reproducibility
        rng = torch.Generator(device=x_true.device)
        rng.manual_seed(seed)
        self.y = x_true + sigma * torch.randn(
            x_true.shape, generator=rng, device=x_true.device
        )

    def get_objective(self):
        return dict(y=self.y, sigma=self.sigma)

    def evaluate_result(self, x_hat):
        x_true = self.x_true.to(x_hat.device)
        psnr = dinv.metric.PSNR()(x_hat, x_true).item()
        ssim = dinv.metric.SSIM()(x_hat, x_true).item()
        return dict(
            psnr=psnr,
            ssim=ssim,
            x_hat=x_hat.cpu().numpy(),
            x_true=self.x_true.cpu().numpy(),
            y=self.y.cpu().numpy(),
        )

    def get_one_result(self):
        # Trivial baseline: return the noisy image as-is
        return dict(x_hat=self.y)
