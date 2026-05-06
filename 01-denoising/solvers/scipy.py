from benchopt import BaseSolver

import torch


class Solver(BaseSolver):
    """Denoising with DrUNet (pretrained deep network, non-iterative)."""

    name = "scipy"

    def set_objective(self, y, sigma):
        self.y = y
        self.sigma = sigma

    def run(self, _):
        from scipy.ndimage import median_filter
        self.x_hat = median_filter(self.y.cpu().numpy(), size=3)

    def get_result(self):
        return dict(x_hat=torch.tensor(self.x_hat))
