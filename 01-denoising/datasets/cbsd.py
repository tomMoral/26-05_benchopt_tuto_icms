from benchopt import BaseDataset

import torch
import deepinv as dinv


class Dataset(BaseDataset):

    name = "CBSD"

    parameters = {
        'sigma': [0.05, 0.1, 0.2],
        'img_name': [
            'CBSD_0010.png',
        ],
    }

    def get_data(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_true = dinv.utils.load_example(
            self.img_name,
            grayscale=False,
            device=device,
        )

        return dict(x_true=x_true, sigma=self.sigma)
