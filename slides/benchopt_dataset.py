from benchopt import BaseDataset

import deepinv as dinv


class Dataset(BaseDataset):

    name = "CBSD"
    parameters = {'sigma': [0.05, 0.1, 0.2]}

    def get_data(self):
        x_true = dinv.utils.load_example(
            'CBSD_0010.png', grayscale=False,
        )
        return dict(x_true=x_true, sigma=self.sigma)
