import torch
import numpy as np


class NLL_gaussian:
    def __call__(self, x, mu, sigma):
        '''
        Compute negative log-likelihood for Gaussian distribution
        '''
        l = (x - mu) ** 2
        l /= (2 * sigma ** 2)
        l += 0.5 * torch.log(sigma ** 2) + 0.5 * np.log(2 * np.pi)
        return l


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
