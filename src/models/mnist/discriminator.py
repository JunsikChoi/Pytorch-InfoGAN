import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Discriminator Model Definition
'''


class Discriminator(nn.Module):
    '''Shared Part of Discriminator and Recognition Model'''

    def __init__(self, n_c_disc, dim_c_disc, dim_c_cont):
        super(Discriminator, self).__init__()
        self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont
        self.n_c_disc = n_c_disc
        # Shared layers
        self.module_shared = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Reshape(-1, 128*7*7),
            nn.Linear(in_features=128*7*7, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Layer for Disciminating
        self.module_D = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

        self.module_Q = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.latent_disc = nn.Sequential(
            nn.Linear(
                in_features=128, out_features=self.n_c_disc*self.dim_c_disc),
            Reshape(-1, self.n_c_disc, self.dim_c_disc),
            nn.Softmax(dim=2)
        )

        self.latent_cont_mu = nn.Linear(
            in_features=128, out_features=self.dim_c_cont)

        self.latent_cont_var = nn.Linear(
            in_features=128, out_features=self.dim_c_cont)

    def forward(self, z):
        out = self.module_shared(z)
        probability = self.module_D(out)
        probability = probability.squeeze()
        internal_Q = self.module_Q(out)
        c_disc_logits = self.latent_disc(internal_Q)
        c_cont_mu = self.latent_cont_mu(internal_Q)
        c_cont_var = torch.exp(self.latent_cont_var(internal_Q))
        return probability, c_disc_logits, c_cont_mu, c_cont_var


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
