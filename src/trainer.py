import os
import torch
import numpy as np
import time
import datetime
import itertools
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils import *
from models.mnist.discriminator import Discriminator
from models.mnist.generator import Generator
torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, config, data_loader):
        self.n_c_disc = config.n_c_disc
        self.dim_c_disc = config.dim_c_disc
        self.dim_c_cont = config.dim_c_cont
        self.dim_z = config.dim_z
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.lr_G = config.lr_G
        self.lr_D = config.lr_D
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.gpu_id = config.gpu_id
        self.data_loader = data_loader
        self.num_epoch = config.num_epoch
        self.lambda_disc = config.lambda_disc
        self.lambda_cont = config.lambda_cont
        self.log_step = config.log_step
        self.project_root = config.project_root
        self.model_name = config.model_name
        self._set_device(self.gpu_id)
        self.build_models()

    def _set_device(self, gpu_id):
        self.device = torch.device(gpu_id)

    def _sample(self):
        # Sample Z from N(0,1)
        z = torch.randn(self.batch_size, self.dim_z, device=self.device)

        # Sample discrete latent code from Cat(K=dim_c_disc)
        idx = np.zeros((self.n_c_disc, self.batch_size))
        c_disc = torch.zeros(self.batch_size, self.n_c_disc,
                             self.dim_c_disc, device=self.device)
        for i in range(self.n_c_disc):
            idx[i] = np.random.randint(self.dim_c_disc, size=self.batch_size)
            c_disc[torch.arange(0, self.batch_size), i, idx[i]] = 1.0

        # Sample continuous latent code from Unif(-1,1)
        c_cond = torch.rand(self.batch_size, self.dim_c_cont,
                            device=self.device) * 2 - 1

        # Concat z, c_disc, c_cond
        for i in range(self.n_c_disc):
            z = torch.cat((z, c_disc[:, i, :].squeeze()), dim=1)
        z = torch.cat((z, c_cond), dim=1)

        return z, idx

    def _sample_fixed_noise(self):
        # Sample Z from N(0,1)
        fixed_z = torch.randn(self.dim_c_disc*10, self.dim_z)

        # For each discrete variable, fix other discrete variable to 0 and random sample other variables.
        idx = np.arange(self.dim_c_disc).repeat(10)

        c_disc_list = []
        for i in range(self.n_c_disc):
            zero_template = torch.zeros(
                self.dim_c_disc*10, self.n_c_disc, self.dim_c_disc)
            # Set all the other discrete variable to Zero([1,0,...,0])
            for ii in range(self.n_c_disc):
                if (i == ii):
                    pass
                else:
                    zero_template[:, ii, 0] = 1.0
            for j in range(len(idx)):
                zero_template[np.arange(self.dim_c_disc*10), i, idx] = 1.0
            c_disc_list.append(zero_template)

        # Random sample continuous variables
        c_rand = torch.rand(self.dim_c_disc * 10, self.dim_c_cont) * 2 - 1
        c_range = torch.linspace(start=-1, end=1, steps=10)

        c_range_list = []
        for i in range(self.dim_c_disc):
            c_range_list.append(c_range)
        c_range = torch.cat(c_range_list, dim=0)

        c_cont_list = []
        for i in range(self.dim_c_cont):
            c_zero = torch.zeros(self.dim_c_disc * 10, self.dim_c_cont)
            c_zero[:, i] = c_range
            c_cont_list.append(c_zero)

        fixed_z_dict = {}
        for idx_c_disc in range(len(c_disc_list)):
            for idx_c_cont in range(len(c_cont_list)):
                z = fixed_z.clone()
                for j in range(self.n_c_disc):
                    z = torch.cat(
                        (z, c_disc_list[idx_c_disc][:, j, :].squeeze()), dim=1)
                z = torch.cat((z, c_cont_list[idx_c_cont]), dim=1)
                fixed_z_dict[(idx_c_disc, idx_c_cont)] = z
        return fixed_z_dict

    def build_models(self):
        # Initiate Models
        self.G = Generator(self.dim_z, self.n_c_disc, self.dim_c_disc,
                           self.dim_c_cont).to(self.device)
        self.D = Discriminator(self.n_c_disc, self.dim_c_disc,
                               self.dim_c_cont).to(self.device)

        # Initialize
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)

        return

    def set_optimizer(self, param_list, lr):
        params_to_optimize = itertools.chain(*param_list)
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params_to_optimize, lr=lr, betas=(self.beta1, self.beta2))
            return optimizer
        else:
            raise NotImplementedError

    def _generate_data(self, z, epoch, idx_c_d, idx_c_c):
        with torch.no_grad():
            gen_data = self.G(z).detach().cpu()
        title = f'Fixed_{self.model_name}_E-{epoch}_Cd-{idx_c_d}_Cc-{idx_c_c}'
        plt.figure(figsize=(10, 10))
        plt.title(title, fontsize=25)
        # plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f'Continuous Code Index = {idx_c_c}', fontsize=20)
        plt.ylabel(f'Discrete Code Index = {idx_c_d}', fontsize=20)
        plt.imshow(np.transpose(vutils.make_grid(
            gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        result_dir = os.path.join(
            self.project_root, 'results', self.model_name)
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(os.path.join(result_dir, title+'.png'))
        plt.close('all')
        return

    def _generate_data2(self, samples, epoch, idx):
            # with torch.no_grad():
                # gen_data = self.G(z).detach().cpu()
        gen_data = samples.detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(
            gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        result_dir = os.path.join(
            self.project_root, 'results', self.model_name)
        os.makedirs(result_dir, exist_ok=True)
        title = f'{self.model_name}-Epoch_{epoch}-C_disc_{idx}.png'
        plt.savefig(os.path.join(result_dir, title))
        plt.close('all')
        return

    def train(self):
        # Set opitmizers
        optim_G = self.set_optimizer([self.G.parameters(), self.D.module_Q.parameters(
        ), self.D.latent_disc.parameters(), self.D.latent_cont_mu.parameters()], lr=self.lr_G)
        # , self.D.latent_cont_var.parameters()
        optim_D = self.set_optimizer(
            [self.D.module_shared.parameters(), self.D.module_D.parameters()], lr=self.lr_D)
        # optim_G = self.set_optimizer(
        #     [self.G.parameters(), self.D.parameters()])
        # optim_D = self.set_optimizer([self.D.parameters()])
        # optim_Info = self.set_optimizer(
        #     [self.G.parameters(), self.D.parameters()])

        # Loss functions
        adversarial_loss = torch.nn.BCELoss()
        categorical_loss = torch.nn.CrossEntropyLoss()
        # continuous_loss = NLL_gaussian()
        continuous_loss = torch.nn.MSELoss(reduction='none')
        # Sample fixed latent codes for comparison
        fixed_z_dict = self._sample_fixed_noise()

        start_time = time.time()
        num_steps = len(self.data_loader)
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            step = 0
            for i, (data, _) in enumerate(self.data_loader, 0):
                if (data.size()[0] != self.batch_size):
                    self.batch_size = data.size()[0]

                data_real = data.to(self.device)

                # Update Discriminator
                # Reset optimizer
                optim_D.zero_grad()
                # Calculate Loss D(real)

                prob_real, _, _, _ = self.D(data_real)
                label_real = torch.full(
                    (self.batch_size,), 1, device=self.device)
                loss_D_real = adversarial_loss(prob_real, label_real)

                # Calculate gradient -> grad accums to module_shared / modue_D
                loss_D_real.backward()

                # calculate Loss D(fake)
                # Sample noise, latent codes
                z, idx = self._sample()
                data_fake = self.G(z)
                prob_fake_D, _, _, _ = self.D(data_fake.detach())
                label_fake = torch.full(
                    (self.batch_size,), 0, device=self.device)
                loss_D_fake = adversarial_loss(prob_fake_D, label_fake)

                # Calculate gradient -> grad accums to module_shared / modue_D
                loss_D_fake.backward()
                loss_D = loss_D_real.item() + loss_D_fake.item()

                # Update Parameters for D
                optim_D.step()

                # Update Generator and Q
                # Reset Optimizer
                optim_G.zero_grad()

                # Calculate loss for generator
                prob_fake, disc_logits, mu, sigma = self.D(data_fake)
                loss_G = adversarial_loss(prob_fake, label_real)

                # Calculate loss for discrete latent code
                target = torch.LongTensor(idx).to(self.device)
                loss_c_disc = 0
                for j in range(self.n_c_disc):
                    loss_c_disc += categorical_loss(
                        disc_logits[:, j, :], target[j, :])
                loss_c_disc = loss_c_disc * self.lambda_disc

                # Calculate loss for continuous latent code
                # loss_c_cont = continuous_loss(
                # z[:, self.dim_z+self.n_c_disc*self.dim_c_disc:], mu, sigma).mean(0)
                loss_c_cont = continuous_loss(
                    mu, z[:, self.dim_z+self.n_c_disc*self.dim_c_disc:]).mean(dim=0)
                loss_c_cont = loss_c_cont * self.lambda_cont

                loss_info = loss_G + loss_c_disc + loss_c_cont.sum()
                # loss_info = loss_G
                # loss_info = loss_G + loss_c_disc
                loss_info.backward()
                optim_G.step()

                # Print log info
                if (step % self.log_step == 0):
                    print('==========')
                    print(f'Model Name: {self.model_name}')
                    print('Epoch [%d/%d], Step [%d/%d], Elapsed Time: %s \nLoss D : %.4f, Loss Info: %.4f\nLoss_Disc: %.4f Loss_Cont: %.4f Loss_Gen: %.4f'
                          % (epoch + 1, self.num_epoch, step, num_steps, datetime.timedelta(seconds=time.time()-start_time), loss_D, loss_info.item(), loss_c_disc.item(), loss_c_cont.sum().item(), loss_G.item()))
                    for c in range(len(loss_c_cont)):
                        print('Loss of %dth continuous latent code: %.4f' %
                              (c+1, loss_c_cont[c].item()))
                    print(
                        f'Prob_real_D:{prob_real.mean()}, Prob_fake_D:{prob_fake_D.mean()}, Prob_fake_G:{prob_fake.mean()}')
                step += 1
                if step == 1:
                    self._generate_data2(
                        data_fake[:100, ...], epoch=epoch, idx=i+1)
            for key in fixed_z_dict.keys():
                fixed_z = fixed_z_dict[key].to(self.device)
                idx_c_disc = key[0]
                idx_c_cont = key[1]
                self._generate_data(fixed_z, epoch, idx_c_disc, idx_c_cont)
        return

    def extract_grad_dict(self, m):
        param_dict = {}
        for name, param in m.named_parameters():
            if param.grad is None:
                param_dict[name] = None
            else:
                param_dict[name] = param.grad.clone()
        return param_dict

    def compare_grad(self, d1, d2):
        different_name = []
        for key in d1.keys():
            if d1[key] is None:
                if d2[key] is not None:
                    different_name.append(key)
                else:
                    pass
            else:
                if not torch.equal(d1[key], d2[key]):
                    different_name.append(key)
        if len(different_name) == 0:
            print("Same grad!")
        return different_name

    def test(self):
        pass
