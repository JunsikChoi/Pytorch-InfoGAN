import torch
import numpy as np
from visdom import Visdom


class NLL_gaussian:
    def __call__(self, x, mu, sigma):
        '''
        Compute negative log-likelihood for Gaussian distribution
        '''
        l = (x - mu) ** 2
        l /= (2 * sigma ** 2)
        l += 0.5 * torch.log(sigma ** 2) + 0.5 * np.log(2 * np.pi)
        return l


class Logger(object):
    def __init__(self):
        self.log_target = {}

    def write(self, split_name, value):
        self.log_target[split_name]['value'].append(value)
        return

    def create_target(self, name, split_name, caption):
        self.log_target[split_name] = {'caption': caption,
                                       'name': name, 'value': []}
        return

    def clear_data(self):
        for target in self.log_target.keys():
            self.log_target[target]['value'] = []

    def pour_to_plotter(self, plotter):
        for target in self.log_target.keys():
            if target != 's':
                plotter.plot_line(var_name=self.log_target[target]['name'], split_name=target, title_name=self.log_target[
                    target]['caption'], x=self.log_target['s']['value'], y=self.log_target[target]['value'])
        return


class VisdomPlotter(object):
    """Plots to Visdom"""

    def __init__(self, config):
        self.viz = Visdom(server=config.visdom_server)
        self.config = config
        self.env = config.model_name
        # self.env = 'test'
        self.plots = {}
        self._show_configurations()

    def _show_configurations(self):
        config_dict = vars(self.config)
        config_text = "<strong>Model Configurations</strong> </br>"
        for key in config_dict.keys():
            config_text = config_text + f'{key}: {config_dict[key]} </br>'
        self.viz.text(
            text=config_text, env=self.env)

    def plot_line(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array(x), Y=np.array(y), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='iterations',
                ylabel=var_name
            ))
            return
        else:
            self.viz.line(X=np.array(x), Y=np.array(
                y), env=self.env, win=self.plots[var_name], name=split_name, update='append')
            return

    def plot_image(self, var_name, img, caption):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.image(
                img=img, env=self.env, opts=dict(caption=caption, title=caption))
            return
        else:
            self.viz.image(
                img=img, win=self.plots[var_name], env=self.env, opts=dict(caption=caption, title=caption))
            return

    def plot_image_grid(self, var_name, imgs, caption):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(
                imgs, 10, 1, env=self.env, opts=dict(caption=caption, title=caption, xlabel='Continuous Code', ylabel='Discrete Code'))
            return
        else:
            self.viz.images(
                imgs, 10, 1, win=self.plots[var_name], env=self.env, opts=dict(caption=caption, title=caption, xlabel='Continuous Code', ylabel='Discrete Code'))
            return


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
