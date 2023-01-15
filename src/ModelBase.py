import os
import torch
from torch.optim import lr_scheduler
from config.settings import DEVICE
from evaluation.AngularError import AngularError


class ModelBase:
    """ Basic Operations of Model"""
    def __init__(self):
        self._device = DEVICE
        self._criterion = AngularError(self._device)
        self._optimizer = None
        self._network = None

    def print_network(self):
        print('\n-------------------------\n')
        print(self._network)
        print('\n-------------------------\n')

    def get_loss(self, pred, label):
        return self._criterion(pred, label)

    def set_optimizer(self, learning_rate, optimizer_type='adam'):
        optimizer_map = {'adam': torch.optim.Adam, 'rmsprop': torch.optim.RMSprop}
        self._optimizer = optimizer_map[optimizer_type](self._network.parameters(), lr=learning_rate)

    def log_network(self, path_to_log, para):
        open(os.path.join(path_to_log, 'net_param.txt'), 'a+').write(str(self._network) + '\n'
                                                                     + str(para))

    def evaluation_mode(self):
        self._network = self._network.eval()

    def train_mode(self):
        self._network = self._network.train()

    def save(self, path_to_log):
        torch.save(self._network.state_dict(), os.path.join(path_to_log, 'model_cc_b1.pth'))

    def load_model(self, model_path):
        self._network.load_state_dict(torch.load(model_path))

    def lr_scheduler(self, maxnum):
        return lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=maxnum)

    def fine_tune(self, path):
        pretrained_dict = torch.load(path)
        net_state_dict = self._network.state_dict()
        pretrained_dict_1 = {k: v for k, v in
                             pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_1)
        self._network.load_state_dict(net_state_dict)
