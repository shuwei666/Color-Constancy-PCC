import torch
import numpy as np

DEVICE_TYPE = 'cuda:0'


def get_device():
    if DEVICE_TYPE == 'cpu':
        print('\n Running on device "cpu" \n')
        return torch.device('cpu')
    else:
        print(f'Running on device {DEVICE_TYPE}', '\n')
        return torch.device(DEVICE_TYPE)


DEVICE = get_device()

# -----Define random seed for reproduction ---


def set_seed(seed=666):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False

