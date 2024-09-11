import torch
import numpy as np
import random

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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

