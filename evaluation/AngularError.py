from evaluation.Loss import Loss
import torch
import math
from torch.nn.functional import normalize
from config.settings import DEVICE


class AngularError(Loss):
    """ The angular error, which used as loss and error function"""
    def __init__(self, device):
        super().__init__(device)

    def _compute(self, pred, label, safe_v=0.999999):
        # Based on r and g chromaticity, calculating the b chromaticity

        chrom_b = (1 - pred.sum(axis=1)).unsqueeze(1)
        pred = torch.cat((pred, chrom_b), dim=1)

        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)

        return torch.mean(angle).to(DEVICE)
