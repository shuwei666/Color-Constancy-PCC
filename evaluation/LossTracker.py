import numpy as np


class LossTracker:
    """Tracker Loss and do the calculation"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count, self.max = 0, 0, 0, 0, 0
        self.loss = []

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.loss.append(self.val)
        self.count += n
        self.sum += self.val * n
        self.avg = self.sum / self.count

        self.max = np.max(np.array(self.loss))

    def get_loss(self):
        return self.avg
