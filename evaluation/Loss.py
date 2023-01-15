

class Loss:

    def __init__(self, device):
        self._device = device

    def _compute(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self._compute(*args).to(self._device)
