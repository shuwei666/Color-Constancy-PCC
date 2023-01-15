
from config.settings import DEVICE
from src.ModelBase import ModelBase
from src.PCC import JosaPcc


class ModelOperator(ModelBase):
    """ Abstract of predict and optimize of DNN model, which inherited from ModelBase"""
    def __init__(self):
        super().__init__()
        self._network = JosaPcc().to(DEVICE)

    def predict(self, img):

        return self._network(img)

    def optimize(self, img, label):
        self._optimizer.zero_grad()
        pred = self.predict(img)

        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()

        return loss.item()


