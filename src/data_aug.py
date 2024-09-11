import random
import numpy as np
from src.utils import norm_img


class AwbAug:
    def __init__(self, illu_path):
        """
        Data Augmentation method, i.e., AWB-Aug.

        A series of illuminants are selected from all images in the datasets as centers of circles.
        Then the data augmentation was performed by randomly assigning the RGB values with the
        chromaticity distance to the selected illuminants smaller than 0.01,
        i.e., I_aug = I_origin * np.diag(illu_aug / illu_gd)
        """

        self.img = None
        self.illu = None
        self.illu_path = illu_path

    def awb_aug(self, gd, img):
        def circle_point(illu, radius=0.01):
            while True:
                res_r = random.uniform(max(0, illu[0] - radius), min(0.999999, illu[0] + radius))
                res_g = random.uniform(max(0, illu[1] - radius), min(0.999999 - res_r, illu[1] + radius))
                if (res_r - illu[0])**2 + (res_g - illu[1])**2 <= radius**2:
                    return np.array([res_r, res_g, 1 - res_r - res_g])

        aug_illu = np.load(random.choice(self.illu_path))
        aug_illu = circle_point(aug_illu)
        new_img = np.dot(img, np.diag(aug_illu / gd))
        return norm_img(new_img), aug_illu

    # def crop(self, img):
    #     self.img = cv2.resize(img, (64, 64))
    #     return self.img
