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
        self.illu = gd
        self.img = img

        def circle_point(illu, radius=0.01):
            while True:
                res_r = random.uniform(illu[0] - radius, illu[0] + radius)
                res_g = random.uniform(illu[1] - radius, illu[1] + radius)
                dis = (res_r - illu[0]) ** 2 + (res_g - illu[1]) ** 2
                if dis <= radius ** 2:
                    return np.array([res_r, res_g, 1 - res_r - res_g])

        num = len(self.illu_path)
        random_idx = np.random.randint(0, num, 1)[0]
        illu_name = self.illu_path[random_idx]
        aug_illu = np.load(illu_name)
        # awb-aug
        aug_illu = circle_point(aug_illu)

        new_img = np.dot(self.img, np.diag(aug_illu / self.illu))

        return norm_img(new_img), aug_illu

    # def crop(self, img):
    #     self.img = cv2.resize(img, (64, 64))
    #     return self.img
