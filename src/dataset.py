
from torch.utils.data import Dataset
import glob
from src.utils import hwc_to_chw, k_fold, EPS
import numpy as np
from pathlib import Path
from src.data_aug import AwbAug


FULL_TEST = False


class CcData(Dataset):
    def __init__(self, path, train=True, fold_num=0):
        self.path = path
        self.train = train
        self.illu_full = glob.glob(path + 'numpy_labels' + '/*.npy')
        self.img_full = glob.glob(path + 'numpy_data' + '/*.npy')
        self.img_full.sort(key=lambda x: x.split('\\')[-1].split('_')[-1].split('.')[0])
        self.illu_full.sort(key=lambda x: x.split('\\')[-1].split('_')[-1].split('.')[0])

        train_test = k_fold(n_splits=3, num=len(self.img_full))
        img_idx = train_test['train' if self.train else 'test'][fold_num]

        self.fold_data = [self.img_full[i] for i in img_idx]
        self.fold_illu = [self.illu_full[i] for i in img_idx]
        self.data_aug = AwbAug(self.illu_full)
        # if FULL_TEST:
        #     self.fold_data = self.img_full

    def __len__(self):
        return len(self.fold_data)

    def feature_select(self, img_tmp, thresh_dark=0.02, thresh_saturation=0.98):
        """
        The four feature selected, i.e., bright, max, mean and dark pixels
        """
        img_tmp = img_tmp.reshape(-1, 3)
        img_tmp = img_tmp[np.all(img_tmp > thresh_dark, axis=1), :]
        img_tmp = img_tmp[np.all(img_tmp < thresh_saturation, axis=1), :]
        # 0. Brightest pixel
        bright_v = img_tmp[np.argmax(img_tmp.sum(axis=1))]
        # 1. Maximum pixel
        max_wp = img_tmp.max(axis=0)
        # 2. Average pixel
        mean_v = img_tmp.mean(axis=0)
        # 3. Darkest pixel
        dark_v = img_tmp[np.argmin(img_tmp.sum(axis=1))]
        # ---Testing the weight of different features---
        # mask_feature = np.array([0, 0, 0])
        # dark_v = mask_feature
        feature_data = np.vstack([bright_v, max_wp, mean_v, dark_v])
        # feature_num = len(feature_data)
        feature_data /= (feature_data.sum(axis=1).reshape(-1, 1) + EPS)
        feature_data = feature_data[:, :2]

        return feature_data

    def __getitem__(self, idx):
        """ Gets next data in the dataloader.

        Note: We pre-processed the input data in the format of '.npy' for fast processing. If
        you want to train your own dataset, the corresponding of loadig image should also be changed.

        """

        # img_name = self.fold_data[idx].split('/')[-1].split('.')[0]
        img_data = np.load(self.fold_data[idx])
        gd_data = np.load(self.fold_illu[idx])
        if self.train:
            img_data, gd_data = self.data_aug.awb_aug(gd_data, img_data)

        feature_data = self.feature_select(img_data)
        del img_data
        return feature_data.astype(np.float32), gd_data.astype(np.float32)
