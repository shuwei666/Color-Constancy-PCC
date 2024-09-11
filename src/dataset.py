from torch.utils.data import Dataset
import glob
from src.utils import k_fold, EPS
import numpy as np
from pathlib import Path
from src.data_aug import AwbAug


FULL_TEST = False


class CcData(Dataset):
    def __init__(self, path, train=True, fold_num=0):
        self.train = train
        self.illu_full = sorted(glob.glob(f'{path}numpy_labels/*.npy'))
        self.img_full = sorted(glob.glob(f'{path}numpy_data/*.npy'))

        train_test = k_fold(n_splits=3, num=len(self.img_full))
        img_idx = train_test['train' if self.train else 'test'][fold_num]

        self.fold_data = [self.img_full[i] for i in img_idx]
        self.fold_illu = [self.illu_full[i] for i in img_idx]
        self.data_aug = AwbAug(self.illu_full)

    def __len__(self):
        return len(self.fold_data)

    def feature_select(self, img_tmp, thresh_dark=0.02, thresh_saturation=0.98):
        img_tmp = img_tmp.reshape(-1, 3)
        mask = np.all((img_tmp > thresh_dark) & (img_tmp < thresh_saturation), axis=1)
        
        if not np.any(mask):
            feature_data = np.tile(img_tmp.mean(axis=0), (4, 1))
        else:
            img_filtered = img_tmp[mask]
            bright_v = img_filtered[np.argmax(img_filtered.sum(axis=1))]
            max_wp = img_filtered.max(axis=0)
            mean_v = img_filtered.mean(axis=0)
            dark_v = img_filtered[np.argmin(img_filtered.sum(axis=1))]
            feature_data = np.vstack([bright_v, max_wp, mean_v, dark_v])

        feature_data /= (feature_data.sum(axis=1, keepdims=True) + EPS)
        return feature_data[:, :2]

    def __getitem__(self, idx):
        img_data = np.load(self.fold_data[idx])
        gd_data = np.load(self.fold_illu[idx])
        
        if self.train:
            img_data, gd_data = self.data_aug.awb_aug(gd_data, img_data)

        feature_data = self.feature_select(img_data)
        return feature_data.astype(np.float32), gd_data.astype(np.float32)
