
from torch import Tensor
import numpy as np
import pandas as pd
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

EPS = 1e-7


def norm_img(img):
    """ normalized image values to [0., 1.]"""
    img /= (img.max() + EPS)
    img = img.clip(0., 1.)
    return img


def hwc_to_chw(img: np.ndarray):
    """ Converts an image from height * width * channel to (channel * height * width)"""
    return img.transpose(2, 0, 1)


def chw_to_hwx(x: Tensor) -> Tensor:
    """ Converts a Tensor to an Image """
    img = x.cpu().numpy()
    img = img.transpose(0, 2, 3, 1)[0, :, :, :]
    return img


def print_metrics(current_metrics: dict, best_metrics: dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst95"], best_metrics["wst95"]))
    print(" Best ......... : {:.4f} (Best: {:.4f})".format(current_metrics["bst"], best_metrics["bst"]))


def print_single_metric(current_metrics):
    print(" Mean ......... : {:.4f}".format(current_metrics["mean"]))
    print(" Median ....... : {:.4f}".format(current_metrics["median"]))
    print(" Trimean ...... : {:.4f}".format(current_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f}".format(current_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f}".format(current_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f}".format(current_metrics["wst95"]))
    print(" Best ......... : {:.4f}".format(current_metrics["bst"]))


def save_log(best_stats, current_loss, training_loss, val_loss, path_to_log):
    log_data = pd.DataFrame({
        'Tr-loss': [training_loss],
        'Val-loss': [val_loss],
        'b-mean': best_stats['mean'],
        'b-median': best_stats['median'],
        'b-tri-mean': best_stats['trimean'],
        'b-b25': best_stats['bst25'],
        'b-wst25': best_stats['wst25'],
        'b-wst': best_stats['wst95'],
        'b-bst': best_stats['bst'],
        **{k: [v] for k, v in current_loss.items()}
    })
    head = log_data.keys() if not os.path.exists(path_to_log) else False
    log_data.to_csv(path_to_log, mode='a', header=head, index=False)


def log_sys(args):
    dt = datetime.now()
    path_to_log = os.path.join('./log', args.data_name,
                               f'fold_{args.fold_num}_'
                               f'-{dt.day}-{dt.hour}-{dt.minute}')

    os.makedirs(path_to_log, exist_ok=True)
    path_to_metrics_log = os.path.join(path_to_log, 'error.csv')
    vis_log_tr = os.path.join(f'./vis_log', f'{dt.day}-{dt.hour}-{dt.minute}', 'train')
    vis_log_acc = os.path.join(f'./vis_log', f'{dt.day}-{dt.hour}-{dt.minute}', 'acc')
    os.makedirs(vis_log_tr, exist_ok=True)
    os.makedirs(vis_log_acc, exist_ok=True)

    param_info = {'lr': args.lr, 'batch_size': args.batch_size,
                  'fold_num': args.fold_num, 'data_name': args.data_name,
                  'time_file': f'{dt.day}-{dt.hour}-{dt.minute}',
                  'seed': f'{args.seed}'}

    return SummaryWriter(vis_log_tr), SummaryWriter(vis_log_acc), \
           path_to_log, path_to_metrics_log, param_info


def k_fold(n_splits=3, num=0):
    """
    Randomly Split the training and testing datasets.
    """
    assert n_splits is 3, "three-cross validation"
    num = np.arange(num)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=666)
    tr, te = [], []
    train_test = {}
    for train, test in kf.split(num):
        tr.append((train.tolist()))
        te.append((test.tolist()))

    train_test['train'] = tr
    train_test['test'] = te

    return train_test
