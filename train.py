
import time
import torch
import os

from torch.utils.data import DataLoader
from src.utils import *
from src.dataset import CcData
from src.ModelOperator import ModelOperator
from evaluation.Evaluator import Evaluator
from evaluation.LossTracker import LossTracker
from config.settings import DEVICE, set_seed
from config.param_config import parse_args

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(args):
    # Logging
    write_tr, write_val, path_to_log, path_to_metrics_log, param_info = log_sys(args)
    # Model instance
    model = ModelOperator()
    model.set_optimizer(learning_rate=args.lr)
    model.log_network(path_to_log, param_info)
    # Dataset loader
    data_dir = args.data_path
    data_train = CcData(data_dir, train=True, fold_num=args.fold_num)
    train_loader = DataLoader(data_train, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=False)
    print('============================')
    print(fr'Training set size ... {len(data_train)}')

    data_val = CcData(data_dir, train=False, fold_num=args.fold_num)
    test_loader = DataLoader(data_val, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, drop_last=False)

    print(fr'Validation set size ... {len(data_val)}')

    # Model training
    train_loss, val_loss = LossTracker(), LossTracker()
    evaluator = Evaluator()
    best_val_error, best_metrics = 100.0, evaluator.get_best_metrics()
    model.train_mode()

    scheduler = model.lr_scheduler(args.epochs)

    for epoch in range(args.epochs):
        train_loss.reset()
        for idx, data in enumerate(train_loader):
            img, illu = data
            img, illu = img.to(DEVICE), illu.to(DEVICE)
            loss = model.optimize(img, illu)
            train_loss.update(loss)

        scheduler.step()

        val_loss.reset()
        start = time.time()
        # Model eval
        if epoch % 2 == 0:

            model.evaluation_mode()
            evaluator.reset_errors()
            print('=========Validation!========')
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    img, illu = data
                    img, illu = img.to(DEVICE), illu.to(DEVICE)
                    pred = model.predict(img)
                    loss = model.get_loss(pred, illu).item()
                    val_loss.update(loss)
                    evaluator.add_error(loss)

        print(f'Epoch: {epoch}/{args.epochs}')
        print(f'Training Mean Loss:{train_loss.avg:.2f}')
        val_time = time.time() - start
        loss_metrics = evaluator.compute_metrics()

        if val_time > 0.1:
            print('\n*************************************')
            print_metrics(loss_metrics, best_metrics)
            print('*************************************\n')
            # write_tr.add_scalar('loss', train_loss.avg, epoch)
            # write_val.add_scalar('loss', val_loss.avg, epoch)
            # write_val.close()
            # write_tr.close()
        if 0 < val_loss.avg < best_val_error:
            best_val_error = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print('Saving new best model...\n')
            model.save(path_to_log)

        save_log(best_metrics, loss_metrics, train_loss.avg, val_loss.avg, path_to_metrics_log)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    main(args)
