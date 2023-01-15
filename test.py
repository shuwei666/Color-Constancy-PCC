from src.utils import print_single_metric
import torch
from src.ModelOperator import ModelOperator
from src.dataset import CcData
from torch.utils.data import DataLoader
from evaluation.Evaluator import Evaluator
from config.settings import DEVICE

# ------------------
TRAIN_MODEL = False
DATA_DIR = './dataset/CC2018/'
BATCH_SIZE = 1
NUM_WORKERS = 12
# ------------------


def main():

    evaluator = Evaluator()
    model = ModelOperator()

    for num_fold in range(3):
        fold_evaluator = Evaluator()
        data_test = CcData(DATA_DIR, train=TRAIN_MODEL, fold_num=num_fold)
        test_loader = DataLoader(data_test, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        path_to_pretrained = './pretrain_models/' + 'fold' + str(num_fold) + '/model_cc_b1.pth'
        model.load_model(path_to_pretrained)
        model.evaluation_mode()
        print('=========Testing!========')
        with torch.no_grad():
            for img, illu in test_loader:
                img, illu = img.to(DEVICE), illu.to(DEVICE)
                pred = model.predict(img)
                loss = model.get_loss(pred, illu).item()
                fold_evaluator.add_error(loss)
                evaluator.add_error(loss)

        metrics = fold_evaluator.compute_metrics()
        print(f'---The fold_{num_fold} error---')
        print_single_metric(metrics)
    print('*****************')
    metrics = evaluator.compute_metrics()
    print('\t\t---Total error---')
    print_single_metric(metrics)


if __name__ == '__main__':

    main()
