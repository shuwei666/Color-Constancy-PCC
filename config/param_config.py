import argparse

# --------------------------------------------
EPOCHS = 10000
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
FOLD_NUM = 0  # should be manually changed, i.e, 0, 1, and 2
DATA_NAME = 'CC2018'  # HUAWEI, NUS-8, TAU
DEFAULT_SEED = 666
NUM_WORKERS = 12  # Depend on your CPU kernel number or GPU

DATASET_PATH = './dataset/CC2018/'  # should be manually changed to your own data path


# -------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, help='Number of epochs', default=EPOCHS)
    parser.add_argument('-batch_size', type=int, help='Batch size', default=BATCH_SIZE)
    parser.add_argument('-lr', type=float, help='Learning rate', default=LEARNING_RATE)
    parser.add_argument('-fold_num', type=int, help='Use three cross-validation. '
                                                    'The number should be changed manually'
                                                    ' from : 0, 1, 2', default=FOLD_NUM)
    parser.add_argument('-data_name', type=str, help='For better Visualization'
                                                     'when changing different datasets', default=DATA_NAME)
    parser.add_argument('-seed', type=int, help='Default seed', default=DEFAULT_SEED)
    parser.add_argument('-num_workers', type=int, help='Influence the speed of dataset loader', default=NUM_WORKERS)
    parser.add_argument('-data_path', type=str, help='dataset path', default=DATASET_PATH)
    # parser.add_argument('-train_mode', type=bool, help='Decide train or val/test mode', default=TRAIN_MODE)
    parser.add_argument('--output', action='store_true', default=True, help="shows output")

    args = parser.parse_args()

    if args.output:
        print(f'dataset path: {args.data_path}')
        print(f'num_workers: {args.num_workers}')
        print(f'batch_size: {args.batch_size}')
        print(f'epochs : {args.epochs}')
        print(f'learning rate : {args.lr}')
        print(f'manual_seed: {args.seed}')
        print(f'fold number: {args.fold_num}')

    return args
