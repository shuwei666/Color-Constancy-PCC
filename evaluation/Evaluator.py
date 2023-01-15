import numpy as np


class Evaluator:

    def __init__(self):
        monitored_metrics = ["mean", "median", "trimean", "bst25", "wst25", "wst95", 'bst']
        self.__errors = []
        self.__metrics = {}
        self.__best_metrics = {m: 100.0 for m in monitored_metrics}

    def get_best_metrics(self):
        return self.__best_metrics

    def add_error(self, error):
        tmp = self.__errors.append(error)
        return self

    def reset_errors(self):
        self.__errors = []

    def get_errors(self):
        return self.__errors

    def compute_metrics(self):
        self.__errors = sorted(self.__errors)
        # self.__errors = self.__errors[~np.isnan(self.__errors)]
        self.__metrics = {
            'mean': np.mean(self.__errors),
            'median': np.median(self.__errors),
            'trimean': 0.25 * self.__g(0.25) + 0.5 * self.__g(0.5) + 0.25 * self.__g(0.75),
            'bst25': np.mean(self.__errors[:int(len(self.__errors) * 0.25)]),
            'wst25': np.mean(self.__errors[int(len(self.__errors) * 0.75):]),
            'wst95': np.mean(self.__errors[int(len(self.__errors) * 0.95):]),
            'bst': np.min(self.__errors)
        }

        return self.__metrics

    def update_best_metrics(self) -> dict:
        self.__best_metrics["mean"] = self.__metrics["mean"]
        self.__best_metrics["median"] = self.__metrics["median"]
        self.__best_metrics["trimean"] = self.__metrics["trimean"]
        self.__best_metrics["bst25"] = self.__metrics["bst25"]
        self.__best_metrics["wst25"] = self.__metrics["wst25"]
        self.__best_metrics["wst95"] = self.__metrics["wst95"]
        self.__best_metrics["bst"] = self.__metrics["bst"]
        return self.__best_metrics

    def __g(self, f):
        return np.percentile(self.__errors, f * 100)
