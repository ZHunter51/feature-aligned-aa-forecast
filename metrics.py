__all__ = ["mae", "mse", "rmse", "mape", "smape", "mase"]

from typing import Union

import numpy as np


def safe_div(
    a: Union[np.ndarray, np.floating], b: Union[np.ndarray, np.floating]
) -> Union[np.ndarray, np.floating]:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def mse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.square(true - pred)))


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(true - pred))))


def mape(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(safe_div(true - pred, true))))


def smape(true: np.ndarray, pred: np.ndarray) -> float:
    return float(
        np.mean(safe_div(np.abs(true - pred), (np.abs(true) + np.abs(pred)) / 2))
    )


def mase(true: np.ndarray, pred: np.ndarray) -> float:
    return float(
        safe_div(np.mean(np.abs(true - pred)), np.mean(np.abs(true[1:] - true[:-1])))
    )
