import argparse
import os
import random

import numpy as np
import torch
from torch import cuda
from torch.backends import cudnn

import losses
import metrics
import models
import scalers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-domains", type=str, required=True, nargs="+", help="Source domains"
    )
    parser.add_argument(
        "--target-domain", type=str, required=True, help="Target domain"
    )
    parser.add_argument(
        "--forecast-horizon", type=int, default=10, help="Forecast horizon"
    )
    parser.add_argument(
        "--lookback-multiple", type=float, default=5, help="Lookback multiple"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NHiTS",
        choices=models.__all__,
        help="Model architecture",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="SMAPE",
        choices=losses.__all__,
        help="Forecasting loss function",
    )
    parser.add_argument(
        "--regularizer",
        type=str,
        default="Sinkhorn",
        choices=[
            "None",
            "Sinkhorn",
            "EnergyMMD",
            "GaussianMMD",
            "LaplacianMMD",
        ],
        help="Regularizer measure (None for vanilla model)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Regularizing temperature"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="softmax",
        choices=scalers.__all__,
        help="Normalizing function",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="SMAPE",
        choices=metrics.__all__,
        help="Evaluation metric for validation and test",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num-lr-cycles", type=int, default=50, help="Number of learning rate cycles"
    )
    parser.add_argument("--batch-size", type=int, default=2**12, help="Batch size")
    parser.add_argument(
        "--num-iters", type=int, default=1_000, help="Number of iterations"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type used for torch and numpy",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=75_000,
        help="Fixed data size for each domain (None to use all data)",
    )
    args = parser.parse_args()
    if args.regularizer == "None":
        args.temperature = 0
        args.scaler = None
    args.lookback_horizon = int(args.forecast_horizon * args.lookback_multiple)
    args.pred_learning_rate = args.learning_rate
    args.align_learning_rate = args.learning_rate * args.temperature
    return args


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
