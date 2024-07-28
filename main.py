import logging as lg
import warnings
from itertools import combinations

import torch
import wandb
from geomloss import SamplesLoss
from torch import cuda, nn, optim
from tqdm import trange

import losses
import metrics
import models
import scalers
from data import get_dataloaders
from utils import parse_args, seed_everything

warnings.filterwarnings("ignore")


def train(args):
    logger = lg.getLogger()
    formatter = lg.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = lg.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    wandb.init(config=args)

    logger.info(f"Start training with args: {vars(args)}")
    device = "cuda" if cuda.is_available() else "cpu"
    seed_everything(args.seed)
    torch.set_default_dtype(getattr(torch, args.dtype))
    num_source_domains = len(args.source_domains)

    logger.info("Loading data")
    trainloaders, validloaders, testloader = get_dataloaders(
        source_domains=args.source_domains,
        target_domain=args.target_domain,
        forecast_horizon=args.forecast_horizon,
        lookback_horizon=args.lookback_horizon,
        batch_size=args.batch_size,
        dtype=args.dtype,
        fixed_data_size=args.data_size,
    )
    trainloader = zip(*trainloaders)

    logger.info("Configuring model and training")
    model = getattr(models, args.model)(
        args.forecast_horizon, args.lookback_horizon
    ).to(device)
    blocks = model.blocks
    if cuda.device_count() > 1:
        model = nn.DataParallel(model)
        blocks = model.module.blocks

    pred_loss_fn = getattr(losses, args.loss)()
    pred_optimizer = optim.Adam(model.parameters(), lr=args.pred_learning_rate)
    pred_scheduler = optim.lr_scheduler.CyclicLR(
        pred_optimizer,
        base_lr=args.pred_learning_rate / 100,
        max_lr=args.pred_learning_rate,
        step_size_up=args.num_iters // (args.num_lr_cycles * 2),
        mode="triangular2",
        cycle_momentum=False,
    )

    if do_regularization := (args.regularizer and args.temperature > 0):
        scale_fn = getattr(scalers, args.scaler)
        align_loss_fn = SamplesLoss(loss=args.regularizer.replace("MMD", "").lower())
        align_optimizer = optim.Adam(
            sum([list(block.extraction_layer.parameters()) for block in blocks], []),
            lr=args.align_learning_rate,
        )
        align_scheduler = optim.lr_scheduler.CyclicLR(
            align_optimizer,
            base_lr=args.align_learning_rate / 100,
            max_lr=args.align_learning_rate,
            step_size_up=args.num_iters // (args.num_lr_cycles * 2),
            mode="triangular2",
            cycle_momentum=False,
        )

    metric_fn = getattr(metrics, args.metric.lower())

    logger.info("Start training")
    for iter in trange(args.num_iters):
        model.train()
        trainbatches = next(trainloader)
        hists = torch.cat([hist.to(device) for hist, _ in trainbatches], dim=0)
        futrs = torch.cat([futr.to(device) for _, futr in trainbatches], dim=0)
        preds, feats = model(hists)
        pred_loss = pred_loss_fn(futrs, preds)

        if do_regularization:
            align_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for feat in feats:
                stack_align_loss = torch.tensor(0.0, device=device, requires_grad=True)
                feat_by_domain = [
                    feat[
                        domain_idx
                        * args.batch_size : (domain_idx + 1)
                        * args.batch_size
                    ]
                    for domain_idx in range(num_source_domains)
                ]
                for feat1, feat2 in combinations(feat_by_domain, 2):
                    stack_align_loss = torch.max(
                        stack_align_loss,
                        align_loss_fn(scale_fn(feat1), scale_fn(feat2)),
                    )
                align_loss = align_loss + stack_align_loss
            align_optimizer.zero_grad()
            align_loss.backward(retain_graph=True)
            align_optimizer.step()
            align_scheduler.step()
            wandb.log({"iter": iter, f"train_{args.regularizer}": align_loss.item()})

        pred_optimizer.zero_grad()
        pred_loss.backward()
        pred_optimizer.step()
        pred_scheduler.step()
        wandb.log({"iter": iter, f"train_{args.loss}": pred_loss.item()})

        if iter % 10 == 0:
            model.eval()
            with torch.no_grad():
                for validloader in validloaders:
                    for hist, futr in validloader:
                        hist = hist.to(device)
                        futr = futr.numpy()
                        pred = model(hist).detach().cpu().numpy()
                        metric = metric_fn(futr, pred)
                        wandb.log({"iter": iter, f"valid_{args.metric}": metric})

        if iter % 100 == 0:
            model.eval()
            with torch.no_grad():
                for hist, futr in testloader:
                    hist = hist.to(device)
                    futr = futr.numpy()
                    pred = model(hist).detach().cpu().numpy()
                    metric = metric_fn(futr, pred)
                    wandb.log({"iter": iter, f"test_{args.metric}": metric})

    logger.info("Finish training")
    wandb.finish()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
