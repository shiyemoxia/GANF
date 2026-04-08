import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_

from models.GANF import GANF


def _extract_labels(dataset):
    labels = getattr(dataset, "label", None)
    if labels is None:
        return None
    if hasattr(labels, "values"):
        return np.asarray(labels.values, dtype=int)
    return np.asarray(labels, dtype=int)


def _compute_roc(labels, scores):
    if labels is None:
        return float("nan")
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def _format_metric(value):
    if np.isnan(value):
        return "NA"
    return "{:.4f}".format(value)


def _collect_scores(model, loader, adjacency, device):
    scores = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            loss = -model.test(x, adjacency.data).cpu().numpy()
            scores.append(loss)
    return np.concatenate(scores)


def _load_dataloaders(args):
    from dataset import load_timeseries, load_traffic, load_water

    data_path = Path(args.data_dir)
    dataset_type = args.dataset_type.lower()

    if dataset_type == "metr-la":
        train_loader, val_loader, test_loader, n_sensor = load_traffic(
            "{}/{}.h5".format(args.data_dir, args.dataset),
            args.batch_size,
        )
        return train_loader, val_loader, test_loader, n_sensor, None, None

    if data_path.is_file():
        train_loader, val_loader, test_loader, n_sensor = load_water(
            str(data_path), args.batch_size
        )
    else:
        train_loader, val_loader, test_loader, n_sensor = load_timeseries(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            dataset_type=dataset_type,
            machine=args.machine,
            window_size=args.window_size,
            stride_size=args.stride_size,
            val_ratio=args.val_ratio,
            label_reduction=args.label_reduction,
        )

    val_labels = _extract_labels(val_loader.dataset)
    test_labels = _extract_labels(test_loader.dataset)
    return train_loader, val_loader, test_loader, n_sensor, val_labels, test_labels


parser = argparse.ArgumentParser()
# files
parser.add_argument(
    "--data_dir",
    type=str,
    default="./data/SWaT_Dataset_Attack_v0.csv",
    help="Path to a legacy SWAT csv file or to a dataset directory.",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    default="swat",
    help="Dataset type: swat/wadi/psm/smd/msl/smap or legacy csv SWAT.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="metr-la",
    help="Traffic dataset name used only when dataset_type=metr-la.",
)
parser.add_argument(
    "--machine",
    type=str,
    default="",
    help="Optional SMD machine id such as machine-1-1.",
)
parser.add_argument(
    "--window_size",
    type=int,
    default=60,
    help="Sliding window size for ts_cflow-backed datasets.",
)
parser.add_argument(
    "--stride_size",
    type=int,
    default=10,
    help="Sliding window stride for ts_cflow-backed datasets.",
)
parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.5,
    help="Fraction of labeled evaluation windows used for validation.",
)
parser.add_argument(
    "--label_reduction",
    type=str,
    default="any",
    choices=["any", "first", "last"],
    help="How point labels are reduced to a window label.",
)
parser.add_argument("--output_dir", type=str, default="./checkpoint/model")
parser.add_argument("--name", default="GANF_Water")
# restore
parser.add_argument("--graph", type=str, default="None")
parser.add_argument("--model", type=str, default="None")
parser.add_argument("--seed", type=int, default=18, help="Random seed to use.")
# made parameters
parser.add_argument(
    "--n_blocks",
    type=int,
    default=1,
    help="Number of blocks to stack in a model.",
)
parser.add_argument(
    "--n_components",
    type=int,
    default=1,
    help="Number of Gaussian clusters for mixture models.",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=32,
    help="Hidden layer size for MADE.",
)
parser.add_argument(
    "--n_hidden",
    type=int,
    default=1,
    help="Number of hidden layers in each MADE.",
)
parser.add_argument("--batch_norm", type=bool, default=False)
# training params
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
parser.add_argument(
    "--log_interval",
    type=int,
    default=5,
    help="How often to save checkpoints.",
)

parser.add_argument("--h_tol", type=float, default=1e-4)
parser.add_argument("--rho_max", type=float, default=1e16)
parser.add_argument("--max_iter", type=int, default=20)
parser.add_argument("--lambda1", type=float, default=0.0)
parser.add_argument("--rho_init", type=float, default=1.0)
parser.add_argument("--alpha_init", type=float, default=0.0)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

print(args)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Loading dataset")
train_loader, val_loader, test_loader, n_sensor, val_labels, test_labels = _load_dataloaders(args)

rho = args.rho_init
alpha = args.alpha_init
lambda1 = args.lambda1
h_A_old = np.inf

max_iter = args.max_iter
rho_max = args.rho_max
h_tol = args.h_tol
epoch = 0

if args.graph != "None":
    init = torch.load(args.graph, map_location=device).abs()
    print("Load graph from " + args.graph)
else:
    init = torch.zeros([n_sensor, n_sensor], device=device)
    init = xavier_uniform_(init).abs()
    init = init.fill_diagonal_(0.0)
A = torch.tensor(init, requires_grad=True, device=device)

model = GANF(
    args.n_blocks,
    1,
    args.hidden_size,
    args.n_hidden,
    dropout=0.0,
    batch_norm=args.batch_norm,
)
model = model.to(device)

if args.model != "None":
    model.load_state_dict(torch.load(args.model, map_location=device))
    print("Load model from " + args.model)

save_path = os.path.join(args.output_dir, args.name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

loss_best = np.inf

for _ in range(max_iter):
    while rho < rho_max:
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters(), "weight_decay": args.weight_decay},
                {"params": [A]},
            ],
            lr=args.lr,
            weight_decay=0.0,
        )

        for _ in range(args.n_epochs):
            loss_train = []
            epoch += 1
            model.train()

            for x in train_loader:
                x = x.to(device)

                optimizer.zero_grad()
                loss = -model(x, A)
                h = torch.trace(torch.matrix_exp(A * A)) - n_sensor
                total_loss = loss + 0.5 * rho * h * h + alpha * h + lambda1 * A.abs().sum()

                total_loss.backward()
                clip_grad_value_(model.parameters(), 1)
                optimizer.step()
                loss_train.append(loss.item())
                A.data.copy_(torch.clamp(A.data, min=0, max=1))

            model.eval()
            loss_val = np.nan_to_num(_collect_scores(model, val_loader, A, device))
            loss_test = np.nan_to_num(_collect_scores(model, test_loader, A, device))

            roc_val = _compute_roc(val_labels, loss_val)
            roc_test = _compute_roc(test_labels, loss_test)
            print(
                "Epoch: {}, train -log_prob: {:.2f}, val -log_prob: {:.2f}, roc_val: {}, roc_test: {}, h: {}".format(
                    epoch,
                    np.mean(loss_train),
                    np.mean(loss_val),
                    _format_metric(roc_val),
                    _format_metric(roc_test),
                    h.item(),
                )
            )

        print("rho: {}, alpha {}, h {}".format(rho, alpha, h.item()))
        print("===========================================")
        torch.save(A.data, os.path.join(save_path, "graph_{}.pt".format(epoch)))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_{}.pt".format(args.name, epoch)))

        del optimizer
        if args.cuda:
            torch.cuda.empty_cache()

        if h.item() > 0.5 * h_A_old:
            rho *= 10
        else:
            break

    h_A_old = h.item()
    alpha += rho * h.item()

    if h_A_old <= h_tol or rho >= rho_max:
        break

optimizer = torch.optim.Adam(
    [
        {"params": model.parameters(), "weight_decay": args.weight_decay},
        {"params": [A]},
    ],
    lr=args.lr,
    weight_decay=0.0,
)

for _ in range(30):
    loss_train = []
    epoch += 1
    model.train()
    for x in train_loader:
        x = x.to(device)

        optimizer.zero_grad()
        loss = -model(x, A)
        h = torch.trace(torch.matrix_exp(A * A)) - n_sensor
        total_loss = loss + 0.5 * rho * h * h + alpha * h + lambda1 * A.abs().sum()

        total_loss.backward()
        clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        loss_train.append(loss.item())
        A.data.copy_(torch.clamp(A.data, min=0, max=1))

    model.eval()
    loss_val = np.nan_to_num(_collect_scores(model, val_loader, A, device))
    loss_test = np.nan_to_num(_collect_scores(model, test_loader, A, device))
    roc_val = _compute_roc(val_labels, loss_val)
    roc_test = _compute_roc(test_labels, loss_test)
    print(
        "Epoch: {}, train -log_prob: {:.2f}, val -log_prob: {:.2f}, roc_val: {}, roc_test: {}, h: {}".format(
            epoch,
            np.mean(loss_train),
            np.mean(loss_val),
            _format_metric(roc_val),
            _format_metric(roc_test),
            h.item(),
        )
    )

    if np.mean(loss_val) < loss_best:
        loss_best = np.mean(loss_val)
        print("save model {} epoch".format(epoch))
        torch.save(A.data, os.path.join(save_path, "graph_best.pt"))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_best.pt".format(args.name)))

    if epoch % args.log_interval == 0:
        torch.save(A.data, os.path.join(save_path, "graph_{}.pt".format(epoch)))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_{}.pt".format(args.name, epoch)))
