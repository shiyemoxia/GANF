import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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


def _load_dataloaders(args):
    from dataset import load_timeseries, load_water

    data_path = Path(args.data_dir)
    dataset_type = args.dataset_type.lower()

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

    test_labels = _extract_labels(test_loader.dataset)
    return train_loader, val_loader, test_loader, n_sensor, test_labels


parser = argparse.ArgumentParser()
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
    "--machine",
    type=str,
    default="",
    help="Optional SMD machine id such as machine-1-1.",
)
parser.add_argument("--window_size", type=int, default=60)
parser.add_argument("--stride_size", type=int, default=10)
parser.add_argument("--val_ratio", type=float, default=0.5)
parser.add_argument(
    "--label_reduction",
    type=str,
    default="any",
    choices=["any", "first", "last"],
)
parser.add_argument("--output_dir", type=str, default="./checkpoint/model")
parser.add_argument("--name", default="GANF_Water")
parser.add_argument("--graph", type=str, default="./checkpoint/eval/GANF_water_seed_18/graph_best.pt")
parser.add_argument("--model", type=str, default="./checkpoint/eval/water/GANF_water_seed_18_best.pt")
parser.add_argument("--seed", type=int, default=10, help="Random seed to use.")
parser.add_argument("--n_blocks", type=int, default=1)
parser.add_argument("--n_components", type=int, default=1)
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--n_hidden", type=int, default=1)
parser.add_argument("--batch_norm", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=512)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

print(args)
print("Loading dataset")
_, _, test_loader, _, test_labels = _load_dataloaders(args)

model = GANF(
    args.n_blocks,
    1,
    args.hidden_size,
    args.n_hidden,
    dropout=0.0,
    batch_norm=args.batch_norm,
)
model = model.to(device)

if args.model in ("None", "", None):
    raise ValueError("Please provide --model for evaluation")
if args.graph in ("None", "", None):
    raise ValueError("Please provide --graph for evaluation")

model.load_state_dict(torch.load(args.model, map_location=device))
A = torch.load(args.graph, map_location=device).to(device)
model.eval()

loss_test = []
with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        loss = -model.test(x, A.data).cpu().numpy()
        loss_test.append(loss)
loss_test = np.concatenate(loss_test)

roc_test = _compute_roc(test_labels, loss_test)
print(
    "The ROC score on {} dataset is {}".format(
        args.dataset_type.upper(),
        _format_metric(roc_test),
    )
)
