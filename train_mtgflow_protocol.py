"""Train GANF under the MTGFLOW-compatible window-level protocol.

This runner keeps GANF's model and optimization style, but replaces dataset
loading and model selection with the same protocol used in the current paper:

1. Use MTGFLOW's dataset loaders.
2. Use the loader-provided train / val / test window splits.
3. Select checkpoints by the best test window-level AUROC over training.

The goal is a fair apples-to-apples comparison against TemporalNF and MTGFLOW
under one shared evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_

from models.GANF import GANF


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_exclude_cols(raw: str | None) -> list[str]:
    if not raw:
        return []
    cols: list[str] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            cols.append(token)
    return cols


def load_exclude_cols(raw: str | None, raw_file: str | None) -> list[str]:
    cols = parse_exclude_cols(raw)
    if raw_file:
        path = Path(raw_file).resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols.append(line)

    deduped: list[str] = []
    seen: set[str] = set()
    for col in cols:
        key = str(col).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or s.size == 0 or y.size != s.size or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def load_mtgflow_loaders(args):
    mtgflow_root = Path(args.mtgflow_root).resolve()
    dataset_root = str(Path(args.data_path).resolve())
    if str(mtgflow_root) not in sys.path:
        sys.path.insert(0, str(mtgflow_root))

    with pushd(mtgflow_root):
        from Dataset import load_smd_smap_msl, loader_PSM, loader_SWat, loader_WADI  # type: ignore

        dataset = str(args.dataset).lower()
        if dataset == "swat":
            train_loader, val_loader, test_loader, _ = loader_SWat(
                dataset_root,
                args.batch_size,
                args.window_size,
                args.stride,
                args.train_split,
            )
        elif dataset == "psm":
            train_loader, val_loader, test_loader, _ = loader_PSM(
                dataset_root,
                args.batch_size,
                args.window_size,
                args.stride,
                args.train_split,
            )
        elif dataset == "wadi":
            train_loader, val_loader, test_loader, _ = loader_WADI(
                dataset_root,
                args.batch_size,
                args.window_size,
                args.stride,
                args.train_split,
                exclude_cols=args.exclude_cols,
            )
        elif dataset == "smd":
            if not args.machine:
                raise ValueError("--machine is required when dataset=smd.")
            train_loader, val_loader, test_loader, _ = load_smd_smap_msl(
                args.machine,
                args.batch_size,
                args.window_size,
                args.stride,
                args.train_split,
                root=dataset_root,
            )
        elif dataset == "msl":
            train_loader, val_loader, test_loader, _ = load_smd_smap_msl(
                "MSL",
                args.batch_size,
                args.window_size,
                args.stride,
                args.train_split,
                root=dataset_root,
            )
        elif dataset == "smap":
            train_loader, val_loader, test_loader, _ = load_smd_smap_msl(
                "SMAP",
                args.batch_size,
                args.window_size,
                args.stride,
                args.train_split,
                root=dataset_root,
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_loader, val_loader, test_loader


def infer_num_sensors(loader) -> int:
    batch = next(iter(loader))
    x = batch[0]
    if x.ndim != 4:
        raise ValueError(f"Expected MTGFLOW batch with rank 4, got {tuple(x.shape)}")
    return int(x.shape[1])


def collect_scores(model: GANF, loader, adjacency: torch.Tensor, device: torch.device) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            loss = -model.test(x, adjacency.data).detach().cpu().numpy()
            scores.append(loss)
    return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def build_model(args) -> GANF:
    return GANF(
        args.n_blocks,
        1,
        args.hidden_size,
        args.n_hidden,
        dropout=args.dropout,
        model=args.flow_model,
        batch_norm=args.batch_norm,
    )


def initialize_graph(n_sensor: int, device: torch.device) -> torch.Tensor:
    init = torch.zeros([n_sensor, n_sensor], device=device)
    init = xavier_uniform_(init).abs()
    init = init.fill_diagonal_(0.0)
    return init.detach().clone().requires_grad_(True)


def maybe_save_best(
    best_auc: float,
    test_auc: float,
    seed: int,
    epoch: int,
    checkpoint_path: Path,
    model: GANF,
    adjacency: torch.Tensor,
) -> tuple[float, int]:
    if np.isnan(test_auc) or test_auc <= best_auc:
        return best_auc, -1

    torch.save(
        {
            "seed": seed,
            "epoch": epoch,
            "test_auc": float(test_auc),
            "graph": adjacency.detach().cpu(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return float(test_auc), epoch


def run_one_seed(args, seed: int) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_mtgflow_loaders(args)
    n_sensor = infer_num_sensors(train_loader)

    model = build_model(args).to(device)
    adjacency = initialize_graph(n_sensor, device)

    checkpoint_dir = Path(args.output_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.dataset.lower()}_ganf_seed{seed}.pt"

    print("=" * 60)
    print(f"Seed {seed}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Window size: {args.window_size}")
    print(f"Train split: {args.train_split}")
    print(f"Sensors: {n_sensor}")
    print(f"Checkpoint: {checkpoint_path}")

    rho = args.rho_init
    alpha = args.alpha_init
    lambda1 = args.lambda1
    h_A_old = np.inf
    epoch = 0
    best_auc = float("-inf")
    best_epoch = -1

    for _ in range(args.max_iter):
        while rho < args.rho_max:
            optimizer = torch.optim.Adam(
                [
                    {"params": model.parameters(), "weight_decay": args.weight_decay},
                    {"params": [adjacency]},
                ],
                lr=args.lr,
                weight_decay=0.0,
            )

            for _ in range(args.n_epochs):
                epoch += 1
                model.train()
                loss_train: list[float] = []

                for x, _, _ in train_loader:
                    x = x.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = -model(x, adjacency)
                    h = torch.trace(torch.matrix_exp(adjacency * adjacency)) - n_sensor
                    total_loss = loss + 0.5 * rho * h * h + alpha * h + lambda1 * adjacency.abs().sum()

                    total_loss.backward()
                    clip_grad_value_(model.parameters(), 1.0)
                    optimizer.step()
                    adjacency.data.copy_(torch.clamp(adjacency.data, min=0.0, max=1.0))
                    loss_train.append(float(loss.item()))

                val_scores = np.nan_to_num(collect_scores(model, val_loader, adjacency, device))
                test_scores = np.nan_to_num(collect_scores(model, test_loader, adjacency, device))
                val_auc = safe_auc(val_loader.dataset.label, val_scores)
                test_auc = safe_auc(test_loader.dataset.label, test_scores)
                best_auc, maybe_epoch = maybe_save_best(
                    best_auc,
                    test_auc,
                    seed,
                    epoch,
                    checkpoint_path,
                    model,
                    adjacency,
                )
                if maybe_epoch > 0:
                    best_epoch = maybe_epoch

                print(
                    "Epoch: {}, train -log_prob: {:.2f}, roc_val: {}, roc_test: {}, best_test: {}, h: {}".format(
                        epoch,
                        np.mean(loss_train) if loss_train else float("nan"),
                        "NA" if np.isnan(val_auc) else f"{val_auc:.4f}",
                        "NA" if np.isnan(test_auc) else f"{test_auc:.4f}",
                        "NA" if np.isnan(best_auc) else f"{best_auc:.4f}",
                        h.item(),
                    )
                )

            print(f"rho: {rho}, alpha {alpha}, h {h.item()}")
            print("===========================================")

            del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if h.item() > 0.5 * h_A_old:
                rho *= 10
            else:
                break

        h_A_old = h.item()
        alpha += rho * h.item()
        if h_A_old <= args.h_tol or rho >= args.rho_max:
            break

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "weight_decay": args.weight_decay},
            {"params": [adjacency]},
        ],
        lr=args.lr,
        weight_decay=0.0,
    )

    for _ in range(args.final_epochs):
        epoch += 1
        model.train()
        loss_train = []

        for x, _, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = -model(x, adjacency)
            h = torch.trace(torch.matrix_exp(adjacency * adjacency)) - n_sensor
            total_loss = loss + 0.5 * rho * h * h + alpha * h + lambda1 * adjacency.abs().sum()

            total_loss.backward()
            clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            adjacency.data.copy_(torch.clamp(adjacency.data, min=0.0, max=1.0))
            loss_train.append(float(loss.item()))

        val_scores = np.nan_to_num(collect_scores(model, val_loader, adjacency, device))
        test_scores = np.nan_to_num(collect_scores(model, test_loader, adjacency, device))
        val_auc = safe_auc(val_loader.dataset.label, val_scores)
        test_auc = safe_auc(test_loader.dataset.label, test_scores)
        best_auc, maybe_epoch = maybe_save_best(
            best_auc,
            test_auc,
            seed,
            epoch,
            checkpoint_path,
            model,
            adjacency,
        )
        if maybe_epoch > 0:
            best_epoch = maybe_epoch

        print(
            "Epoch: {}, train -log_prob: {:.2f}, roc_val: {}, roc_test: {}, best_test: {}, h: {}".format(
                epoch,
                np.mean(loss_train) if loss_train else float("nan"),
                "NA" if np.isnan(val_auc) else f"{val_auc:.4f}",
                "NA" if np.isnan(test_auc) else f"{test_auc:.4f}",
                "NA" if np.isnan(best_auc) else f"{best_auc:.4f}",
                h.item(),
            )
        )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_auc": float(best_auc),
        "checkpoint_path": str(checkpoint_path),
    }


def summarize(results: Iterable[dict]) -> None:
    rows = list(results)
    aucs = np.asarray([row["best_auc"] for row in rows], dtype=np.float64)
    print("")
    print("=" * 60)
    print("GANF MTGFLOW-PROTOCOL SUMMARY")
    print("=" * 60)
    for row in rows:
        print(
            f"seed={row['seed']:>4} | best_epoch={row['best_epoch']:>3} | "
            f"best_window_auc={row['best_auc']:.4f}"
        )
    print("")
    print(
        f"mean={aucs.mean():.4f} std={aucs.std(ddof=0):.4f} "
        f"min={aucs.min():.4f} max={aucs.max():.4f}"
    )


def write_results_csv(results: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "best_epoch", "best_auc", "checkpoint_path"],
        )
        writer.writeheader()
        writer.writerows(results)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["swat", "psm", "wadi", "smd", "msl", "smap"])
    parser.add_argument("--data-path", required=True, help="Dataset root or file path.")
    parser.add_argument("--mtgflow-root", default="MTGFLOW", help="Path to the MTGFLOW repository root.")
    parser.add_argument("--machine", default="", help="Required when dataset=smd.")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--flow-model", type=str, default="MAF", choices=["MAF", "RealNVP"])
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-hidden", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--n-epochs", type=int, default=1, help="Epochs per DAG update stage.")
    parser.add_argument("--final-epochs", type=int, default=30, help="Final fine-tuning epochs after DAG updates.")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--h-tol", type=float, default=1e-4)
    parser.add_argument("--rho-max", type=float, default=1e16)
    parser.add_argument("--lambda1", type=float, default=0.0)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--alpha-init", type=float, default=0.0)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--results-csv", type=str, required=True)
    parser.add_argument("--exclude-cols", type=str, default="")
    parser.add_argument("--exclude-cols-file", type=str, default="")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.seeds = parse_seeds(args.seeds)
    args.exclude_cols = load_exclude_cols(args.exclude_cols, args.exclude_cols_file)

    results = []
    for seed in args.seeds:
        results.append(run_one_seed(args, seed))

    summarize(results)
    output_csv = Path(args.results_csv).resolve()
    write_results_csv(results, output_csv)
    print(f"Results CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
