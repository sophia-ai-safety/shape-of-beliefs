"""
Linear Field Probing (LFP) on comma->number activations.

In the paper's setup, each dataset corresponds to a different latent Gaussian mean
(m300 ... m700), and activations at comma->number positions encode the model's
current belief state over the next number token.

This script trains a multiclass linear probe on those activations for one layer:
each class is one dataset/mean condition, with sequence splits kept fixed
(seq_0000..0007 train, seq_0008..0009 test). Early comma->number positions are
dropped (`number_start_index`) to focus on the post-equilibration regime.

The learned class weight vectors are then treated as a local linear field over
belief states, following the LFP idea in the paper: many local linear readouts
can approximate a curved manifold better than a single global direction.

Saved outputs include probe weights/state_dict, train/test accuracy, per-dataset
accuracy, and cosine-geometry diagnostics of class vectors (matrix, eigenvalues,
and cumulative explained variance).

Uses dataset-specific activations in
`data/activations/<dataset>/model_layers_<L>_batch*.pt`.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

output_dir_extension = "probes/epoch{epochs:03d}_biasFalse"

number_start_index = 500 # remove the first com2num tokens until the model equilibrates


BASE_DIR = Path(__file__).resolve().parent
ACTIVS_DIR = BASE_DIR / "data" / "activations"
# Train/fit on these datasets (labels 0, 1, ..., n)
TRAIN_DATASETS = [
    "gaussian_m300_s100_l1000_n10",
    "gaussian_m350_s100_l1000_n10",
    "gaussian_m400_s100_l1000_n10",
    "gaussian_m450_s100_l1000_n10",
    "gaussian_m500_s100_l1000_n10",
    "gaussian_m550_s100_l1000_n10",
    "gaussian_m600_s100_l1000_n10",
    "gaussian_m650_s100_l1000_n10",
    "gaussian_m700_s100_l1000_n10",
]

# reserves sequences 0 to 7 for training and sequences 8 & 9 for test
TRAIN_SEQ_IDS = set(range(0, 8))
TEST_SEQ_IDS = set(range(8, 10))
PROBE_BIAS = False

BATCH_SIZE = 2048
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-2

SEED = 0

DEFAULT_LAYER = 1
DEFAULT_EPOCHS = 100


@dataclass
class ProbeSplit:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def _parse_sequence_index(seq_id: str) -> int:
    """eg: returns 1 for seq_0001"""
    suffix = seq_id.split("_")[-1]
    return int(suffix)


def _iter_dataset_acts(dataset: str, layer: int) -> Iterable[Tuple[str, torch.Tensor, int]]:
    """Yield (seq_id, acts, seq_len) for each sequence in dataset."""
    site = f"model_layers_{layer}"
    ds_dir = ACTIVS_DIR / dataset
    pattern = f"{site}_batch*.pt"
    files = sorted(ds_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No activation files found for {dataset}: {ds_dir / pattern}")

    for fpath in files:
        payload = torch.load(fpath, map_location="cpu")
        acts = payload["activations"]  # [batch, seq_len, d_model]
        lengths = payload.get("lengths")
        seq_ids = payload.get("sequence_ids")
        if seq_ids is None:
            raise ValueError(f"Missing sequence_ids in {fpath}")
        for i, seq_id in enumerate(seq_ids):
            seq_len = int(lengths[i].item()) if lengths is not None else acts.shape[1]
            yield seq_id, acts[i, :seq_len], seq_len


def _com2num_from_acts(acts: torch.Tensor, number_start_index: int = 500) -> torch.Tensor:
    """Return com→num activations while removing early positions."""
    seq_len = acts.size(0)
    com2num = acts[2:seq_len:2]
    if com2num.size(0) > number_start_index:
        com2num = com2num[number_start_index:]
    return com2num


def collect_split(
    dataset: str, label: int, layer: int, number_start_index: int = 500
) -> ProbeSplit:
    """Load activations for one dataset and build train/test splits."""
    train_rows: List[torch.Tensor] = []
    test_rows: List[torch.Tensor] = []

    for seq_id, acts, _ in _iter_dataset_acts(dataset, layer):
        seq_idx = _parse_sequence_index(seq_id)
        com2num = _com2num_from_acts(acts, number_start_index)
        if seq_idx in TRAIN_SEQ_IDS:
            train_rows.append(com2num)
        elif seq_idx in TEST_SEQ_IDS:
            test_rows.append(com2num)

    if not train_rows:
        raise ValueError(f"No training activations found for {dataset}")
    if not test_rows:
        raise ValueError(f"No test activations found for {dataset}")

    train_x = torch.cat(train_rows, dim=0)
    # print(train_x.shape)
    test_x = torch.cat(test_rows, dim=0)
    # print(test_x.shape)
    train_y = torch.full((train_x.size(0),), label, dtype=torch.long)
    # print(train_y)
    test_y = torch.full((test_x.size(0),), label, dtype=torch.long)
    # print(test_y.shape)
    return ProbeSplit(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    epochs: int,
    bias: bool = PROBE_BIAS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> torch.nn.Linear:
    """Train a multiclass linear probe."""
    probe = torch.nn.Linear(train_x.size(1), num_classes, bias=bias)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay) #added weight decay
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:02d}/{epochs}: loss={epoch_loss:.4f}")
    return probe


def evaluate(
    probe: torch.nn.Linear, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    datasets: Sequence[str],
) -> Tuple[float, dict[str, float]]:
    with torch.no_grad():
        logits = probe(x)
        preds = logits.argmax(dim=-1)
    overall = (preds == y).float().mean().item()

    per_dataset: dict[str, float] = {}
    for idx, name in enumerate(datasets):
        mask = y == idx
        if mask.any():
            per_dataset[name] = (preds[mask] == y[mask]).float().mean().item()
    return overall, per_dataset


def _print_cosine_matrix(cosine_matrix: torch.Tensor, labels: Sequence[str]):
    labels = list(labels)
    print("Probe weight cosine similarity matrix (rows labeled):")
    for label, row in zip(labels, cosine_matrix):
        formatted = " ".join(f"{val:+.2f}" for val in row)
        print(f"{label} {formatted}")


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on activation data.")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="Layer index to use.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    train_feats: List[torch.Tensor] = []
    train_labels: List[torch.Tensor] = []
    in_domain_test_feats: List[torch.Tensor] = []
    in_domain_test_labels: List[torch.Tensor] = []

    # Train on the specified training datasets (and keep their held-out slices for reference).
    for label, dataset in enumerate(TRAIN_DATASETS):
        split = collect_split(dataset, label, args.layer, number_start_index)
        train_feats.append(split.train_x)
        train_labels.append(split.train_y)
        in_domain_test_feats.append(split.test_x)
        in_domain_test_labels.append(split.test_y)

    train_x = torch.cat(train_feats, dim=0)
    train_y = torch.cat(train_labels, dim=0)
    test_x = torch.cat(in_domain_test_feats, dim=0)
    test_y = torch.cat(in_domain_test_labels, dim=0)

    probe = train_probe(train_x, train_y, num_classes=len(TRAIN_DATASETS), epochs=args.epochs)
    train_acc, _ = evaluate(probe, train_x, train_y, TRAIN_DATASETS)
    test_acc, per_dataset_acc = evaluate(probe, test_x, test_y, TRAIN_DATASETS)

    # Cosine similarity between learned class vectors.
    with torch.no_grad():
        weight = probe.weight  # [num_classes, d_model]
        weight -= weight.mean(dim=0, keepdim=True) # added from ChatGPT, to remove "trasnlation freedom" -- keep or not?
        norm_w = F.normalize(weight, dim=1)
        cosine_matrix = norm_w @ norm_w.t()
        eigvals = torch.linalg.eigvalsh(cosine_matrix)
        eigvals_desc = torch.flip(eigvals, dims=[0])
        total = eigvals_desc.sum()
        if total.abs() > 1e-8:
            explained = eigvals_desc / total
        else:
            explained = torch.zeros_like(eigvals_desc)
        cumulative_explained = torch.cumsum(explained, dim=0)

    output_dir = BASE_DIR / output_dir_extension.format(epochs=args.epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"linear_probe_layer{args.layer}.pt"
    torch.save(
        {
            "layer": args.layer,
            "epochs": args.epochs,
            "train_datasets": TRAIN_DATASETS,
            "label_to_index": {ds: i for i, ds in enumerate(TRAIN_DATASETS)},
            "com2num_limit": number_start_index,
            "train_sequences": sorted(TRAIN_SEQ_IDS),
            "test_sequences": sorted(TEST_SEQ_IDS),
            "train_size": int(train_y.numel()),
            "test_size": int(test_y.numel()),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "per_dataset_accuracy": per_dataset_acc,
            "cosine_matrix": cosine_matrix.cpu(),
            "cosine_eigenvalues_desc": eigvals_desc.cpu(),
            "cosine_explained_variance": explained.cpu(),
            "cosine_cumulative_explained": cumulative_explained.cpu(),
            "probe_state_dict": probe.state_dict(),
        },
        output_path,
    )

    print(f"Saved probe to {output_path}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    for ds, acc in per_dataset_acc.items():
        print(f"  {ds}: {acc:.4f}")
    _print_cosine_matrix(cosine_matrix.cpu(), TRAIN_DATASETS)
    print("Cosine matrix eigenvalues (desc):")
    print("  " + " ".join(f"{v:.4f}" for v in eigvals_desc))
    # print("Explained variance (desc):")
    # print("  " + " ".join(f"{v:.4f}" for v in explained))
    print("Cumulative explained variance:")
    print("  " + " ".join(f"{v:.4f}" for v in cumulative_explained))


if __name__ == "__main__":
    main()
