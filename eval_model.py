"""
Evaluate a trained MLP model (.npz) against a dataset.
Outputs JSON: overall accuracy, per-class accuracy, confidence-sorted decile groups.
"""

import numpy as np
import torch
import torch.nn as nn
import json
import argparse
import os
import glob


class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=(128, 128), out_dim=3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_from_npz(model, npz_path):
    data = np.load(npz_path, allow_pickle=True)
    coefs = data["coefs"]
    intercepts = data["intercepts"]
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    for lin, W, b in zip(linears, coefs, intercepts):
        lin.weight.data = torch.from_numpy(W.T).float()
        lin.bias.data = torch.from_numpy(b).float()


def infer_architecture(npz_path):
    """Read weight shapes from .npz to determine hidden layer sizes."""
    data = np.load(npz_path, allow_pickle=True)
    coefs = data["coefs"]
    hidden = []
    for W in coefs:
        hidden.append(W.shape[0])
    # last layer is output (3 classes); everything before is hidden
    return tuple(hidden[:-1]), hidden[0] if hidden else 1024


def load_txt_matrix(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals = [float(x) for x in line.split(",")]
            except ValueError:
                continue
            rows.append(vals)
    return np.array(rows, dtype=np.float32)


def main(model_path, data_dir, output_json=None):
    # infer architecture from .npz
    hidden, _ = infer_architecture(model_path)

    device = torch.device("cpu")
    model = MLP(in_dim=1024, hidden=hidden, out_dim=3).to(device)
    load_from_npz(model, model_path)
    model.eval()

    # load normalized data
    label_map = {0: "sphere", 1: "cube", 2: "tetrahedron"}
    patterns = {
        0: "sphere*-normalized.txt",
        1: "cube*-normalized.txt",
        2: "tetrahedron*-normalized.txt",
    }
    Xs, ys = [], []
    for label in [0, 1, 2]:
        matched = sorted(glob.glob(os.path.join(data_dir, patterns[label])))
        if not matched:
            fb = os.path.join(data_dir, f"{label_map[label]}_mix-normalized.txt")
            if os.path.exists(fb):
                matched = [fb]
        for fname in matched:
            Xc = load_txt_matrix(fname)
            yc = np.full((Xc.shape[0],), label, dtype=np.int64)
            Xs.append(Xc)
            ys.append(yc)

    if not Xs:
        result = {"error": "No normalized data files found"}
        if output_json:
            with open(output_json, "w") as f:
                json.dump(result, f)
        else:
            print(json.dumps(result))
        return

    X = np.vstack(Xs)
    y_true = np.concatenate(ys)

    # inference
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        pred = pred.cpu().numpy()
        confidence = confidence.cpu().numpy()

    accuracy = float((pred == y_true).mean())

    # per-class accuracy
    per_class = []
    for cls in [0, 1, 2]:
        mask = y_true == cls
        if mask.sum() > 0:
            cls_acc = float((pred[mask] == y_true[mask]).mean())
        else:
            cls_acc = 0.0
        per_class.append({
            "class": cls,
            "name": label_map[cls],
            "accuracy": round(cls_acc, 4),
            "count": int(mask.sum()),
        })

    # decile groups sorted by confidence
    sort_idx = np.argsort(confidence)
    pred_sorted = pred[sort_idx]
    true_sorted = y_true[sort_idx]
    conf_sorted = confidence[sort_idx]

    n = len(pred_sorted)
    deciles = []
    for i in range(10):
        start = int(round(n * i / 10))
        end = int(round(n * (i + 1) / 10))
        if start >= n:
            break
        group_pred = pred_sorted[start:end]
        group_true = true_sorted[start:end]
        group_conf = conf_sorted[start:end]
        group_acc = float((group_pred == group_true).mean())
        deciles.append({
            "group": i + 1,
            "range_pct": f"{int(round(start/n*100))}–{int(round(end/n*100))}%",
            "accuracy": round(group_acc, 4),
            "avg_confidence": round(float(group_conf.mean()), 4),
            "min_confidence": round(float(group_conf.min()), 4),
            "max_confidence": round(float(group_conf.max()), 4),
            "count": int(len(group_pred)),
        })

    # overall avg confidence
    avg_confidence = round(float(confidence.mean()), 4)

    result = {
        "model": os.path.basename(model_path),
        "data_dir": os.path.basename(data_dir),
        "total_samples": int(n),
        "overall_accuracy": round(accuracy, 4),
        "avg_confidence": avg_confidence,
        "per_class": per_class,
        "deciles": deciles,
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLP model on dataset")
    parser.add_argument("--model", required=True, help="Path to .npz model file")
    parser.add_argument("--data-dir", required=True, help="Directory with *-normalized.txt files")
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()
    main(args.model, args.data_dir, args.output_json)
