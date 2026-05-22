"""Main adversarial training with ensemble PGD using surrogate models."""

import glob
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import argparse

# 3classes: 0=sphere, 1=cube, 2=tetrahedron
CLASS_PATTERNS = {
    0: "sphere*-normalized.txt",
    1: "cube*-normalized.txt",
    2: "tetrahedron*-normalized.txt",
}
CLASS_FALLBACK = {
    0: "sphere_mix-normalized.txt",
    1: "cube_mix-normalized.txt",
    2: "tetrahedron_mix-normalized.txt",
}


def load_txt_matrix(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",")])
    X = np.array(rows, dtype=np.float32)
    return X


def stratified_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)

    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            n_test = 0
        else:
            n_test = int(np.floor(n * test_size))
            n_test = max(1, n_test)
            n_test = min(n_test, n - 1)

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_test_parts.append(X[test_idx])
        y_test_parts.append(y[test_idx])
        X_train_parts.append(X[train_idx])
        y_train_parts.append(y[train_idx])

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_test = np.vstack(X_test_parts)
    y_test = np.concatenate(y_test_parts)

    train_perm = rng.permutation(len(y_train))
    test_perm = rng.permutation(len(y_test))
    return X_train[train_perm], X_test[test_perm], y_train[train_perm], y_test[test_perm]


class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=(128, 128), out_dim=3, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def ensemble_pgd_attack(models, x, y, epsilon=0.2, alpha=0.01, steps=10, random_start=True,
                        clamp_min=0.0, clamp_max=1.0):
    for m in models:
        m.eval()

    x_orig = x.detach().clone()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, clamp_min, clamp_max)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        loss = 0.0
        for m in models:
            logits = m(x_adv)
            loss = loss + F.cross_entropy(logits, y)
        loss = loss / len(models)

        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, clamp_min, clamp_max)

    return x_adv.detach()


def classification_report_np(y_true, y_pred, num_classes=3):
    eps = 1e-12
    lines = []
    lines.append("class  precision  recall  f1  support")
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        lines.append(f"{c:5d}  {precision:9.4f}  {recall:6.4f}  {f1:4.4f}  {support:7d}")

    return "\n".join(lines)


def main(data_dir=".", surrogate_dir=".", output_path="1.npz", epochs=30,
         epsilon=0.06, alpha=0.01, pgd_steps=10, lambda_adv=0.6,
         num_surrogates=5, arch_config=None):
    # ---- data loading ----
    Xs, ys = [], []

    for label, pattern in CLASS_PATTERNS.items():
        matched_files = sorted(glob.glob(os.path.join(data_dir, pattern)))

        if not matched_files:
            fb = os.path.join(data_dir, CLASS_FALLBACK[label])
            if os.path.exists(fb):
                matched_files = [fb]

        if not matched_files:
            print(f"Warning: no files matched pattern: {pattern} in {data_dir}")
            continue

        for fname in matched_files:
            Xc = load_txt_matrix(fname)
            yc = np.full((Xc.shape[0],), label, dtype=np.int64)
            Xs.append(Xc)
            ys.append(yc)
            print(f"Loaded {fname}: {Xc.shape[0]} samples, label={label}")

    if not Xs:
        raise RuntimeError("No normalized txt files found.")

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    for c in np.unique(y):
        print("class", c, "count =", int((y == c).sum()))

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=42)
    print("train size =", len(y_train), "test size =", len(y_test))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- load surrogate models dynamically ----
    if arch_config:
        archs = json.loads(arch_config)
        surrogates = []
        for i, arch in enumerate(archs):
            h = tuple(arch["hidden"])
            dp = arch.get("dropout", 0.0)
            m = MLP(hidden=h, dropout=dp).to(device)
            m.load_state_dict(torch.load(
                os.path.join(surrogate_dir, f"surrogate_{i+1}.pth"),
                map_location=device))
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            surrogates.append(m)
    else:
        hidden_pool = [
            (64, 32), (128, 64), (256, 64),
            (128, 128), (256, 128), (512, 128), (256, 256),
        ]
        surrogates = []
        for i in range(num_surrogates):
            h = hidden_pool[i % len(hidden_pool)]
            dp = round(0.05 * (i % 5), 2)
            m = MLP(hidden=h, dropout=dp).to(device)
            m.load_state_dict(torch.load(
                os.path.join(surrogate_dir, f"surrogate_{i+1}.pth"),
                map_location=device))
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            surrogates.append(m)
    print(f"Loaded {len(surrogates)} surrogate models")

    # ---- training ----
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_clean_correct = 0
        total_adv_correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits_clean = model(xb)
            loss_clean = criterion(logits_clean, yb)

            xb_adv = ensemble_pgd_attack(
                models=[model] + surrogates,
                x=xb, y=yb,
                epsilon=epsilon, alpha=alpha, steps=pgd_steps, random_start=True
            )

            logits_adv = model(xb_adv)
            loss_adv = criterion(logits_adv, yb)

            loss = (1 - lambda_adv) * loss_clean + lambda_adv * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

            pred_clean = logits_clean.argmax(dim=1)
            pred_adv = logits_adv.argmax(dim=1)

            total_clean_correct += (pred_clean == yb).sum().item()
            total_adv_correct += (pred_adv == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_clean_acc = total_clean_correct / total
        train_adv_acc = total_adv_correct / total

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={train_loss:.4f}  "
                f"clean_acc={train_clean_acc:.4f}  "
                f"adv_acc={train_adv_acc:.4f}"
            )

    # ---- evaluation ----
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())

    if len(all_pred) == 0:
        raise RuntimeError(
            f"test_loader produced 0 batches. "
            f"len(test_ds)={len(test_ds)}, batch_size={test_loader.batch_size}."
        )

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    acc = (pred == true).mean()
    print("Accuracy:", acc)
    print(classification_report_np(true, pred, num_classes=3))

    # ---- save weights ----
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    coefs = []
    intercepts = []

    for lin in linears:
        W = lin.weight.detach().cpu().numpy().T
        b = lin.bias.detach().cpu().numpy()
        coefs.append(W)
        intercepts.append(b)

    coefs_obj = np.array(coefs, dtype=object)
    intercepts_obj = np.array(intercepts, dtype=object)
    classes_ = np.array([0, 1, 2], dtype=np.int64)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        output_path,
        coefs=coefs_obj,
        intercepts=intercepts_obj,
        classes_=classes_
    )

    print(f"Saved weights to {output_path}")
    print("Layer shapes:", [w.shape for w in coefs], [b.shape for b in intercepts])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial training with ensemble PGD")
    parser.add_argument("--data-dir", default=".", help="Directory with *-normalized.txt files")
    parser.add_argument("--surrogate-dir", default=".", help="Directory with surrogate_*.pth files")
    parser.add_argument("--output", default="1.npz", help="Path for output weights .npz")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--epsilon", type=float, default=0.06, help="PGD perturbation radius (default: 0.06)")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size (default: 0.01)")
    parser.add_argument("--pgd-steps", type=int, default=10, help="PGD iterations (default: 10)")
    parser.add_argument("--lambda-adv", type=float, default=0.6, help="Adversarial loss weight (default: 0.6)")
    parser.add_argument("--num-surrogates", type=int, default=5, help="Number of surrogate models (default: 5)")
    parser.add_argument("--arch-config", type=str, default=None,
                        help='JSON array of per-model arch, e.g. \'[{"hidden":[64,32],"dropout":0.0}]')
    args = parser.parse_args()
    main(args.data_dir, args.surrogate_dir, args.output, args.epochs,
         args.epsilon, args.alpha, args.pgd_steps, args.lambda_adv,
         args.num_surrogates, args.arch_config)
