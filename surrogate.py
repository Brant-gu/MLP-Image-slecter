"""Train 5 surrogate MLP models and generate adversarial examples via ensemble PGD."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
import argparse


# -----------------------------
# 1. Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=(64, 32), out_dim=3, dropout=0.0):
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


# -----------------------------
# 2. Training
# -----------------------------
def train_one_model(hidden, seed, save_path, loader, device,
                    epochs=10, lr=1e-3, dropout=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(hidden=hidden, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        print(f"  {os.path.basename(save_path)} | epoch {epoch+1}/{epochs} | "
              f"loss={total_loss/total:.4f} | acc={correct/total:.4f}")

    torch.save(model.state_dict(), save_path)
    return model


# -----------------------------
# 3. Ensemble PGD Attack
# -----------------------------
def ensemble_pgd_attack(models, x, y, epsilon=0.05, alpha=0.01, steps=10,
                        random_start=True):
    for m in models:
        m.eval()

    x_orig = x.detach().clone()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        loss = 0.0
        for m in models:
            logits = m(x_adv)
            loss = loss + nn.CrossEntropyLoss()(logits, y)

        loss = loss / len(models)

        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


# -----------------------------
# 4. Evaluate
# -----------------------------
def evaluate_model(model, X, y, device, batch_size=256):
    model.eval()
    eval_loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

    return correct / total


# -----------------------------
# 5. Main
# -----------------------------
def main(data_dir=".", output_dir=".", epochs=10,
         epsilon=0.03, alpha=0.01, pgd_steps=5, num_surrogates=5,
         arch_config=None):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    X = np.load(os.path.join(data_dir, "train_X.npy"))
    y = np.load(os.path.join(data_dir, "train_y.npy"))
    print(f"Loaded data: X={X.shape}, y={y.shape}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # build surrogate configs — custom or auto
    if arch_config:
        configs = json.loads(arch_config)
        for i, cfg in enumerate(configs):
            cfg["seed"] = i
            cfg["save"] = f"surrogate_{i+1}.pth"
            cfg["hidden"] = tuple(cfg["hidden"])
        print(f"\n=== Training {len(configs)} surrogate models (custom arch) ===")
    else:
        hidden_pool = [
            (64, 32), (128, 64), (256, 64),
            (128, 128), (256, 128), (512, 128), (256, 256),
        ]
        configs = []
        for i in range(num_surrogates):
            h = hidden_pool[i % len(hidden_pool)]
            dp = round(0.05 * (i % 5), 2)
            configs.append({
                "hidden": h, "seed": i, "dropout": dp,
                "save": f"surrogate_{i+1}.pth",
            })
        print(f"\n=== Training {len(configs)} surrogate models (auto arch) ===")

    models = []
    for cfg in configs:
        save_path = os.path.join(output_dir, cfg["save"])
        model = train_one_model(
            hidden=cfg["hidden"],
            seed=cfg["seed"],
            save_path=save_path,
            loader=loader,
            device=device,
            epochs=epochs,
            dropout=cfg["dropout"]
        )
        models.append(model)

    # generate adversarial examples
    print("\n=== Generating adversarial examples ===")
    adv_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    adv_X_list, adv_y_list = [], []
    for xb, yb in adv_loader:
        xb, yb = xb.to(device), yb.to(device)
        xb_adv = ensemble_pgd_attack(
            models=models, x=xb, y=yb,
            epsilon=epsilon, alpha=alpha, steps=pgd_steps, random_start=True
        )
        adv_X_list.append(xb_adv.cpu())
        adv_y_list.append(yb.cpu())

    adv_X = torch.cat(adv_X_list, dim=0).numpy()
    adv_y = torch.cat(adv_y_list, dim=0).numpy()

    np.save(os.path.join(output_dir, "adv_X.npy"), adv_X)
    np.save(os.path.join(output_dir, "adv_y.npy"), adv_y)
    print(f"Saved adv_X.npy  shape={adv_X.shape}")
    print(f"Saved adv_y.npy  shape={adv_y.shape}")

    delta = np.abs(adv_X - X)
    print(f"mean perturbation: {delta.mean():.6f}")
    print(f"max  perturbation: {delta.max():.6f}")

    # evaluate surrogates on clean vs adv
    print("\n=== Evaluate on clean vs adv ===")
    for i, m in enumerate(models):
        clean_acc = evaluate_model(m, X, y, device)
        adv_acc = evaluate_model(m, adv_X, adv_y, device)
        print(f"Surrogate {i+1}: clean_acc={clean_acc:.4f}, adv_acc={adv_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train surrogate models & generate adversarial examples")
    parser.add_argument("--data-dir", default=".", help="Directory with train_X.npy / train_y.npy")
    parser.add_argument("--output-dir", default=".", help="Directory to save surrogate_*.pth and adv_*.npy")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per surrogate (default: 10)")
    parser.add_argument("--epsilon", type=float, default=0.03, help="PGD perturbation radius (default: 0.03)")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size (default: 0.01)")
    parser.add_argument("--pgd-steps", type=int, default=5, help="PGD iterations (default: 5)")
    parser.add_argument("--num-surrogates", type=int, default=5, help="Number of surrogate models (default: 5)")
    parser.add_argument("--arch-config", type=str, default=None,
                        help='JSON array of per-model arch, e.g. \'[{"hidden":[64,32],"dropout":0.0}]')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.epochs,
         args.epsilon, args.alpha, args.pgd_steps,
         args.num_surrogates, args.arch_config)
