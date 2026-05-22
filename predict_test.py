import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=(128, 128), out_dim=3):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def load_txt_matrix(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if not line[0].isdigit() and not line[0] == '-':
                continue

            try:
                vals = [float(x) for x in line.split(",")]
            except ValueError:
                continue

            rows.append(vals)

    X = np.array(rows, dtype=np.float32)
    return X

def load_from_npz(model, npz_path):
    data = np.load(npz_path, allow_pickle=True)
    coefs = data["coefs"]
    intercepts = data["intercepts"]

    linears = [m for m in model.net if isinstance(m, nn.Linear)]

    for lin, W, b in zip(linears, coefs, intercepts):
        lin.weight.data = torch.from_numpy(W.T).float()
        lin.bias.data = torch.from_numpy(b).float()

def print_accuracy_stats(pred):
    #  0,1,2 
    y_true = np.array([i % 3 for i in range(len(pred))], dtype=np.int64)

    # Overall Accuracy
    overall_acc = (pred == y_true).mean()
    print("\nOverall accuracy:")
    print(f"{overall_acc:.4f} ({overall_acc * 100:.2f}%)")

    # Per-class Accuracy
    label_map = {
        0: "sphere",
        1: "cube",
        2: "tetrahedron"
    }

    print("\nPer-class accuracy:")
    for cls in [0, 1, 2]:
        mask = (y_true == cls)
        cls_acc = (pred[mask] == y_true[mask]).mean()
        print(f"Class {cls} ({label_map[cls]}): {cls_acc:.4f} ({cls_acc * 100:.2f}%)")


def print_group_accuracy(pred, y_true, group_size=3000):
    print(f"\nGroup accuracy (every {group_size} samples):")

    total_groups = len(pred) // group_size

    for g in range(total_groups):
        start = g * group_size
        end = start + group_size

        group_pred = pred[start:end]
        group_true = y_true[start:end]

        acc = (group_pred == group_true).mean()

        print(f"Group {g+1}: {acc:.4f} ({acc * 100:.2f}%)")

    # 处理剩余部分（如果不是3000的整数倍）
    if len(pred) % group_size != 0:
        start = total_groups * group_size
        group_pred = pred[start:]
        group_true = y_true[start:]

        acc = (group_pred == group_true).mean()
        print(f"Remaining ({len(group_pred)} samples): {acc:.4f} ({acc * 100:.2f}%)")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = MLP().to(device)
load_from_npz(model, "0-0-70.npz")
model.eval()



X_test = load_txt_matrix("adv_X_PGD-30000.txt")
print("Test shape:", X_test.shape)

X_test_t = torch.from_numpy(X_test).to(device)

with torch.no_grad():
    logits = model(X_test_t)                        
    probs = torch.softmax(logits, dim=1)           
    confidence, pred = torch.max(probs, dim=1)     

    pred = pred.cpu().numpy()
    confidence = confidence.cpu().numpy()

# real tag 0-1-2, 0,1,2,0,1,2...
y_true = np.array([i % 3 for i in range(len(pred))], dtype=np.int64)

#test
saved_rows = []
saved_groups = 0
max_groups = 10  

for i in range(len(pred)):
    if y_true[i] == 2 and pred[i] == 2:
        if i >= 2:
            saved_rows.append(X_test[i - 2])
            saved_rows.append(X_test[i - 1])
            saved_rows.append(X_test[i])
            saved_groups += 1

            if saved_groups == max_groups:
                break

# if saved_rows:
#     saved_rows = np.array(saved_rows, dtype=np.float32)
#     np.savetxt("30rows.txt", saved_rows, fmt="%.6f", delimiter=",")
#     print(f"\nSaved {saved_groups} groups, total {len(saved_rows)} rows to 30rows.txt")
# else:
#     print("\nNo correctly predicted tetrahedron groups found.")

# total Accuracy
accuracy = (pred == y_true).mean()
print("\nOverall accuracy:")
print(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")

# Per-class accuracy
label_map = {
    0: "sphere",
    1: "cube",
    2: "tetrahedron"
}

print("\nPer-class accuracy:")
for cls in [0, 1, 2]:
    mask = (y_true == cls)
    cls_acc = (pred[mask] == y_true[mask]).mean()
    print(f"Class {cls} ({label_map[cls]}): {cls_acc:.4f} ({cls_acc * 100:.2f}%)")

# Predicted class counts
unique, counts = np.unique(pred, return_counts=True)
print("\nPredicted class counts:")
for u, c in zip(unique, counts):
    print(f"{u} ({label_map[u]}): {c}")

# each sample prediction
print("\nPrediction results:")
for i, p in enumerate(pred):
    correct = "OK" if p == y_true[i] else "WRONG"
    print(f"Sample {i}: true={y_true[i]} ({label_map[y_true[i]]}), "
          f"pred={p} ({label_map[p]}), "
          f"confidence={confidence[i]:.4f}, {correct}")
    
#print now model layer
for m in model.net:
    if isinstance(m, nn.Linear):
        print(m.weight.shape)
    
print_accuracy_stats(pred)
print_group_accuracy(pred, y_true, group_size=3000)