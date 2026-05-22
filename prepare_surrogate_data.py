"""
Read normalized txt files and stack into train_X.npy / train_y.npy for surrogate.py.
"""

import numpy as np
import glob
import os
import argparse


def load_txt_matrix(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",")])
    return np.array(rows, dtype=np.float32)


def prepare_data(input_dir=".", output_dir="."):
    patterns = {
        0: "sphere*-normalized.txt",
        1: "cube*-normalized.txt",
        2: "tetrahedron*-normalized.txt",
    }
    fallback = {
        0: "sphere_mix-normalized.txt",
        1: "cube_mix-normalized.txt",
        2: "tetrahedron_mix-normalized.txt",
    }

    Xs, ys = [], []

    for label in [0, 1, 2]:
        matched = sorted(glob.glob(os.path.join(input_dir, patterns[label])))
        if not matched:
            fb = os.path.join(input_dir, fallback[label])
            if os.path.exists(fb):
                matched = [fb]

        if not matched:
            print(f"WARNING: no files for class {label} (pattern: {patterns[label]})")
            continue

        for fname in matched:
            Xc = load_txt_matrix(fname)
            yc = np.full((Xc.shape[0],), label, dtype=np.int64)
            Xs.append(Xc)
            ys.append(yc)
            print(f"Loaded {fname}: {Xc.shape[0]} samples, label={label}")

    if not Xs:
        raise RuntimeError("No normalized files found. Run normalize.py first.")

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    perm = np.random.default_rng(42).permutation(len(y))
    X, y = X[perm], y[perm]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train_X.npy"), X.astype(np.float32))
    np.save(os.path.join(output_dir, "train_y.npy"), y.astype(np.int64))

    print(f"\nSaved -> {output_dir}/train_X.npy  shape={X.shape}")
    print(f"Saved -> {output_dir}/train_y.npy  shape={y.shape}")
    for c in [0, 1, 2]:
        print(f"  class {c}: {(y == c).sum()} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack normalized txt → train_X.npy / train_y.npy")
    parser.add_argument("--input-dir", default=".", help="Directory with *-normalized.txt files")
    parser.add_argument("--output-dir", default=".", help="Directory to save .npy files")
    args = parser.parse_args()
    prepare_data(args.input_dir, args.output_dir)
