"""Normalize pixel values from [0,255] to [0,1] for all shape txt files."""

import numpy as np
import glob
import os
import argparse


def normalize_files(input_dir=".", output_dir="."):
    prefixes = ["sphere", "cube", "tetrahedron", "mixed-data"]
    files = []
    for p in prefixes:
        files += glob.glob(os.path.join(input_dir, f"{p}*.txt"))

    # filter out files that already have "-normalized" suffix to avoid double-normalizing
    files = [f for f in files if "-normalized" not in f]

    if not files:
        print(f"No raw txt files found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for fname in files:
        basename = os.path.basename(fname)
        outname = os.path.join(output_dir, basename.replace(".txt", "-normalized.txt"))

        with open(fname, "r") as f:
            lines = f.readlines()

        with open(outname, "w") as f:
            for line in lines:
                nums = np.array(line.strip().split(","), dtype=float)
                nums = nums / 255.0
                nums = np.round(nums, 4)
                newline = ",".join(str(x) for x in nums)
                f.write(newline + "\n")

        print(f"Finished: {outname}")

    print("ALL DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize pixel values 0-255 → 0-1")
    parser.add_argument("--input-dir", default=".", help="Directory with raw txt files")
    parser.add_argument("--output-dir", default=".", help="Directory for normalized files")
    args = parser.parse_args()
    normalize_files(args.input_dir, args.output_dir)
