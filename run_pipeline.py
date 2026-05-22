"""
One-click pipeline: runs all steps sequentially and collects outputs
into a timestamped batch folder.

Usage:
    python run_pipeline.py                      # 1000 images/class, default settings
    python run_pipeline.py -n 2000 --epochs 50  # custom settings
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def run_step(cmd, description):
    """Run a shell command and abort on failure."""
    print("\n" + "=" * 60)
    print(f"  STEP: {description}")
    print(f"  CMD:  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
    if result.returncode != 0:
        print(f"\nERROR: step failed — {description}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Full MLP adversarial training pipeline"
    )
    parser.add_argument("-n", "--n-per-class", type=int, default=1000,
                        help="Images per class (default: 1000)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Main training epochs (default: 30)")
    parser.add_argument("--surrogate-epochs", type=int, default=10,
                        help="Surrogate training epochs (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-name", type=str, default=None,
                        help="Custom batch folder name (default: auto timestamp)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip data generation (use existing txt files)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Parent directory for batch folders (default: .)")
    args = parser.parse_args()

    # ---- create batch folder ----
    if args.batch_name:
        batch_dir = os.path.join(args.output_dir, args.batch_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(args.output_dir, f"batch_{timestamp}")

    os.makedirs(batch_dir, exist_ok=True)
    print(f"\n{'*' * 60}")
    print(f"  Batch folder: {batch_dir}")
    print(f"  Images/class: {args.n_per_class}")
    print(f"{'*' * 60}")

    # ---- step 1: generate raw images ----
    if not args.skip_generate:
        run_step(
            ["python", "generate_data.py",
             "-n", str(args.n_per_class),
             "--seed", str(args.seed),
             "--output-dir", batch_dir],
            "1/5  Generate raw 32×32 images (0-255)"
        )
    else:
        print("\n  (skipping data generation)")

    # ---- step 2: normalize ----
    run_step(
        ["python", "normalize.py",
         "--input-dir", batch_dir,
         "--output-dir", batch_dir],
        "2/5  Normalize pixel values 0-255 → 0-1"
    )

    # ---- step 3: prepare surrogate data ----
    run_step(
        ["python", "prepare_surrogate_data.py",
         "--input-dir", batch_dir,
         "--output-dir", batch_dir],
        "3/5  Stack normalized data → train_X.npy / train_y.npy"
    )

    # ---- step 4: train surrogates ----
    run_step(
        ["python", "surrogate.py",
         "--data-dir", batch_dir,
         "--output-dir", batch_dir],
        "4/5  Train 5 surrogate models + generate adv examples"
    )

    # ---- step 5: main adversarial training ----
    run_step(
        ["python", "TRAIN-5.py",
         "--data-dir", batch_dir,
         "--surrogate-dir", batch_dir,
         "--output", os.path.join(batch_dir, "1.npz")],
        "5/5  Main adversarial training (ensemble PGD)"
    )

    # ---- summary ----
    print("\n" + "*" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs in: {batch_dir}")
    print("*" * 60)

    print("\nContents:")
    for item in sorted(os.listdir(batch_dir)):
        fpath = os.path.join(batch_dir, item)
        size = os.path.getsize(fpath)
        print(f"  {item:40s}  {size:>12,d} bytes")


if __name__ == "__main__":
    main()
