"""
train_datasets_with_ranking.py
==============================
Trains the ranking V-Net (MarginRankingLoss, arch=large) for all 6 datasets:
  Family, Carcinogenesis, Mutagenesis, NCTRER, Lymphography, Animals

For each dataset it runs:
    python train_vocell_ranking.py
        --lps_file      <lps_file>
        --dataset_file  <dataset_file>
        --embeddings    <embeddings>
        --strategy      bootstrap
        --arch          large
        --device        cuda
        --output_dir    <Dataset>_ranking
        [+ any extra args forwarded from CLI]

Usage:
    python train_datasets_with_ranking.py
    python train_datasets_with_ranking.py --datasets family carcinogenesis
    python train_datasets_with_ranking.py --n_rounds 10 --epochs 300
"""

import argparse
import os
import subprocess
import sys

BASE = "../datasets"

# ── Per-dataset configuration ─────────────────────────────────────────────────
DATASETS = {
    "family": {
        "lps":        [
            "LPs/Family/lps_difficult.json",
            f"{BASE}/family/training_data/training_data_prep.json",
        ],
        "dataset":    "generated_search_data/vnet_search_data_family.json",
        "embeddings": f"{BASE}/family/embeddings/DeCaL_entity_embeddings.csv",
        "output_dir": "Family_ranking",
    },
    "carcinogenesis": {
        "lps":        [
            "LPs/Carcinogenesis/lps.json",
            f"{BASE}/carcinogenesis/training_data/training_data_prep.json",
        ],
        "dataset":    "generated_search_data/vnet_search_data_carcinogenesis.json",
        "embeddings": f"{BASE}/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv",
        "output_dir": "Carcinogenesis_ranking",
    },
    "mutagenesis": {
        "lps":        [
            "LPs/Mutagenesis/lps.json",
            f"{BASE}/mutagenesis/training_data/training_data_prep.json",
        ],
        "dataset":    "generated_search_data/vnet_search_data_mutagenesis.json",
        "embeddings": f"{BASE}/mutagenesis/embeddings/DeCaL_entity_embeddings.csv",
        "output_dir": "Mutagenesis_ranking",
    },
    "nctrer": {
        "lps":        [f"{BASE}/nctrer/training_data/training_data_prep.json"],
        "dataset":    "generated_search_data/vnet_search_data_nctrer.json",
        "embeddings": f"{BASE}/nctrer/embeddings/DeCaL_entity_embeddings.csv",
        "output_dir": "Nctrer_ranking",
    },
    "lymphography": {
        "lps":        [f"{BASE}/lymphography/training_data/training_data_prep.json"],
        "dataset":    "generated_search_data/vnet_search_data_lymphography.json",
        "embeddings": f"{BASE}/lymphography/embeddings/DeCaL_entity_embeddings.csv",
        "output_dir": "Lymphography_ranking",
    },
    "animals": {
        "lps":        [f"{BASE}/animals/training_data/training_data_prep.json"],
        "dataset":    "generated_search_data/vnet_search_data_animals.json",
        "embeddings": f"{BASE}/animals/embeddings/DeCaL_entity_embeddings.csv",
        "output_dir": "Animals_ranking",
    },
}

ALL_DATASETS = list(DATASETS.keys())


def run(cmd: list[str], dry_run: bool) -> bool:
    print("\n  CMD:", " ".join(cmd))
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Train ranking V-Net for all datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS,
                        help="Datasets to train (default: all).")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--n_rounds",     type=int,   default=5)
    parser.add_argument("--sample_lp_frac", type=float, default=0.5)
    parser.add_argument("--sample_ex_frac", type=float, default=0.5)
    parser.add_argument("--margin",       type=float, default=0.05)
    parser.add_argument("--max_pairs",    type=int,   default=50_000)
    parser.add_argument("--hidden_dim",   type=int,   default=16)
    parser.add_argument("--dry_run",      action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    ok_runs  = []
    err_runs = []

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]

        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*60}")

        # ── Verify required files exist ───────────────────────────────────
        missing = []
        if not os.path.exists(cfg["dataset"]):
            missing.append(cfg["dataset"])
        if not os.path.exists(cfg["embeddings"]):
            missing.append(cfg["embeddings"])
        for lp in cfg["lps"]:
            if not os.path.exists(lp):
                missing.append(lp)

        if missing and not args.dry_run:
            print(f"  [SKIP] Missing files:")
            for m in missing:
                print(f"    {m}")
            err_runs.append((ds_name, "missing files"))
            continue

        os.makedirs(cfg["output_dir"], exist_ok=True)

        # ── One run per LP file ───────────────────────────────────────────
        for lps_file in cfg["lps"]:
            tag = os.path.splitext(os.path.basename(lps_file))[0]
            print(f"\n  LP file : {lps_file}")

            cmd = [
                sys.executable, "train_vocell_ranking.py",
                "--lps_file",       lps_file,
                "--dataset_file",   cfg["dataset"],
                "--embeddings",     cfg["embeddings"],
                "--strategy",       "bootstrap",
                "--arch",           "large",
                "--device",         "cuda",
                "--output_dir",     cfg["output_dir"],
                "--epochs",         str(args.epochs),
                "--n_rounds",       str(args.n_rounds),
                "--sample_lp_frac", str(args.sample_lp_frac),
                "--sample_ex_frac", str(args.sample_ex_frac),
                "--margin",         str(args.margin),
                "--max_pairs",      str(args.max_pairs),
                "--hidden_dim",     str(args.hidden_dim),
            ]

            success = run(cmd, args.dry_run)
            key = f"{ds_name}/{tag}"
            if success:
                ok_runs.append(key)
            else:
                print(f"  [ERROR] Training failed for {key}")
                err_runs.append((key, "non-zero exit"))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Successful : {len(ok_runs)}")
    for x in ok_runs:
        print(f"    ✓  {x}")
    if err_runs:
        print(f"  Failed     : {len(err_runs)}")
        for name, reason in err_runs:
            print(f"    ✗  {name}  ({reason})")
    print()


if __name__ == "__main__":
    main()
