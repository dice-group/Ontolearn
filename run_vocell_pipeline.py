"""
run_vocell_pipeline.py
======================
Orchestrate the full VoCell-BS pipeline for one or more datasets:

  Step 1 (generate) — run beam search and record search trees
  Step 2 (train)    — train the V-Net offline
  Step 3 (evaluate) — run all learners and save result CSV

Usage examples
--------------
# Full pipeline for all datasets:
    python run_vocell_pipeline.py

# Only carcinogenesis and mutagenesis, all steps:
    python run_vocell_pipeline.py --datasets carcinogenesis mutagenesis

# Skip generation (data already exists), run train + evaluate:
    python run_vocell_pipeline.py --steps train evaluate

# Only evaluate on family with existing checkpoints:
    python run_vocell_pipeline.py --datasets family --steps evaluate

# Dry-run: print commands without executing:
    python run_vocell_pipeline.py --dry_run
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).parent / 'configs' / 'datasets.yaml'
PRUNECEL_JAR = 'PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar'


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge(defaults: dict, dataset_cfg: dict) -> dict:
    """Dataset values override defaults."""
    merged = dict(defaults)
    merged.update(dataset_cfg)
    return merged


def run(cmd: list[str], dry_run: bool) -> None:
    print('\n$ ' + ' '.join(cmd))
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'[!] Command failed with exit code {result.returncode}')
            sys.exit(result.returncode)


def step_generate(name: str, cfg: dict, dry_run: bool) -> None:
    print(f'\n{"="*60}\n[generate] {name}\n{"="*60}')
    run([
        sys.executable, 'generate_vnet_dataset.py',
        '--lp_file',    cfg['lps_file'],
        '--output',     cfg['dataset_file'],
        '--kb',         cfg['kb'],
        '--sparql',     cfg['sparql'],
        '--beam_width', str(cfg['beam_width']),
        '--time_limit', str(cfg['time_limit']),
    ], dry_run)


def step_train(name: str, cfg: dict, dry_run: bool) -> None:
    print(f'\n{"="*60}\n[train] {name}\n{"="*60}')
    run([
        sys.executable, 'train_vocell_v_net.py',
        '--lps_file',    cfg['lps_file'],
        '--dataset_file', cfg['dataset_file'],
        '--embeddings',  cfg['embeddings'],
        '--strategy',    cfg['strategy'],
        '--epochs',      str(cfg['epochs']),
        '--output_dir',  cfg['ckpt_dir'],
    ], dry_run)


def step_evaluate(name: str, cfg: dict, dry_run: bool) -> None:
    print(f'\n{"="*60}\n[evaluate] {name}\n{"="*60}')
    learners = cfg.get('learners', ['vocell', 'celoe', 'ocel'])
    strategy = cfg['strategy']
    vocell_sparql   = cfg.get('vocell_sparql', cfg['sparql'])
    prunecel_sparql = cfg.get('prunecel_sparql', cfg['sparql'])

    cmd = [
        sys.executable, 'examples/concept_learning_drill2_train.py',
        '--path_knowledge_base',  cfg['kb'],
        '--path_learning_problem', cfg['lps_file'],
        '--max_runtime',          str(cfg['max_runtime']),
        '--learners',             *learners,
        '--vocell_sparql',        vocell_sparql,
        '--vocell_embeddings',    cfg['embeddings'],
        '--vocell_agg_type',      'mean',
        '--vocell_strategy',      strategy,
        '--vocell_ckpt_dir',      cfg['ckpt_dir'],
        '--prunecel_sparql_url',  prunecel_sparql,
        '--prunecel_jar',         PRUNECEL_JAR,
        '--num_problems',         '0',
    ]
    # Pass dataset-specific embeddings for Drill/DrillV if available
    if 'embeddings' in cfg:
        cmd += ['--path_embeddings', cfg['embeddings']]

    run(cmd, dry_run)


def main():
    parser = argparse.ArgumentParser(
        description='Run the full VoCell pipeline from a YAML config.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', default=str(CONFIG_FILE),
        help='Path to datasets YAML config file.',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Dataset names to run (default: all in config).',
    )
    parser.add_argument(
        '--steps', nargs='+',
        default=['generate', 'train', 'evaluate'],
        choices=['generate', 'train', 'evaluate'],
        help='Pipeline steps to execute.',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print commands without executing them.',
    )
    args = parser.parse_args()

    config    = load_config(Path(args.config))
    defaults  = config.get('defaults', {})
    all_ds    = config.get('datasets', {})

    chosen = args.datasets or list(all_ds.keys())
    unknown = set(chosen) - set(all_ds.keys())
    if unknown:
        print(f'Unknown dataset(s): {unknown}. Available: {list(all_ds.keys())}')
        sys.exit(1)

    steps = args.steps

    for name in chosen:
        cfg = merge(defaults, all_ds[name])
        print(f'\n{"#"*60}\nDataset: {name.upper()}\n{"#"*60}')
        print(f'  steps    : {steps}')
        print(f'  strategy : {cfg["strategy"]}')
        print(f'  ckpt_dir : {cfg["ckpt_dir"]}')

        if 'generate' in steps:
            step_generate(name, cfg, args.dry_run)
        if 'train' in steps:
            step_train(name, cfg, args.dry_run)
        if 'evaluate' in steps:
            step_evaluate(name, cfg, args.dry_run)

    print('\n✓ Pipeline complete.')


if __name__ == '__main__':
    main()
