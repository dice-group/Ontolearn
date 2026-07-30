"""
run_all_experiments_ranking.py
==============================
Runs Vocell evaluation with ranking V-Net checkpoints across all datasets.

Unlike run_all_experiments.py (which reads LP JSON files), this script reads
the LP names directly from the existing results CSVs in results/ so that we
evaluate on exactly the same LPs that were already benchmarked.

For each dataset:
  1. Reads LP names from results/{dataset}_results.csv
  2. Builds a temporary LP JSON that vocell.py can consume
  3. Starts a Fuseki SPARQL endpoint
  4. Runs vocell.py with the ranking checkpoint (arch=large, bootstrap)
  5. Appends results to results_ranking/{dataset}_results.csv
  6. Stops Fuseki

Usage:
  python run_all_experiments_ranking.py
  python run_all_experiments_ranking.py --datasets family carcinogenesis
  python run_all_experiments_ranking.py --dry_run
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration  (mirrors run_all_experiments.py)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DATASETS = "../datasets"
FUSEKI        = os.path.join(
    os.path.dirname(__file__),
    "../apache-jena-fuseki-4.10.0/fuseki-server",
)
FUSEKI_LOG    = "/tmp/fuseki_ranking_experiment.log"
FUSEKI_PORT   = 3030
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")
OUT_DIR       = os.path.join(os.path.dirname(__file__), "results_ranking")

DATASETS = {
    "animals": {
        "kb":          f"{BASE_DATASETS}/animals/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/animals/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "animals",
        "checkpoint":  "Animals_ranking/vocell_v_net_bootstrap_ranking_large.pt",
        "results_csv": os.path.join(RESULTS_DIR, "animals_results.csv"),
        "lp_sources":  [f"{BASE_DATASETS}/animals/training_data/training_data_prep.json"],
    },
    "carcinogenesis": {
        "kb":          f"{BASE_DATASETS}/carcinogenesis/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "carcinogenesis",
        "checkpoint":  "Carcinogenesis_ranking/vocell_v_net_bootstrap_ranking_large.pt",
        "results_csv": os.path.join(RESULTS_DIR, "carcinogenesis_results.csv"),
        "lp_sources":  [
            "LPs/Carcinogenesis/lps.json",
            f"{BASE_DATASETS}/carcinogenesis/training_data/training_data_prep.json",
        ],
    },
    "family": {
        "kb":          f"{BASE_DATASETS}/family/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/family/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "family",
        "checkpoint":  "Family_ranking/vocell_v_net_bootstrap_ranking_large.pt",
        "results_csv": os.path.join(RESULTS_DIR, "family_results.csv"),
        "lp_sources":  [
            "LPs/Family/lps_difficult.json",
            f"{BASE_DATASETS}/family/training_data/training_data_prep.json",
        ],
    },
    "lymphography": {
        "kb":          f"{BASE_DATASETS}/lymphography/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/lymphography/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "lymphography",
        "checkpoint":  "Lymphography_ranking/vocell_v_net_bootstrap_ranking_large.pt",
        "results_csv": os.path.join(RESULTS_DIR, "lymphography_results.csv"),
        "lp_sources":  [f"{BASE_DATASETS}/lymphography/training_data/training_data_prep.json"],
    },
    "mutagenesis": {
        "kb":          f"{BASE_DATASETS}/mutagenesis/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/mutagenesis/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "mutagenesis",
        "checkpoint":  "Mutagenesis_ranking/vocell_v_net_bootstrap_ranking_large.pt",
        "results_csv": os.path.join(RESULTS_DIR, "mutagenesis_results.csv"),
        "lp_sources":  [
            "LPs/Mutagenesis/lps.json",
            f"{BASE_DATASETS}/mutagenesis/training_data/training_data_prep.json",
        ],
    },
    "nctrer": {
        "kb":          f"{BASE_DATASETS}/nctrer/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/nctrer/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "nctrer",
        "checkpoint":  "Nctrer_ranking/vocell_v_net_bootstrap_ranking_large.pt",
        "results_csv": os.path.join(RESULTS_DIR, "nctrer_results.csv"),
        "lp_sources":  [f"{BASE_DATASETS}/nctrer/training_data/training_data_prep.json"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  LP extraction from existing results CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_lp_names_from_csv(csv_path: str) -> list:
    """Return sorted unique LP names from a results CSV."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        names = sorted(set(row["lp_name"] for row in reader if row.get("lp_name")))
    return names


def build_temp_lp_json(lp_names: list, lp_sources: list) -> str:
    """
    Build a temporary LP JSON containing only the LPs whose names appear in
    lp_names, with their actual positive/negative examples read from lp_sources.
    Returns the path to the temp file.
    """
    # Load all LPs from source files into one merged dict
    all_lps = {}
    for src in lp_sources:
        if not os.path.exists(src):
            continue
        try:
            with open(src) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ Could not parse {src}: {e} — skipping this source.")
            continue
        problems = raw.get('problems', raw) if isinstance(raw, dict) else {}
        all_lps.update(problems)

    wanted = set(lp_names)
    problems = {name: data for name, data in all_lps.items() if name in wanted}

    # Warn about any LP names not found in the source files
    missing = wanted - set(problems)
    if missing:
        print(f"  ⚠ {len(missing)} LP(s) from results CSV not found in source files — skipping them.")
        print(f"    e.g. {sorted(missing)[:3]}")

    fd, path = tempfile.mkstemp(suffix=".json", prefix="ranking_lps_")
    with os.fdopen(fd, "w") as f:
        json.dump(problems, f)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Fuseki helpers  (identical to run_all_experiments.py)
# ─────────────────────────────────────────────────────────────────────────────

def fuseki_kill_port():
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{FUSEKI_PORT}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], check=False)
                print(f"  Killed stale process on port {FUSEKI_PORT} (pid {pid})")
        if pids:
            time.sleep(1)
    except Exception:
        pass


def fuseki_start(kb_path: str, dataset_name: str) -> subprocess.Popen:
    abs_kb = os.path.abspath(kb_path)
    if not os.path.exists(abs_kb):
        raise FileNotFoundError(f"KB not found: {abs_kb}")
    fuseki_kill_port()
    cmd = [
        os.path.abspath(FUSEKI),
        f"--port={FUSEKI_PORT}",
        f"--file={abs_kb}",
        f"/{dataset_name}",
    ]
    log_fh = open(FUSEKI_LOG, "w")
    proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
    sparql_url = f"http://localhost:{FUSEKI_PORT}/{dataset_name}/sparql"
    ping_url   = f"http://localhost:{FUSEKI_PORT}/$/ping"
    import urllib.request, urllib.error
    for _ in range(60):
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError(
                f"Fuseki exited early (rc={proc.returncode}) — check {FUSEKI_LOG}")
        try:
            urllib.request.urlopen(ping_url, timeout=2)
            req = urllib.request.Request(
                sparql_url + "?query=ASK+%7B%7D",
                headers={"Accept": "application/sparql-results+json"},
            )
            urllib.request.urlopen(req, timeout=3)
            print(f"  Fuseki up: {sparql_url}")
            return proc
        except urllib.error.HTTPError as e:
            if e.code == 400:
                print(f"  Fuseki up: {sparql_url}")
                return proc
        except Exception:
            pass
    raise RuntimeError(
        f"Fuseki did not serve {sparql_url} within 30 s — check {FUSEKI_LOG}")


def fuseki_stop(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    fuseki_kill_port()
    print("  Fuseki stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Run one vocell.py invocation
# ─────────────────────────────────────────────────────────────────────────────

def run_vocell_ranking(
    lps_file:    str,
    kb:          str,
    sparql:      str,
    embeddings:  str,
    checkpoint:  str,   # full path to the .pt file
    results_csv: str,
    num_lps:     int,
    device:      str,
    dry_run:     bool = False,
):
    cmd = [
        sys.executable, "vocell.py",
        "--lps_file",            lps_file,
        "--kb",                  kb,
        "--sparql",              sparql,
        "--embeddings",          embeddings,
        "--agg_types",           "mean",
        "--training_strategies", checkpoint,   # full .pt path passed directly
        "--num_lps",             str(num_lps),
        "--beam_width",          "10",
        "--max_concepts",        "150",
        "--time_limit",          "60",
        "--device",              device,
        "--no_baseline",
        "--allow_recursion",
        "--results_csv",         results_csv,
    ]
    print("  CMD:", " ".join(cmd))
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Vocell ranking experiments on LPs from existing results CSVs.")
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Datasets to run (default: all available).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (cpu | cuda).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    overall_ok  = []
    overall_err = []

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*70}")

        # ── Verify required files ─────────────────────────────────────────
        missing = []
        for path in [cfg["kb"], cfg["embeddings"], cfg["results_csv"]]:
            if not os.path.exists(os.path.abspath(path)):
                missing.append(path)
        if missing:
            print(f"  [SKIP] Missing files for {ds_name}:")
            for m in missing:
                print(f"    {m}")
            overall_err.append((ds_name, "missing files"))
            continue

        if not os.path.exists(cfg["checkpoint"]) and not args.dry_run:
            print(f"  [SKIP] Checkpoint not found: {cfg['checkpoint']}")
            overall_err.append((ds_name, f"no checkpoint {cfg['checkpoint']}"))
            continue

        # ── Load LP names from existing results CSV ───────────────────────
        lp_names = load_lp_names_from_csv(cfg["results_csv"])
        print(f"  LPs loaded from results CSV: {len(lp_names)}")

        # ── Build temp LP JSON ────────────────────────────────────────────
        tmp_lp_file = build_temp_lp_json(lp_names, cfg["lp_sources"])
        print(f"  Temp LP file: {tmp_lp_file}")

        # ── Start Fuseki ──────────────────────────────────────────────────
        fuseki_proc = None
        sparql_url  = f"http://localhost:{FUSEKI_PORT}/{cfg['sparql_name']}/sparql"
        if not args.dry_run:
            try:
                fuseki_proc = fuseki_start(cfg["kb"], cfg["sparql_name"])
            except Exception as e:
                print(f"  [SKIP] Could not start Fuseki for {ds_name}: {e}")
                overall_err.append((ds_name, str(e)))
                os.unlink(tmp_lp_file)
                continue

        ds_results_csv = os.path.join(OUT_DIR, f"{ds_name}_results.csv")

        try:
            ok = run_vocell_ranking(
                lps_file    = tmp_lp_file,
                kb          = cfg["kb"],
                sparql      = sparql_url,
                embeddings  = cfg["embeddings"],
                checkpoint  = cfg["checkpoint"],
                results_csv = ds_results_csv,
                num_lps     = 0,   # 0 = run all (LP list already filtered to results)
                device      = args.device,
                dry_run     = args.dry_run,
            )
            if ok:
                overall_ok.append(ds_name)
            else:
                overall_err.append((ds_name, "non-zero exit"))

        except Exception as e:
            print(f"  [ERROR] {ds_name}: {e}")
            overall_err.append((ds_name, str(e)))

        finally:
            if fuseki_proc:
                fuseki_stop(fuseki_proc)
            if os.path.exists(tmp_lp_file):
                os.unlink(tmp_lp_file)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  Successful : {len(overall_ok)}")
    for x in overall_ok:
        print(f"    ✓  {x}")
    if overall_err:
        print(f"  Failed / skipped: {len(overall_err)}")
        for name, reason in overall_err:
            print(f"    ✗  {name}  ({reason})")
    print(f"\nResults saved to: {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
