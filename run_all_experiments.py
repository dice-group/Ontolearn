"""
run_all_experiments.py
======================
Orchestrates Vocell evaluation across all 6 datasets × 3 aggregation methods.

For each dataset:
  1. Starts a dedicated Fuseki SPARQL endpoint
  2. Runs vocell.py for each aggregation type (mean / deepsets / settransformer)
  3. Results are appended to results/{dataset}_results.csv
  4. Fuseki is stopped before moving to the next dataset

Usage:
  python run_all_experiments.py [--datasets animals carcinogenesis ...]
                                [--agg_types mean deepsets settransformer]
                                [--device cuda]
                                [--dry_run]
"""

import argparse
import os
import subprocess
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset configurations
# ─────────────────────────────────────────────────────────────────────────────
BASE_DATASETS = "../datasets"
FUSEKI        = os.path.join(
    os.path.dirname(__file__),
    "../apache-jena-fuseki-4.10.0/fuseki-server",
)
FUSEKI_LOG    = "/tmp/fuseki_experiment.log"
FUSEKI_PORT   = 3030
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")

DATASETS = {
    "animals": {
        "kb":          f"{BASE_DATASETS}/animals/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/animals/embeddings/DeCaL_entity_embeddings.csv",
        "lps":         [f"{BASE_DATASETS}/animals/training_data/training_data_prep.json"],
        "sparql_name": "animals",
        "num_lps":     150,
    },
    "carcinogenesis": {
        "kb":          f"{BASE_DATASETS}/carcinogenesis/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv",
        "lps":         [
            # "LPs/Carcinogenesis/lps.json",
            f"{BASE_DATASETS}/carcinogenesis/training_data/training_data_prep.json",
        ],
        "sparql_name": "carcinogenesis",
        "num_lps":     150,
    },
    "family": {
        "kb":          f"{BASE_DATASETS}/family/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/family/embeddings/DeCaL_entity_embeddings.csv",
        "lps":         [
            "LPs/Family/lps_difficult.json",
            f"{BASE_DATASETS}/family/training_data/training_data_prep.json",
        ],
        "sparql_name": "family",
        "num_lps":     150,
    },
    "lymphography": {
        "kb":          f"{BASE_DATASETS}/lymphography/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/lymphography/embeddings/DeCaL_entity_embeddings.csv",
        "lps":         [f"{BASE_DATASETS}/lymphography/training_data/training_data_prep.json"],
        "sparql_name": "lymphography",
        "num_lps":     150,
    },
    "mutagenesis": {
        "kb":          f"{BASE_DATASETS}/mutagenesis/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/mutagenesis/embeddings/DeCaL_entity_embeddings.csv",
        "lps":         [
            "LPs/Mutagenesis/lps.json",
            f"{BASE_DATASETS}/mutagenesis/training_data/training_data_prep.json",
        ],
        "sparql_name": "mutagenesis",
        "num_lps":     150,
    },
    "nctrer": {
        "kb":          f"{BASE_DATASETS}/nctrer/kb/ontology.owl",
        "embeddings":  f"{BASE_DATASETS}/nctrer/embeddings/DeCaL_entity_embeddings.csv",
        "lps":         [f"{BASE_DATASETS}/nctrer/training_data/training_data_prep.json"],
        "sparql_name": "nctrer",
        "num_lps":     150,
    },
}

AGG_TO_STRATEGY = {
    "mean":           "bootstrap",
    "deepsets":       "bootstrap",
    "settransformer": "bootstrap",
}

# checkpoint dir: {Dataset_title}_{agg_label}
AGG_LABEL = {
    "mean":           "mean",
    "deepsets":       "DS",
    "settransformer": "ST",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Fuseki helpers
# ─────────────────────────────────────────────────────────────────────────────

def fuseki_kill_port():
    """Kill any process already listening on FUSEKI_PORT."""
    try:
        # lsof-based kill — works on Linux/macOS
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{FUSEKI_PORT}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], check=False)
                print(f"  Killed stale process on port {FUSEKI_PORT} (pid {pid})")
        if pids:
            time.sleep(1)  # give the OS time to release the port
    except Exception:
        pass


def fuseki_start(kb_path: str, dataset_name: str) -> subprocess.Popen:
    """Kill any existing Fuseki, then start a fresh one serving kb_path at /{dataset_name}."""
    abs_kb = os.path.abspath(kb_path)
    if not os.path.exists(abs_kb):
        raise FileNotFoundError(f"KB not found: {abs_kb}")

    # Always evict whatever is on the port first
    fuseki_kill_port()

    cmd = [
        os.path.abspath(FUSEKI),
        f"--port={FUSEKI_PORT}",
        f"--file={abs_kb}",
        f"/{dataset_name}",
    ]
    log_fh = open(FUSEKI_LOG, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)

    # Wait up to 30s for THIS process to become ready, verifying the right dataset
    sparql_url = f"http://localhost:{FUSEKI_PORT}/{dataset_name}/sparql"
    ping_url   = f"http://localhost:{FUSEKI_PORT}/$/ping"
    import urllib.request, urllib.error
    for _ in range(60):
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError(f"Fuseki exited early (rc={proc.returncode}) — check {FUSEKI_LOG}")
        try:
            urllib.request.urlopen(ping_url, timeout=2)
            # Also verify the dataset endpoint responds (not a stale server)
            req = urllib.request.Request(
                sparql_url + "?query=ASK+%7B%7D",
                headers={"Accept": "application/sparql-results+json"},
            )
            urllib.request.urlopen(req, timeout=3)
            print(f"  Fuseki up: {sparql_url}")
            return proc
        except urllib.error.HTTPError as e:
            if e.code == 400:   # endpoint exists but query malformed — still up
                print(f"  Fuseki up: {sparql_url}")
                return proc
        except Exception:
            pass
    raise RuntimeError(f"Fuseki did not serve {sparql_url} within 30 s — check {FUSEKI_LOG}")


def fuseki_stop(proc: subprocess.Popen):
    """Terminate the Fuseki process and ensure the port is freed."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    # Belt-and-suspenders: kill anything still on the port
    fuseki_kill_port()
    print("  Fuseki stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Run one vocell.py invocation
# ─────────────────────────────────────────────────────────────────────────────

def run_vocell(
    lps_file:    str,
    kb:          str,
    sparql:      str,
    embeddings:  str,
    checkpoint:  str,
    agg_type:    str,
    strategy:    str,
    results_csv: str,
    num_lps:     int,
    device:      str,
    dry_run:     bool = False,
):
    cmd = [
        sys.executable, "vocell.py",
        "--lps_file",           lps_file,
        "--kb",                 kb,
        "--sparql",             sparql,
        "--embeddings",         embeddings,
        "--agg_types",          agg_type,
        "--training_strategies", strategy,
        "--checkpoint",         checkpoint,
        "--num_lps",            str(num_lps),
        "--beam_width",         "10",
        "--max_concepts",       "150",
        "--time_limit",         "60",
        "--device",             device,
        "--no_baseline",
        "--allow_recursion",
        "--results_csv",        results_csv,
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
    parser = argparse.ArgumentParser(description="Run all Vocell experiments.")
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Datasets to run (default: all).")
    parser.add_argument("--agg_types", nargs="+",
                        default=["mean", "deepsets", "settransformer"],
                        choices=["mean", "deepsets", "settransformer"],
                        help="Aggregation methods to run (default: all).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (cpu | cuda).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    overall_ok  = []
    overall_err = []

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*70}")

        # ── Verify required files exist ───────────────────────────────────
        missing = []
        for path in [cfg["kb"], cfg["embeddings"]] + cfg["lps"]:
            if not os.path.exists(os.path.abspath(path)):
                missing.append(path)
        if missing:
            print(f"  [SKIP] Missing files for {ds_name}:")
            for m in missing:
                print(f"    {m}")
            overall_err.append((ds_name, "missing files"))
            continue

        # ── Start Fuseki ──────────────────────────────────────────────────
        fuseki_proc = None
        sparql_url  = f"http://localhost:{FUSEKI_PORT}/{cfg['sparql_name']}/sparql"
        if not args.dry_run:
            try:
                fuseki_proc = fuseki_start(cfg["kb"], cfg["sparql_name"])
            except Exception as e:
                print(f"  [SKIP] Could not start Fuseki for {ds_name}: {e}")
                overall_err.append((ds_name, str(e)))
                continue

        ds_results_csv = os.path.join(RESULTS_DIR, f"{ds_name}_results.csv")

        # ── Run each LP file × agg type ───────────────────────────────────
        try:
            for lps_file in cfg["lps"]:
                lps_tag = os.path.splitext(os.path.basename(lps_file))[0]
                for agg in args.agg_types:
                    title_ds  = ds_name.capitalize()
                    ckpt_dir  = f"{title_ds}_{AGG_LABEL[agg]}"
                    strategy  = AGG_TO_STRATEGY[agg]

                    print(f"\n  LP file : {lps_file}")
                    print(f"  Agg     : {agg}  |  Checkpoint: {ckpt_dir}")

                    if not os.path.exists(ckpt_dir) and not args.dry_run:
                        print(f"  [SKIP] Checkpoint dir not found: {ckpt_dir}")
                        overall_err.append((ds_name, f"no checkpoint {ckpt_dir}"))
                        continue

                    ok = run_vocell(
                        lps_file    = lps_file,
                        kb          = cfg["kb"],
                        sparql      = sparql_url,
                        embeddings  = cfg["embeddings"],
                        checkpoint  = ckpt_dir,
                        agg_type    = agg,
                        strategy    = strategy,
                        results_csv = ds_results_csv,
                        num_lps     = cfg["num_lps"],
                        device      = args.device,
                        dry_run     = args.dry_run,
                    )
                    if ok:
                        overall_ok.append(f"{ds_name}/{lps_tag}/{agg}")
                    else:
                        overall_err.append((f"{ds_name}/{lps_tag}/{agg}", "non-zero exit"))

        except Exception as e:
            print(f"  [ERROR] Unexpected error for {ds_name}: {e}")
            overall_err.append((ds_name, str(e)))

        finally:
            if fuseki_proc:
                fuseki_stop(fuseki_proc)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  Successful runs : {len(overall_ok)}")
    for x in overall_ok:
        print(f"    ✓  {x}")
    if overall_err:
        print(f"  Failed / skipped: {len(overall_err)}")
        for name, reason in overall_err:
            print(f"    ✗  {name}  ({reason})")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
