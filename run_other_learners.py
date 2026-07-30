"""
run_other_learners.py
=====================
Evaluates classic OWL concept learners on the same LPs used for Vocell
(read from results/ CSVs).  Follows the same patterns as
examples/concept_learning_drill2_train.py.

Learners supported:
  celoe  ocel  tdl  evolearner  alcsat  spell  nero
  drill  drillv  pruncel (needs --prunecel_jar)

Usage (one dataset per terminal for parallel runs):
  python run_other_learners.py --dataset animals        --port 3031
  python run_other_learners.py --dataset carcinogenesis --port 3032
  python run_other_learners.py --dataset family         --port 3033
  python run_other_learners.py --dataset lymphography   --port 3034
  python run_other_learners.py --dataset mutagenesis    --port 3035
  python run_other_learners.py --dataset nctrer         --port 3036

  # Or all sequentially:
  python run_other_learners.py --dataset all

Options:
  --learners celoe ocel tdl evolearner alcsat spell nero drill drillv pruncel
  --num_lps  N      cap on LPs (0 = all, default 150)
  --time_limit T    seconds per LP per learner (default 60)
  --port P          Fuseki port (default 3030, needed only by PruneCEL Java wrapper)
  --prunecel_jar    path to PruneCEL JAR (default PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from typing import Dict, List, Optional, Set

warnings.filterwarnings("ignore")

# Make local workspace copy take priority
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import CELOE, OCEL, TDL, EvoLearner, ALCSAT, SPELL, NERO
from ontolearn.learners import Drill
from ontolearn.metrics import F1
from ontolearn.heuristics import CELOEHeuristic, CeloeBasedReward
from ontolearn.utils.static_funcs import compute_f1_score
import torch

sys.path.insert(0, os.path.join(_WORKSPACE, 'examples'))
from drillv_variants import DrillV_Complex
from prunecel_wrapper import PruneCELWrapper, check_prunecel_available

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE = "../datasets"

DATASETS = {
    "animals": {
        "kb":          f"{BASE}/animals/kb/ontology.owl",
        "embeddings":  f"{BASE}/animals/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "animals",
        "lp_sources":  [f"{BASE}/animals/training_data/training_data_prep.json"],
        "results_csv": os.path.join(_WORKSPACE, "results", "animals_results.csv"),
    },
    "carcinogenesis": {
        "kb":          f"{BASE}/carcinogenesis/kb/ontology.owl",
        "embeddings":  f"{BASE}/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "carcinogenesis",
        "lp_sources":  [
            "LPs/Carcinogenesis/lps.json",
            f"{BASE}/carcinogenesis/training_data/training_data_prep.json",
        ],
        "results_csv": os.path.join(_WORKSPACE, "results", "carcinogenesis_results.csv"),
    },
    "family": {
        "kb":          f"{BASE}/family/kb/ontology.owl",
        "embeddings":  f"{BASE}/family/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "family",
        "lp_sources":  [
            "LPs/Family/lps_difficult.json",
            f"{BASE}/family/training_data/training_data_prep.json",
        ],
        "results_csv": os.path.join(_WORKSPACE, "results", "family_results.csv"),
    },
    "lymphography": {
        "kb":          f"{BASE}/lymphography/kb/ontology.owl",
        "embeddings":  f"{BASE}/lymphography/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "lymphography",
        "lp_sources":  [f"{BASE}/lymphography/training_data/training_data_prep.json"],
        "results_csv": os.path.join(_WORKSPACE, "results", "lymphography_results.csv"),
    },
    "mutagenesis": {
        "kb":          f"{BASE}/mutagenesis/kb/ontology.owl",
        "embeddings":  f"{BASE}/mutagenesis/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "mutagenesis",
        "lp_sources":  [
            "LPs/Mutagenesis/lps.json",
            f"{BASE}/mutagenesis/training_data/training_data_prep.json",
        ],
        "results_csv": os.path.join(_WORKSPACE, "results", "mutagenesis_results.csv"),
    },
    "nctrer": {
        "kb":          f"{BASE}/nctrer/kb/ontology.owl",
        "embeddings":  f"{BASE}/nctrer/embeddings/DeCaL_entity_embeddings.csv",
        "sparql_name": "nctrer",
        "lp_sources":  [f"{BASE}/nctrer/training_data/training_data_prep.json"],
        "results_csv": os.path.join(_WORKSPACE, "results", "nctrer_results.csv"),
    },
}

FUSEKI_BIN = os.path.join(_WORKSPACE, "../apache-jena-fuseki-4.10.0/fuseki-server")
FUSEKI_LOG = "/tmp/fuseki_other_learners.log"
OUT_DIR    = os.path.join(_WORKSPACE, "results_other_learners")
RENDERER   = DLSyntaxObjectRenderer()

ALL_LEARNERS = [
    "celoe", "ocel", "evolearner", "alcsat", "spell",
    "nero", "drill", "drillv", "pruncel",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Fuseki helpers  (needed only by PruneCEL Java wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def fuseki_kill_port(port: int):
    try:
        r = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                           capture_output=True, text=True)
        for pid in r.stdout.strip().split():
            if pid:
                subprocess.run(["kill", "-9", pid], check=False)
                print(f"  Killed stale PID {pid} on port {port}")
        if r.stdout.strip():
            time.sleep(1)
    except Exception:
        pass


def fuseki_start(kb_path: str, dataset_name: str, port: int) -> subprocess.Popen:
    abs_kb = os.path.abspath(kb_path)
    if not os.path.exists(abs_kb):
        raise FileNotFoundError(f"KB not found: {abs_kb}")
    fuseki_kill_port(port)
    cmd = [os.path.abspath(FUSEKI_BIN), f"--port={port}",
           f"--file={abs_kb}", f"/{dataset_name}"]
    log_fh = open(FUSEKI_LOG, "w")
    proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
    import urllib.request, urllib.error
    sparql = f"http://localhost:{port}/{dataset_name}/sparql"
    ping   = f"http://localhost:{port}/$/ping"

    # Phase 1: wait for Fuseki to accept connections at all (ping)
    ping_ok = False
    for _ in range(240):  # up to 120 s
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError(f"Fuseki exited early — check {FUSEKI_LOG}")
        try:
            urllib.request.urlopen(ping, timeout=2)
            ping_ok = True
            break
        except Exception:
            pass
    if not ping_ok:
        raise RuntimeError(f"Fuseki ping did not respond within 120 s")

    # Phase 2: wait for the dataset endpoint to finish indexing
    for _ in range(120):  # up to 60 s more
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError(f"Fuseki exited early — check {FUSEKI_LOG}")
        try:
            req = urllib.request.Request(
                sparql + "?query=ASK+%7B%7D",
                headers={"Accept": "application/sparql-results+json"})
            urllib.request.urlopen(req, timeout=3)
            print(f"  Fuseki up: {sparql}")
            return proc
        except urllib.error.HTTPError as e:
            if e.code == 400:
                print(f"  Fuseki up: {sparql}")
                return proc
        except Exception:
            pass
    raise RuntimeError(f"Fuseki dataset endpoint {sparql} not ready within 180 s")


def fuseki_stop(proc: Optional[subprocess.Popen], port: int):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    fuseki_kill_port(port)


# ─────────────────────────────────────────────────────────────────────────────
#  LP loading
# ─────────────────────────────────────────────────────────────────────────────

def load_lp_names_from_results(csv_path: str, num_lps: int = 0) -> List[str]:
    with open(csv_path, newline="") as f:
        names = sorted(set(r["lp_name"] for r in csv.DictReader(f)
                           if r.get("lp_name")))
    return names[:num_lps] if num_lps > 0 else names


def load_lps_from_sources(lp_names: List[str], lp_sources: List[str]) -> Dict:
    all_lps: Dict = {}
    for src in lp_sources:
        if not os.path.exists(src):
            continue
        try:
            with open(src) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ Cannot parse {src}: {e} — skipping")
            continue
        problems = raw.get("problems", raw) if isinstance(raw, dict) else {}
        all_lps.update(problems)
    wanted  = set(lp_names)
    found   = {n: d for n, d in all_lps.items() if n in wanted}
    missing = wanted - set(found)
    if missing:
        print(f"  ⚠ {len(missing)} LPs not found in source files "
              f"(e.g. {sorted(missing)[:3]})")
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  Individual resolution — mirrors concept_learning_drill2_train.py
# ─────────────────────────────────────────────────────────────────────────────

def build_ind_index(kb: KnowledgeBase):
    by_iri, by_local = {}, {}
    for ind in kb.individuals():
        by_iri[str(ind.iri)]              = ind
        by_local[ind.iri.get_remainder()] = ind
    return by_iri, by_local


def resolve_ind(iri_str: str, by_iri: Dict, by_local: Dict) -> OWLNamedIndividual:
    if iri_str in by_iri:
        return by_iri[iri_str]
    local = IRI.create(iri_str).get_remainder()
    return by_local.get(local, OWLNamedIndividual(IRI.create(iri_str)))


# ─────────────────────────────────────────────────────────────────────────────
#  Namespace-safe individual retrieval  (mirrors _safe_kb_individuals)
# ─────────────────────────────────────────────────────────────────────────────

def safe_kb_individuals(kb: KnowledgeBase, pred):
    try:
        return kb.individuals(pred)
    except (AttributeError, TypeError):
        pass
    cls_by_local: dict = {}
    for c in kb.ontology.classes_in_signature():
        cls_by_local[c.iri.get_remainder()] = c

    def _remap(ce):
        if isinstance(ce, OWLClass):
            return cls_by_local.get(ce.iri.get_remainder(), ce)
        if hasattr(ce, 'operands'):
            return type(ce)([_remap(o) for o in ce.operands()])
        if hasattr(ce, 'get_filler'):
            return type(ce)(ce.get_property(), _remap(ce.get_filler()))
        if hasattr(ce, 'get_operand'):
            return type(ce)(_remap(ce.get_operand()))
        return ce

    try:
        return kb.individuals(_remap(pred))
    except Exception:
        return frozenset()


# ─────────────────────────────────────────────────────────────────────────────
#  NERO: patch search() to move index tensors onto the model's device
# ─────────────────────────────────────────────────────────────────────────────

def patch_nero_device(nero_instance):
    """Monkey-patch NERO.search so idx_pos/idx_neg are created on model.device."""
    import types
    import ontolearn.learners.nero as _nero_mod
    orig_search = nero_instance.__class__.search   # unbound

    def _patched_search(self, pos, neg, **kwargs):
        _orig = _nero_mod.torch.LongTensor
        dev   = self.device
        _nero_mod.torch.LongTensor = lambda data: _orig(data).to(dev)
        try:
            return orig_search(self, pos=pos, neg=neg, **kwargs)
        finally:
            _nero_mod.torch.LongTensor = _orig

    nero_instance.search = types.MethodType(_patched_search, nero_instance)


# ─────────────────────────────────────────────────────────────────────────────
#  Predict helper — identical pattern to concept_learning_drill2_train.py
# ─────────────────────────────────────────────────────────────────────────────

def time_predict(learner, lp: PosNegLPStandard, kb: KnowledgeBase) -> dict:
    t0 = time.time()
    # SPELL and ALCSAT expose best_hypothesis() (singular, no args).
    # All search-based learners (CELOE, OCEL, TDL, EvoLearner, NERO, Drill…)
    # expose best_hypotheses() which returns a concept directly for n=1.
    if hasattr(learner, 'best_hypothesis') and not hasattr(learner, 'best_hypotheses'):
        pred = learner.fit(lp).best_hypothesis()
    else:
        pred = learner.fit(lp).best_hypotheses()
    elapsed = time.time() - t0

    f1 = compute_f1_score(
        individuals=frozenset(safe_kb_individuals(kb, pred)),
        pos=lp.pos,
        neg=lp.neg,
    )

    try:
        concept_str = RENDERER.render(pred)
    except Exception:
        concept_str = str(pred)

    # Concepts-tested counter (SAT-based learners have no concept-by-concept search)
    if hasattr(learner, 'max_concept_size') and not hasattr(learner, 'search_tree'):
        concepts_tested = None   # ALCSAT
    elif hasattr(learner, 'max_query_size'):
        concepts_tested = None   # SPELL
    else:
        concepts_tested = getattr(learner, '_number_of_tested_concepts', 0)

    return {
        "f1":              f1,
        "runtime_s":       elapsed,
        "best_concept":    concept_str,
        "concepts_tested": concepts_tested,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CSV writer
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = ["dataset", "lp_name", "learner", "f1",
              "runtime_s", "concepts_tested", "best_concept"]


def append_result(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_done(csv_path: str) -> Set[tuple]:
    """Return set of (lp_name, learner) pairs already written to the CSV."""
    done: Set[tuple] = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((r.get("lp_name", ""), r.get("learner", "")))
    return done


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run classic OWL learners on Vocell LP sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--learners", nargs="+",
                        default=ALL_LEARNERS,
                        choices=ALL_LEARNERS + ["all"],
                        metavar="LEARNER",
                        help=f"Learners to evaluate. Available: {ALL_LEARNERS}")
    parser.add_argument("--num_lps",    type=int,   default=150,
                        help="Max LPs per dataset (0 = all).")
    parser.add_argument("--time_limit", type=float, default=20.0,
                        help="Time budget per LP per learner (seconds).")
    parser.add_argument("--port",       type=int,   default=3030,
                        help="Fuseki port (needed only by PruneCEL Java wrapper).")
    parser.add_argument("--prunecel_jar", type=str,
                        default=os.path.join(_WORKSPACE, "PruneCEL", "target",
                                             "prune-cel-0.0.1-SNAPSHOT.jar"),
                        help="Path to compiled PruneCEL JAR.")
    parser.add_argument("--drill_train_lps", type=int, default=20,
                        help="Number of training LPs for Drill/DrillV (per dataset).")
    parser.add_argument("--drill_episodes", type=int, default=2,
                        help="Drill/DrillV training episodes.")
    args = parser.parse_args()

    if "all" in args.learners:
        args.learners = ALL_LEARNERS

    os.makedirs(OUT_DIR, exist_ok=True)

    ds_names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds_name in ds_names:
        cfg       = DATASETS[ds_name]
        port      = args.port
        int_limit = int(args.time_limit)

        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name.upper()}  (port {port})")
        print(f"{'='*70}")

        # ── Sanity checks ─────────────────────────────────────────────────
        if not os.path.exists(cfg["kb"]):
            print(f"  [SKIP] KB not found: {cfg['kb']}"); continue
        if not os.path.exists(cfg["results_csv"]):
            print(f"  [SKIP] Vocell results CSV not found: {cfg['results_csv']}"); continue

        # ── Load LP names & examples ──────────────────────────────────────
        lp_names = load_lp_names_from_results(cfg["results_csv"], args.num_lps)
        print(f"  LPs to evaluate: {len(lp_names)}")
        lps_dict = load_lps_from_sources(lp_names, cfg["lp_sources"])
        if not lps_dict:
            print("  [SKIP] No LPs resolved."); continue

        # ── Load KB ───────────────────────────────────────────────────────
        print("  Loading KB…")
        kb = KnowledgeBase(path=os.path.abspath(cfg["kb"]))
        by_iri, by_local = build_ind_index(kb)

        # ── Build learner instances ───────────────────────────────────────
        learners: Dict = {}

        if "celoe" in args.learners:
            learners["celoe"] = CELOE(
                knowledge_base=kb, quality_func=F1(),
                heuristic_func=CELOEHeuristic(
                    expansionPenaltyFactor=0.05,
                    startNodeBonus=1.0,
                    nodeRefinementPenalty=0.01),
                max_runtime=int_limit,
                max_num_of_concepts_tested=10_000)

        if "ocel" in args.learners:
            learners["ocel"] = OCEL(
                knowledge_base=kb, quality_func=F1(),
                max_runtime=int_limit,
                max_num_of_concepts_tested=10_000)

        if "tdl" in args.learners:
            learners["tdl"] = TDL(
                knowledge_base=kb,
                use_nominals=False,
                max_runtime=int_limit)

        if "evolearner" in args.learners:
            learners["evolearner"] = EvoLearner(
                knowledge_base=kb,
                use_card_restrictions=False,
                use_data_properties=False,
                max_runtime=int_limit)

        if "alcsat" in args.learners:
            learners["alcsat"] = ALCSAT(
                knowledge_base=kb,
                max_runtime=int_limit,
                max_concept_size=30)

        if "spell" in args.learners:
            learners["spell"] = SPELL(
                knowledge_base=kb,
                max_runtime=int_limit,
                max_query_size=10,
                search_mode="full_approx")

        if "nero" in args.learners:
            _nero = NERO(
                knowledge_base=kb,
                num_embedding_dim=128,
                neural_architecture="DeepSet",
                learning_rate=0.001,
                num_epochs=50,
                batch_size=32)
            patch_nero_device(_nero)
            learners["nero"] = _nero

        # ── Drill / DrillV (need embeddings + one-time training) ──────────
        emb_path = os.path.abspath(cfg["embeddings"])
        drill_ok = os.path.exists(emb_path)
        if not drill_ok and any(n in args.learners for n in ("drill", "drillv")):
            print(f"  ⚠ Embeddings not found ({emb_path}) — Drill/DrillV skipped")

        if "drill" in args.learners and drill_ok:
            learners["drill"] = Drill(
                knowledge_base=kb,
                path_embeddings=emb_path,
                refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                quality_func=F1(),
                reward_func=CeloeBasedReward(),
                epsilon_decay=1.0,
                learning_rate=0.01,
                verbose=0,
                num_of_sequential_actions=1,
                num_episode=args.drill_episodes,
                iter_bound=10_000,
                max_runtime=int_limit)

        if "drillv" in args.learners and drill_ok:
            learners["drillv"] = DrillV_Complex(
                knowledge_base=kb,
                path_embeddings=emb_path,
                refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                quality_func=F1(),
                reward_func=CeloeBasedReward(),
                epsilon_decay=1.0,
                learning_rate=0.01,
                verbose=0,
                num_of_sequential_actions=1,
                num_episode=args.drill_episodes,
                iter_bound=10_000,
                max_runtime=int_limit)

        # Train Drill / DrillV once per dataset before the LP loop.
        # Hard cap: 2 minutes — if training hasn't finished by then, the thread
        # is abandoned and we continue with the partially-trained model.
        DRILL_TRAIN_TIMEOUT = 120
        for name in ["drill", "drillv"]:
            if name not in learners:
                continue
            import threading
            print(f"  Training {name.upper()} (max {DRILL_TRAIN_TIMEOUT}s)…")
            train_exc = [None]

            def _train(learner=learners[name], exc=train_exc):
                try:
                    learner.train(
                        num_of_target_concepts=1,
                        num_learning_problems=args.drill_train_lps)
                except Exception as e:
                    exc[0] = e

            t = threading.Thread(target=_train, daemon=True)
            t.start()
            t.join(timeout=DRILL_TRAIN_TIMEOUT)
            if t.is_alive():
                print(f"  ⚠ {name.upper()} training timed out after "
                      f"{DRILL_TRAIN_TIMEOUT}s — continuing with partial model")
            elif train_exc[0] is not None:
                print(f"  ⚠ {name.upper()} training failed: {train_exc[0]} — skipping")
                del learners[name]
            else:
                print(f"  ✓ {name.upper()} trained")

        # ── PruneCEL Java wrapper (needs JAR + Fuseki) ────────────────────
        fuseki_proc     = None
        pruncel_learner = None
        if "pruncel" in args.learners:
            jar = os.path.abspath(args.prunecel_jar)
            if not check_prunecel_available(jar):
                print(f"  ⚠ PruneCEL JAR not found ({jar}) — pruncel skipped")
            else:
                try:
                    print("  Starting Fuseki for PruneCEL…")
                    fuseki_proc = fuseki_start(cfg["kb"], cfg["sparql_name"], port)
                    sparql_url  = (f"http://localhost:{port}"
                                   f"/{cfg['sparql_name']}/sparql")
                    pruncel_learner = PruneCELWrapper(
                        jar_path=jar,
                        sparql_url=sparql_url,
                        knowledge_base=kb,
                        max_runtime=int_limit)
                    print("  ✓ PruneCEL initialized")
                except Exception as e:
                    print(f"  ⚠ PruneCEL init failed: {e}")

        out_csv = os.path.join(OUT_DIR, f"{ds_name}_results.csv")
        done    = load_done(out_csv)

        # ── Evaluation loop ───────────────────────────────────────────────
        try:
            total = len(lps_dict)
            for lp_idx, (lp_name, lp_data) in enumerate(lps_dict.items(), 1):
                typed_pos = set(
                    resolve_ind(s, by_iri, by_local)
                    for s in lp_data.get("positive_examples", []))
                typed_neg = set(
                    resolve_ind(s, by_iri, by_local)
                    for s in lp_data.get("negative_examples", []))
                if not typed_pos:
                    print(f"  [{lp_idx}/{total}] SKIP {lp_name[:60]} (no pos)")
                    continue

                lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
                print(f"\n  [{lp_idx}/{total}] {lp_name[:70]}  "
                      f"(pos={len(typed_pos)}, neg={len(typed_neg)})")

                # ── Standard learners ─────────────────────────────────────
                for name, learner in learners.items():
                    if (lp_name, name) in done:
                        print(f"    {name:<14} already done — skip")
                        continue

                    # NERO needs a namespace hint for IRI construction
                    if name == "nero":
                        try:
                            a_prop = next(iter(
                                kb.ontology.object_properties_in_signature()))
                            learner.ns = a_prop.iri.get_namespace()
                        except StopIteration:
                            pass

                    try:
                        r = time_predict(learner, lp, kb)
                    except Exception as e:
                        print(f"    {name:<14} ERROR: {e}")
                        r = {"f1": 0.0, "runtime_s": 0.0,
                             "best_concept": "ERROR", "concepts_tested": None}

                    print(f"    {name:<14} F1={r['f1']:.3f}  "
                          f"RT={r['runtime_s']:.1f}s  "
                          f"{r['best_concept'][:50]}")
                    append_result(out_csv, {
                        "dataset":         ds_name,
                        "lp_name":         lp_name,
                        "learner":         name,
                        "f1":              round(r["f1"], 4),
                        "runtime_s":       round(r["runtime_s"], 2),
                        "concepts_tested": r["concepts_tested"],
                        "best_concept":    r["best_concept"],
                    })

                # ── PruneCEL Java ─────────────────────────────────────────
                if pruncel_learner and (lp_name, "pruncel") not in done:
                    t0 = time.time()
                    try:
                        pred = pruncel_learner.fit(lp).best_hypothesis()
                        f1   = compute_f1_score(
                            individuals=frozenset(safe_kb_individuals(kb, pred)),
                            pos=typed_pos, neg=typed_neg)
                        try:
                            concept_str = RENDERER.render(pred)
                        except Exception:
                            concept_str = str(pred)
                    except Exception as e:
                        print(f"    {'pruncel':<14} ERROR: {e}")
                        f1, concept_str = 0.0, "ERROR"
                    rt = time.time() - t0
                    print(f"    {'pruncel':<14} F1={f1:.3f}  RT={rt:.1f}s  "
                          f"{concept_str[:50]}")
                    append_result(out_csv, {
                        "dataset":         ds_name,
                        "lp_name":         lp_name,
                        "learner":         "pruncel",
                        "f1":              round(f1, 4),
                        "runtime_s":       round(rt, 2),
                        "concepts_tested": getattr(
                            pruncel_learner, '_number_of_tested_concepts', None),
                        "best_concept":    concept_str,
                    })

        finally:
            if fuseki_proc:
                fuseki_stop(fuseki_proc, port)

        print(f"\n  Results saved → {out_csv}")

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
