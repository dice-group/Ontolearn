"""End-to-end CEL: CELOE and DRILL with symbolic vs NIR-Transformer retrieval.

Search uses either the default Ontolearn KnowledgeBase (owlready2 StructuralReasoner)
or a KnowledgeBase whose reasoner is owlapy's :class:`owlapy.owl_reasoner.NIRReasoner`.
The learned concept is always scored with the symbolic KB, so approximate retrieval
cannot grade itself.

Ontology, learning problems, pretrained encoder directory, and entity embeddings
are all passed on the command line.

Examples
--------
Download pretrained NIR encoders and DeCaL embeddings first::

    wget https://files.dice-research.org/datasets/CNIR/trained_models.zip -O ./trained_models.zip && unzip trained_models.zip

Then::

    python examples/concept_learning_nir_evaluation.py \\
        --lps LPs/Family/lps.json \\
        --kb KGs/Family/family-benchmark_rich_background.owl \\
        --nir_model trained_models/nir_pretrained_models/NIR_Transformer_family \\
        --embeddings trained_models/embeddings/family/DeCaL_entity_embeddings.csv \\
        --learners celoe --problems Aunt --max_runtime 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from owlapy import owl_expression_to_dl
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology import Ontology
from owlapy.owl_reasoner import NIRReasoner

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import CELOE, Drill
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from ontolearn.utils.static_funcs import compute_f1_score, concept_len

pd.set_option("display.precision", 5)


def make_pellet_kb(owl_path: str) -> KnowledgeBase:
    """Paper baseline: KnowledgeBase default StructuralReasoner (owlready2)."""
    return KnowledgeBase(path=owl_path)


def make_nir_kb(
    owl_path: str,
    model_path: str,
    embeddings_path: str,
    th: float,
    chunksize: int,
    symbolic_max_length: int,
) -> KnowledgeBase:
    """Ontolearn KB whose instance retrieval is owlapy :class:`NIRReasoner`."""
    onto = Ontology(IRI.create("file://" + os.path.abspath(owl_path)))
    reasoner = NIRReasoner(
        onto,
        model_path=model_path,
        embeddings_path=embeddings_path,
        th=th,
        chunksize=chunksize,
        symbolic_max_length=symbolic_max_length,
        model="transformer",
    )
    return KnowledgeBase(path=owl_path, ontology=onto, reasoner=reasoner)


def _stats_source(retrieval_kb):
    if retrieval_kb is None:
        return None
    reasoner = getattr(retrieval_kb, "reasoner", None)
    if reasoner is not None and hasattr(reasoner, "reset_retrieval_stats"):
        return reasoner
    return None


def load_problems(lp_path: str):
    with open(lp_path) as f:
        settings = json.load(f)
    if "problems" in settings:
        return settings["problems"].items(), "positive_examples", "negative_examples"
    return settings.items(), "positive examples", "negative examples"


def typed_individuals(uris):
    return {OWLNamedIndividual(IRI.create(u)) for u in uris}


def score_concept(symbolic_kb: KnowledgeBase, concept, pos, neg) -> float:
    retrieved = frozenset(symbolic_kb.individuals(concept))
    return compute_f1_score(individuals=retrieved, pos=pos, neg=neg)


def run_learner(name, learner, lp, symbolic_kb, retrieval_kb=None):
    stats_obj = _stats_source(retrieval_kb)
    if stats_obj is not None:
        stats_obj.reset_retrieval_stats()
    t0 = time.time()
    pred = learner.fit(lp).best_hypotheses(n=1)
    runtime = time.time() - t0
    f1_pellet = score_concept(symbolic_kb, pred, lp.pos, lp.neg)
    f1_search = f1_pellet
    if retrieval_kb is not None:
        f1_search = compute_f1_score(frozenset(retrieval_kb.individuals(pred)), lp.pos, lp.neg)
    n_tested = getattr(learner, "number_of_tested_concepts", None)
    try:
        length = concept_len(pred)
    except Exception:
        length = None
    stats = stats_obj.retrieval_stats() if stats_obj is not None and hasattr(stats_obj, "retrieval_stats") else {}
    return {
        f"F1-{name}": f1_pellet,
        f"F1Search-{name}": f1_search,
        f"RT-{name}": runtime,
        f"NTested-{name}": n_tested,
        f"Length-{name}": length,
        f"Prediction-{name}": owl_expression_to_dl(pred),
        f"NIRCalls-{name}": stats.get("n_retrieval_calls"),
        f"NIRCacheHits-{name}": stats.get("n_cache_hits"),
        f"NIRMeanQLen-{name}": stats.get("mean_query_length"),
        f"NIRMaxQLen-{name}": stats.get("max_query_length"),
    }


def evaluate(args) -> pd.DataFrame:
    lp_path = os.path.abspath(args.lps)
    kb_path = os.path.abspath(args.kb)
    model_path = os.path.abspath(args.nir_model)
    embeddings_path = os.path.abspath(args.embeddings)

    if not os.path.isfile(lp_path):
        raise FileNotFoundError(f"Learning problems not found: {lp_path}")
    if not os.path.isfile(kb_path):
        raise FileNotFoundError(f"Ontology not found: {kb_path}")
    if not os.path.isdir(model_path) or not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"NIR encoder directory missing or incomplete at {model_path} "
            "(need a folder with config.json)."
        )
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    print(f"LP: {lp_path}")
    print(f"KB: {kb_path}")
    print(f"NIR model: {model_path}")
    print(f"Embeddings: {embeddings_path}")

    symbolic_kb = make_pellet_kb(kb_path)
    nir_kb = make_nir_kb(
        kb_path,
        model_path=model_path,
        embeddings_path=embeddings_path,
        th=args.th,
        chunksize=args.chunksize,
        symbolic_max_length=args.symbolic_max_length,
    )

    learners = [s.strip().lower() for s in args.learners.split(",") if s.strip()]
    drill_emb = args.path_drill_embeddings
    if drill_emb:
        drill_emb = os.path.abspath(drill_emb)
    elif os.path.isfile(embeddings_path) and embeddings_path.endswith(".csv"):
        drill_emb = embeddings_path

    models = {}
    if "celoe" in learners:
        models["CELOE-Symbolic"] = (
            CELOE(knowledge_base=symbolic_kb, quality_func=F1(), max_runtime=args.max_runtime),
            None,
        )
        models["CELOE-NIR"] = (
            CELOE(knowledge_base=nir_kb, quality_func=F1(), max_runtime=args.max_runtime),
            nir_kb,
        )
    if "drill" in learners:
        models["DRILL-Symbolic"] = (
            Drill(
                knowledge_base=make_pellet_kb(kb_path),
                path_embeddings=drill_emb,
                quality_func=F1(),
                max_runtime=args.max_runtime,
                verbose=0,
            ),
            None,
        )
        models["DRILL-NIR"] = (
            Drill(
                knowledge_base=nir_kb,
                path_embeddings=drill_emb,
                quality_func=F1(),
                max_runtime=args.max_runtime,
                verbose=0,
            ),
            nir_kb,
        )

    problems, pos_key, neg_key = load_problems(lp_path)
    wanted = None
    if args.problems:
        wanted = {p.strip() for p in args.problems.split(",") if p.strip()}
    dataset_name = args.dataset or Path(kb_path).stem
    rows = []
    for target, examples in problems:
        if wanted and target not in wanted:
            continue
        print(f"\nTarget: {target}")
        lp = PosNegLPStandard(
            pos=typed_individuals(examples[pos_key]),
            neg=typed_individuals(examples[neg_key]),
        )
        row = {"Dataset": dataset_name, "LP": target}
        for label, (learner, retrieval_kb) in models.items():
            print(f"  {label} ...", end=" ", flush=True)
            metrics = run_learner(label, learner, lp, symbolic_kb, retrieval_kb)
            row.update(metrics)
            print(
                f"F1_pellet={metrics[f'F1-{label}']:.3f}  "
                f"F1_search={metrics[f'F1Search-{label}']:.3f}  "
                f"RT={metrics[f'RT-{label}']:.2f}s  "
                f"{metrics[f'Prediction-{label}']}"
            )
        rows.append(row)

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="CELOE/DRILL: symbolic KB vs NIR-Transformer retrieval")
    parser.add_argument("--lps", type=str, required=True, help="Path to learning problems JSON")
    parser.add_argument("--kb", type=str, required=True, help="Path to the ontology OWL file")
    parser.add_argument("--nir_model", type=str, required=True, help="Directory of a pretrained NIR encoder")
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Entity embeddings CSV, or a directory containing *entity_embeddings.csv",
    )
    parser.add_argument("--path_drill_embeddings", type=str, default=None,
                        help="CSV for Drill. Defaults to --embeddings when that path is a CSV")
    parser.add_argument("--dataset", type=str, default=None, help="Optional label written in the Dataset column")
    parser.add_argument("--learners", type=str, default="celoe,drill", help="Comma-separated: celoe,drill")
    parser.add_argument("--problems", type=str, default=None, help="Optional comma-separated LP names, e.g. Aunt,Brother")
    parser.add_argument("--max_runtime", type=int, default=60)
    parser.add_argument("--th", type=float, default=0.5, help="NIR membership threshold")
    parser.add_argument("--symbolic_max_length", type=int, default=1,
                        help="Use the symbolic reasoner for concepts of this length or shorter; NIR for longer ones")
    parser.add_argument("--chunksize", type=int, default=1024)
    parser.add_argument("--report", type=str, default="celoe_drill_nir.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    df = evaluate(args)
    out = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(df)
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("\nMeans:")
        print(numeric.mean())


if __name__ == "__main__":
    main()
