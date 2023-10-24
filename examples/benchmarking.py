#!/usr/bin/env python3

import json
import os
import random

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric  # noqa: F401
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy
from ontolearn.owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.refinement_operators import ModifiedCELOERefinement
import json
import argparse

# from ontolearn.utils import setup_logging
# setup_logging()


random.seed(0)


def get_default_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="../KGs/Family/family-benchmark_rich_background.owl",
                        help="The path of the knowledge base")
    parser.add_argument("--path_lps", type=str, default="family_lp.json",
                        help="The path of learning problems ")
    parser.add_argument("--k", type=int, default=1,
                        help="Find top k description logic concepts")
    parser.add_argument("--runtime", type=int, default=1,
                        help="Total runtime")

    return parser.parse_args()


def loading(args):
    # (1) Load the knowledge base
    kb = KnowledgeBase(path=args.path)
    # (2) Load learning problems
    with open(args.path_lps) as json_file:
        lps = json.load(json_file)
        print(f"Number of learning problems: {len(lps)}")
    return kb, lps


def fit(knowledge_base, learning_problems, runtimes: int, topk: int):
    for str_target_concept, examples in learning_problems.items():
        # (1) Obtain learning problem.
        p = set(map(OWLNamedIndividual, map(IRI.create, examples['positive_examples'])))
        n = set(map(OWLNamedIndividual, map(IRI.create, examples['negative_examples'])))
        print(f"Target concept: {str_target_concept}\t|P|:{len(p)}\t|N|:{len(n)}")
        # (2) Initialize Learner
        model = CELOE(knowledge_base=knowledge_base,
                      max_runtime=runtimes,
                      refinement_operator=ModifiedCELOERefinement(knowledge_base=knowledge_base),
                      quality_func=Accuracy(),
                      heuristic_func=CELOEHeuristic())
        # @TODO: Add other models
        # (3) Find hypotheses best fitting the input learning problem
        lp = PosNegLPStandard(pos=p, neg=n)
        print("Fitting...", end="\t")
        model.fit(lp)
        best_fit = list(model.best_hypotheses(n=topk))[0]
        print(f"Best prediction:{best_fit}")
        # @TODO: Compute F1 score on positive and negative examples that haven't given to model

def main(args):
    kb, lps = loading(args)
    fit(knowledge_base=kb, learning_problems=lps, runtimes=args.runtime, topk=args.k)


if __name__ == '__main__':
    main(get_default_arguments())
