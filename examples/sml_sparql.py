import os
import sys

from ontolearn.concept_learner import CELOE
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.sparqlkb import SparqlKnowledgeBase
from ontolearn.utils import setup_logging, read_individuals_file
from owlapy.render import ManchesterOWLSyntaxOWLObjectRenderer, DLSyntaxObjectRenderer  # noqa: F401

ENDPOINT_URL = "http://172.17.0.2:3030/ds/query"
# ENDPOINT_URL = "http://172.18.0.2:7200/repositories/carcinogenesis"


async def run_async(data_file, endpoint_url, pos_file, neg_file):
    kb = SparqlKnowledgeBase(data_file, endpoint_url)
    pos = read_individuals_file(pos_file)
    neg = read_individuals_file(neg_file)

    lp = PosNegLPStandard(pos, neg)

    op = ModifiedCELOERefinement(kb,
                                 use_negation=False,
                                 use_inverse=False,
                                 use_card_restrictions=False,
                                 use_numeric_datatypes=False,
                                 use_boolean_datatype=False,
                                 use_time_datatypes=False)

    pred_acc = Accuracy()
    f1 = F1()
    alg = CELOE(kb,
                refinement_operator=op,
                max_runtime=60,
                iter_bound=1_000_000,
                max_num_of_concepts_tested=1_000_000)
    await alg.fit_async(lp)
    await kb.async_client.aclose()
    # render = ManchesterOWLSyntaxOWLObjectRenderer()
    render = DLSyntaxObjectRenderer()
    print("solutions:")
    i = 1
    for h in alg.best_hypotheses(3):
        # individuals_set = kb.individuals_set(h.concept)
        print(f'{i}: {render.render(h.concept)} ('
              f'pred. acc.: {kb.evaluate_concept(h.concept, pred_acc, alg.encoded_learning_problem()).q}, '
              f'F-Measure: {kb.evaluate_concept(h.concept, f1, alg.encoded_learning_problem()).q}'
              f') [Node '
              f'quality: {h.quality}, h-exp: {h.h_exp}, RC: {h.refinement_count}'
              f']')
        i += 1
    print(f'#tested concepts: {alg.number_of_tested_concepts}')


async def main_async():
    lp_dir = sys.argv[1]
    lp_path = lp_dir.split(os.sep)
    pos_file = os.sep.join((lp_dir, 'pos.txt'))
    neg_file = os.sep.join((lp_dir, 'neg.txt'))
    data_file = os.sep.join((*lp_path[:-2], 'data', lp_path[-4] + '.owl'))
    assert os.path.isfile(pos_file), "Need path to SML-Bench learning problem"
    assert os.path.isfile(data_file), "Knowledge base not found, skipping"

    setup_logging("logging_tentris.conf")

    await run_async(data_file, ENDPOINT_URL, pos_file, neg_file)


def run(data_file, endpoint_url, pos_file, neg_file):
    kb = SparqlKnowledgeBase(data_file, endpoint_url)
    pos = read_individuals_file(pos_file)
    neg = read_individuals_file(neg_file)

    lp = PosNegLPStandard(pos, neg)

    op = ModifiedCELOERefinement(kb,
                                 use_negation=False,
                                 use_inverse=False,
                                 use_card_restrictions=False,
                                 use_numeric_datatypes=False,
                                 use_boolean_datatype=False,
                                 use_time_datatypes=False)

    pred_acc = Accuracy()
    f1 = F1()
    alg = CELOE(kb,
                refinement_operator=op,
                max_runtime=60,
                iter_bound=1_000_000,
                max_num_of_concepts_tested=1_000_000)
    alg.fit(lp)
    # render = ManchesterOWLSyntaxOWLObjectRenderer()
    render = DLSyntaxObjectRenderer()
    print("solutions:")
    i = 1
    for h in alg.best_hypotheses(3):
        # individuals_set = kb.individuals_set(h.concept)
        print(f'{i}: {render.render(h.concept)} ('
              f'pred. acc.: {kb.evaluate_concept(h.concept, pred_acc, alg._learning_problem).q}, '
              f'F-Measure: {kb.evaluate_concept(h.concept, f1, alg._learning_problem).q}'
              f') [Node '
              f'quality: {h.quality}, h-exp: {h.h_exp}, RC: {h.refinement_count}'
              f']')
        i += 1
    print(f'#tested concepts: {alg.number_of_tested_concepts}')


def main():
    lp_dir = sys.argv[1]
    lp_path = lp_dir.split(os.sep)
    pos_file = os.sep.join((lp_dir, 'pos.txt'))
    neg_file = os.sep.join((lp_dir, 'neg.txt'))
    data_file = os.sep.join((*lp_path[:-2], 'data', lp_path[-4] + '.owl'))
    assert os.path.isfile(pos_file), "Need path to SML-Bench learning problem"
    assert os.path.isfile(data_file), "Knowledge base not found, skipping"

    setup_logging("logging_tentris.conf")

    run(data_file, ENDPOINT_URL, pos_file, neg_file)


if __name__ == '__main__':
    try:
        # main()
        import asyncio
        asyncio.run(main_async(), debug=True)
    except IndexError:
        print("Syntax:", sys.argv[0], 'path/to/learningtasks/task/owl/lp/problem')
        raise
