import os
import sys

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from ontolearn.utils import setup_logging, read_individuals_file
from owlapy.owl_reasoner import FastInstanceCheckerReasoner, OntologyReasoner
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.render import ManchesterOWLSyntaxOWLObjectRenderer, DLSyntaxObjectRenderer  # noqa: F401


def run(data_file, pos_file, neg_file):
    mgr = OntologyManager()
    onto = mgr.load_ontology(IRI.create("file://" + data_file))
    base_reasoner = OntologyReasoner(onto)
    reasoner = FastInstanceCheckerReasoner(onto, base_reasoner,
                                               negation_default=True)

    kb = KnowledgeBase(ontology=onto, reasoner=reasoner)
    pos = read_individuals_file(pos_file)
    neg = read_individuals_file(neg_file)

    lp = PosNegLPStandard(pos, neg)

    pred_acc = Accuracy()
    f1 = F1()
    alg = CELOE(kb,
                max_runtime=600,
                max_num_of_concepts_tested=1_000_000)
    alg.fit(lp)
    # render = ManchesterOWLSyntaxOWLObjectRenderer()
    render = DLSyntaxObjectRenderer()
    print("solutions:")
    i = 1
    for h in alg.best_hypotheses(3):
        pred_acc_score = kb.evaluate_concept(h.concept, pred_acc, alg.encoded_learning_problem()).q
        f1_score = kb.evaluate_concept(h.concept, f1, alg.encoded_learning_problem()).q
        print(f'{i}: {render.render(h.concept)} ('
              f'pred. acc.: {pred_acc_score}, '
              f'F-Measure: {f1_score}'
              f') [Node '
              f'quality: {h.quality}, h-exp: {h.h_exp}, RC: {h.refinement_count}'
              f']')
        i += 1


def main():
    lp_dir = sys.argv[1]
    lp_path = lp_dir.split(os.sep)
    pos_file = os.sep.join((lp_dir, 'pos.txt'))
    neg_file = os.sep.join((lp_dir, 'neg.txt'))
    data_file = os.sep.join((*lp_path[:-2], 'data', lp_path[-4] + '.owl'))
    assert os.path.isfile(pos_file), "Need path to SML-Bench learning problem"
    assert os.path.isfile(data_file), "Knowledge base not found, skipping"

    setup_logging()

    run(data_file, pos_file, neg_file)


if __name__ == '__main__':
    try:
        main()
    except IndexError:
        print("Syntax:", sys.argv[0], 'path/to/learningtasks/task/owl/lp/problem')
        raise
