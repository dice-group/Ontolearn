import json
import unittest

from ontolearn.model_adapter import ModelAdapter
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.utils import setup_logging
from owlapy.model import OWLClass, OWLNamedIndividual, IRI

setup_logging("logging_test.conf")

NS = 'http://www.benchmark.org/family#'
PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)


class ExpressRefinement_Test(unittest.TestCase):
    def test_celoe_express(self):
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Sister')),
            OWLClass(IRI(NS, 'Daughter')),
            OWLClass(IRI(NS, 'Mother')),
            OWLClass(IRI(NS, 'Grandmother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'PersonWithASibling')),
            OWLClass(IRI(NS, 'Granddaughter')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        model = ModelAdapter(path=PATH_FAMILY,
                             refinement_operator_type=ExpressRefinement,
                             ignore=concepts_to_ignore,
                             # max_runtime=600,
                             # max_num_of_concepts_tested=10_000_000_000,
                             # iter_bound=10_000_000_000,
                             # expansionPenaltyFactor=0.01
                             )
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))

            model = model.fit(pos=typed_pos, neg=typed_neg)

            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            break


if __name__ == '__main__':
    unittest.main()
