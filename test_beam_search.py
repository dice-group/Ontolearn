"""Test that the refinement fix generates ∃ married.Brother"""
import json
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import PruneCELBasedRefinement
from owlapy.owl_property import OWLObjectProperty
from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLThing, OWLClass
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from typing import FrozenSet, Tuple, Dict
from owlapy.class_expression import *
from owlapy.render import DLSyntaxObjectRenderer
import time

# Load KB
kb = KnowledgeBase(path='KGs/Family/family.owl') #kb = KnowledgeBase(path='KGs/Family/family-benchmark_rich_background.owl')

renderer = DLSyntaxObjectRenderer()

# Load Aunt problem
with open('LPs/Family/lps.json', 'r') as f:
    lps = json.load(f)

problem = lps['problems']['Aunt']
pos = frozenset({OWLNamedIndividual(IRI.create(iri)) for iri in problem['positive_examples']})
neg = frozenset({OWLNamedIndividual(IRI.create(iri)) for iri in problem['negative_examples']})



def compute_f1(concept: OWLClassExpression, kb: KnowledgeBase, 
               pos: FrozenSet, neg: FrozenSet) -> Tuple[float, float, float, int, int]:
    """Compute F1, precision, recall, and coverage counts for a concept."""
    try:
        # print(concept)
        instances = kb.individuals_set(concept)
        tp = len(instances.intersection(pos))
        fp = len(instances.intersection(neg))
        fn = len(pos - instances)
    except Exception as e:
        print(f"Error computing F1 for {renderer.render(concept)}: {e}")
        exit(0)
        return 0.0, 0, 0, 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, len(instances)
    # except Exception as e:
    #     print(f"Error computing F1 for {renderer.render(concept)}: {e}")
    #     exit(0)
    #     return 0.0, 0, 0, 0

# Create operator (KB-based, no SPARQL)
operator = PruneCELBasedRefinement(
    knowledge_base=kb,
    sparql_endpoint='http://localhost:3030/family/sparql'
)
operator.set_input_examples(pos, neg)

# Test refining ∃ married.⊤
prop = OWLObjectProperty(IRI.create('http://www.benchmark.org/family#married'))
exist = OWLObjectSomeValuesFrom(property=prop, filler=OWLThing)#OWLClass(IRI.create('http://www.benchmark.org/family#Male')))
# exist = OWLClass(IRI.create('http://www.benchmark.org/family#Female'))
# exist = OWLObjectSomeValuesFrom(property=prop, filler=OWLClass(IRI.create('http://www.benchmark.org/family#Male')))
exist = OWLObjectIntersectionOf([OWLClass(IRI.create('http://www.benchmark.org/family#Sister')), OWLThing])
exist = OWLObjectSomeValuesFrom(property=prop, filler=OWLThing)

exist = OWLObjectAllValuesFrom(property=OWLObjectProperty(IRI.create('http://www.benchmark.org/family#hasSibling')), filler=exist)
# exist = OWLObjectSomeValuesFrom(property=prop, filler=object)
# exist = OWLThing
# exist = OWLObjectComplementOf(OWLThing)

# exist = OWLClass(IRI.create('http://www.benchmark.org/family#Female'))
#Mutagenesis
# prop = OWLObjectProperty(IRI.create('http://dl-learner.org/mutagenesis#hasAtom'))
exist = OWLObjectSomeValuesFrom(property=prop, filler=OWLThing)
# exist = OWLThing

print(f'Refining: {renderer.render(exist)}')
print('='*60)

refinements = list(operator.refine(exist))
print(f'Total refinements: {len(refinements)}')
# exit(0)
max_time = 60  # seconds
start_time = time.time()

min_fp = 100

# while time.time() - start_time < max_time:
for ref in refinements:
    f1, tp, fp, coverage = compute_f1(ref, kb, pos, neg)
    print(f'refinement: {renderer.render(ref)} | F1: {f1:.3f}, prec.: {tp}, Recall: {fp}, Cov: {coverage}')

    # if f1 > 0.6:
    #     A = operator.refine(ref)
    #     print(f'Refining concept: {renderer.render(ref)} further')
    #     for a in A:
    #         f1_a, tp_a, fp_a, coverage_a = compute_f1(a, kb, pos, neg)
    #         if f1_a > f1:
    #             print(f'  Found better refinement: {renderer.render(a)} | F1: {f1_a:.3f}, prec.: {tp_a}, Recall: {fp_a}, Cov: {coverage_a}')
    #     # print(f'{renderer.render(ref)} | F1: {f1:.3f}, prec.: {tp}, Recall: {fp}, Cov: {coverage}')
