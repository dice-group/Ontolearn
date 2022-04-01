""" Test the concept module"""
import json
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.utils import setup_logging
from owlapy.model import OWLClass, IRI
from owlapy.owlready2 import OWLReasoner_Owlready2

setup_logging("ontolearn/logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
kb = KnowledgeBase(path=PATH_FAMILY, reasoner_factory=OWLReasoner_Owlready2)


def test_concept():
    # Processes input kb
    iri = kb.ontology().get_ontology_id().get_ontology_iri()
    assert iri == IRI.create("http://www.benchmark.org/family")
    classes = list(kb.ontology().classes_in_signature())
    assert len(classes) >= 18
    for cls in kb.ontology().classes_in_signature():
        assert isinstance(cls, OWLClass)
        ic = kb.individuals_count(cls)
        assert ic > 0
        inds = kb.individuals_set(cls)
        assert inds.issubset(kb.all_individuals_set())


if __name__ == '__main__':
    test_concept()
