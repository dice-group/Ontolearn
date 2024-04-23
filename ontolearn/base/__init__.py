"""Implementations of owlapy abstract classes based on owlready2."""
from owlapy.util import move
from ontolearn.base._base import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2, \
    OWLOntology_Owlready2, BaseReasoner_Owlready2
from ontolearn.base.complex_ce_instances import OWLReasoner_Owlready2_ComplexCEInstances
from ontolearn.base.fast_instance_checker import OWLReasoner_FastInstanceChecker
move(OWLOntologyManager_Owlready2, OWLReasoner_Owlready2, OWLOntology_Owlready2, BaseReasoner_Owlready2)
__all__ = 'OWLOntologyManager_Owlready2', 'OWLReasoner_Owlready2', 'OWLOntology_Owlready2', 'BaseReasoner_Owlready2', \
    'OWLReasoner_Owlready2_ComplexCEInstances', 'OWLReasoner_FastInstanceChecker'
