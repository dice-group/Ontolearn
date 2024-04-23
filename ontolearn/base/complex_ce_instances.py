"""OWL Reasoner - Complex Class Expression Instances (CCEI)."""
import logging
import types
from logging import warning
from typing import Iterable, cast, Optional, List
import os
import owlready2
from owlapy.class_expression import OWLClassExpression
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLAxiom
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.base import OWLReasoner_Owlready2, OWLOntology_Owlready2, BaseReasoner_Owlready2, \
    OWLOntologyManager_Owlready2
from ontolearn.base.utils import ToOwlready2

logger = logging.getLogger(__name__)


class OWLReasoner_Owlready2_ComplexCEInstances(OWLReasoner_Owlready2):
    __slots__ = '_cnt', '_conv', '_base_reasoner'

    _conv: ToOwlready2
    _base_reasoner: BaseReasoner_Owlready2

    def __init__(self, ontology: OWLOntology_Owlready2, base_reasoner: Optional[BaseReasoner_Owlready2] = None,
                 infer_property_values: bool = True, infer_data_property_values: bool = True, isolate: bool = False):
        """
        OWL Reasoner with support for Complex Class Expression Instances + sync_reasoner.

        Args:
            ontology: The ontology that should be used by the reasoner.
            base_reasoner: Set to BaseReasoner.PELLET (default) or BaseReasoner.HERMIT.
            infer_property_values: Whether to infer property values.
            infer_data_property_values: Whether to infer data property values (only for PELLET).
            isolate: Whether to isolate the reasoner in a new world + copy of the original ontology.
                     Useful if you create multiple reasoner instances in the same script.
        """

        super().__init__(ontology, isolate)
        if isolate:
            new_manager = OWLOntologyManager_Owlready2()
            self.reference_ontology = new_manager.load_ontology(ontology.get_original_iri())
            self.reference_iri = IRI.create(f'file:/isolated_ontology_{id(self.reference_ontology)}.owl')
            new_manager.save_ontology(self.reference_ontology, self.reference_iri)

        self._cnt = 1
        self._conv = ToOwlready2(world=self._world)
        self._base_reasoner = base_reasoner
        self._sync_reasoner(self._base_reasoner, infer_property_values, infer_data_property_values)
        self.infer_property_values = infer_property_values
        self.infer_data_property_values = infer_data_property_values

    def update_isolated_ontology(self, axioms_to_add: List[OWLAxiom] = None,
                                 axioms_to_remove: List[OWLAxiom] = None):
        if self._isolated:
            if axioms_to_add is None and axioms_to_remove is None:
                raise ValueError(f"At least one argument should be specified in method: "
                                 f"{self.update_isolated_ontology.__name__}")
            self._ontology = self.reference_ontology
            super().update_isolated_ontology(axioms_to_add, axioms_to_remove)
            self.reference_ontology.get_owl_ontology_manager().save_ontology(self._ontology, self.reference_iri)
            new_manager = OWLOntologyManager_Owlready2()
            self._ontology = new_manager.load_ontology(IRI.create(f'file://isolated_ontology_'
                                                                  f'{id(self.reference_ontology)}.owl'))
            self._world = new_manager._world
            self._sync_reasoner(self._base_reasoner, self.infer_property_values, self.infer_data_property_values)
        else:
            raise AssertionError(f"Misuse of method '{self.update_isolated_ontology.__name__}'. The reasoner is not "
                                 f"using an isolated ontology.")

    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        if direct:
            warning("direct not implemented")
        with self._world.get_ontology("http://temp.classes/"):
            temp_pred = cast(owlready2.ThingClass, types.new_class("TempCls%d" % self._cnt, (owlready2.owl.Thing,)))
            temp_pred.equivalent_to = [self._conv.map_concept(ce)]
            if self._base_reasoner == BaseReasoner_Owlready2.HERMIT:
                owlready2.sync_reasoner_hermit(self._world.get_ontology("http://temp.classes/"),
                                               self.infer_property_values)
            else:
                owlready2.sync_reasoner_pellet(self._world.get_ontology("http://temp.classes/"),
                                               self.infer_property_values, self.infer_data_property_values)
        instances = list(temp_pred.instances(world=self._world))
        temp_pred.equivalent_to = []
        try:
            owlready2.destroy_entity(temp_pred)
        except AttributeError as e:
            logger.info(f"AttributeError: {e} Source: {__file__} (you can ignore this)")
        self._cnt += 1
        for i in instances:
            yield OWLNamedIndividual(IRI.create(i.iri))

    def __del__(self):
        if self._isolated:
            file_path = f"isolated_ontology_{id(self.reference_ontology)}.owl"
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Error deleting {file_path}: {e}")
