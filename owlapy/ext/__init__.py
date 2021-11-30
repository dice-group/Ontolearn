from abc import ABCMeta
from logging import warning
from typing import Iterable

from owlapy.model import OWLReasoner, OWLDataProperty, OWLDataRange, OWLLiteral


class OWLReasonerEx(OWLReasoner, metaclass=ABCMeta):
    """Extra convenience methods for OWL Reasoners

    (Not part of OWLAPI)"""

    # default
    def data_property_ranges(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataRange]:
        """Gets the data types that are the direct or indirect ranges of this property with respect to the imports
        closure of the root ontology.

        Args:
            pe: The property expression whose ranges are to be retrieved.
            direct: Specifies if the direct ranges should be retrieved (True), or if all ranges should be retrieved
                (False).

        Returns:
        """
        if direct:
            warning("direct not implemented")
        for ax in self.get_root_ontology().data_property_range_axioms(pe):
            yield ax.get_range()

    # default
    def all_data_property_values(self, pe: OWLDataProperty) -> Iterable[OWLLiteral]:
        """Gets all values for the given data property expression that appear in the knowledge base.

        Args:
            pe: The data property expression whose values are to be retrieved

        Returns:
            A set of OWLLiterals containing literals such that for each literal l in the set, the set of reasoner
            axioms entails DataPropertyAssertion(pe ind l) for any ind.
        """
        onto = self.get_root_ontology()
        for ind in onto.individuals_in_signature():
            for lit in self.data_property_values(ind, pe):
                yield lit
