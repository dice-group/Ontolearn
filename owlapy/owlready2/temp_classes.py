import types
from logging import warning
from typing import Iterable, cast, Optional

import owlready2

from owlapy import IRI
from owlapy.model import OWLClass, OWLClassExpression, OWLNamedIndividual
from owlapy.owlready2 import OWLReasoner_Owlready2, OWLOntology_Owlready2, BaseReasoner_Owlready2
from owlapy.owlready2.utils import ToOwlready2


class OWLReasoner_Owlready2_TempClasses(OWLReasoner_Owlready2):
    __slots__ = '_cnt', '_conv', '_base_reasoner'

    _conv: ToOwlready2
    _base_reasoner: BaseReasoner_Owlready2

    def __init__(self, ontology: OWLOntology_Owlready2, base_reasoner: Optional[BaseReasoner_Owlready2] = None):
        super().__init__(ontology)
        self._cnt = 1
        self._conv = ToOwlready2(world=self._world)
        self._base_reasoner = base_reasoner

    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        if isinstance(ce, OWLClass):
            yield from super().instances(ce, direct=direct)
        else:
            if direct:
                warning("direct not implemented")
            with self._world.get_ontology("http://temp.classes/"):
                temp_pred = cast(owlready2.ThingClass, types.new_class("TempCls%d" % self._cnt, (owlready2.owl.Thing,)))
                temp_pred.equivalent_to = [self._conv.map_concept(ce)]
            self._sync_reasoner(other_reasoner=self._base_reasoner)
            instances = list(temp_pred.instances(world=self._world))
            owlready2.destroy_entity(temp_pred)
            self._cnt += 1
            for i in instances:
                yield OWLNamedIndividual(IRI.create(i.iri))
