import types
from typing import Iterable

from owlready2 import AnnotationProperty, ThingClass, owl, destroy_entity

from ontolearn import BaseNode


def export_concepts(concepts: Iterable[BaseNode], file_path: str) -> None:
    temp_entities = []
    my_world = None
    new_concept_onto = None
    i = 0

    for c in concepts:
        if not my_world:
            my_world = c.concept.owl.namespace.world
            new_concept_onto = my_world.get_ontology("https://dice-research.org/predictions/")
            schema_onto = my_world.get_ontology("https://dice-research.org/predictions-schema/")
            with schema_onto:
                class quality_value(AnnotationProperty): pass
        with new_concept_onto:
            pred_class = types.new_class("Prediction_%d" % i, (owl.Thing,))
            my_restr = c.concept.owl
            pred_class.equivalent_to = [my_restr]
            pred_class.is_a.remove(owl.Thing)
            #pred_class.label = dl_render_concept_str(my_restr) # TODO
            pred_class.quality_value = c.quality
            temp_entities.append(pred_class)
            i = i + 1

    new_concept_onto.save(file = file_path)
    for e in temp_entities:
        destroy_entity(e)