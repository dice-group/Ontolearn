from owlready2 import Ontology, World, Thing, Nothing
from typing import Tuple, Dict, Iterable
from .concept import Concept


def build_concepts_mapping(onto: Ontology) -> Tuple[Dict, Concept, Concept]:
    """
    Construct a mapping from full_iri to corresponding Concept objects.

    concept.namespace.base_iri + concept.name
    mappings from concepts uri to concept objects
        1) full_iri:= owlready2.ThingClass.namespace.base_iri + owlready2.ThingClass.name
        2) Concept:
    """
    concepts = dict()
    individuals = set()
    T = Concept(Thing, kwargs={'form': 'Class'}, world=onto.world)
    bottom = Concept(Nothing, kwargs={'form': 'Class'}, world=onto.world)
    # TODO: Think about removing owlready2 instances and use string representation.
    for i in onto.classes():
        temp_concept = Concept(i, kwargs={'form': 'Class'}, world=onto.world)
        concepts[temp_concept.full_iri] = temp_concept
        individuals.update(temp_concept.instances)
    try:
        assert T.instances  # if empty throw error.
        assert individuals.issubset(T.instances)
    except AssertionError:
        print(
            'Sanity checking failed: owlready2.Thing does not contain any individual. To allivate this issue, we explicitly assign all individuals/instances to concept T.')
        T.instances = individuals

    concepts[T.full_iri] = T
    concepts[bottom.full_iri] = bottom
    return concepts, T, bottom


def apply_type_enrichment_from_iterable(concepts: Iterable[Concept], world: World) -> None:
    """
    Extend ABOX by
    (1) Obtaining all instances of selected concepts.
    (2) For all instances in (1) include type information into ABOX.
    @param concepts:
    @return:
    """
    for c in concepts:
        for ind in c.owl.instances(world=world):
            ind.is_a.append(c.owl)


def apply_type_enrichment(concept: Concept) -> None:
    for ind in concept.instances:
        ind.is_a.append(concept.owl)
