from owlready2 import Ontology, World, Thing, Nothing
from typing import Tuple, Dict, Iterable
from .concept import Concept
from .abstracts import BaseNode
from collections import deque
from owlready2 import Thing, get_ontology, Not
import types
from .util import get_full_iri


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
    # @TODO why onto.world is given as parameter ?
    # @TODO if onto.world is important and has to be given as parameter why it has not being used in all other concept generation.
    T = Concept(Thing, kwargs={'form': 'Class'}, world=onto.world)
    bottom = Concept(Nothing, kwargs={'form': 'Class'}, world=onto.world)
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
    @param world:
    @param concepts:
    @return:
    """
    for c in concepts:
        for ind in c.owl.instances(world=world):
            ind.is_a.append(c.owl)


def apply_type_enrichment(concept: Concept) -> None:
    for ind in concept.instances:
        ind.is_a.append(concept.owl)


def type_enrichment(instances, new_concept):
    for i in instances:
        i.is_a.append(new_concept)


def concepts_sorter(A, B):
    if len(A) < len(B):
        return A, B
    if len(A) > len(B):
        return B, A

    args = [A, B]
    args.sort(key=lambda ce: ce.str)
    return args[0], args[1]


def retrieve_concept_chain(node: BaseNode) -> Iterable:
    """
    Given a node return its parent hierarchy
    @param node:
    @return:
    """
    hierarchy = deque()
    if node.parent_node:
        hierarchy.appendleft(node.parent_node)
        while hierarchy[-1].parent_node is not None:
            hierarchy.append(hierarchy[-1].parent_node)
        hierarchy.appendleft(node)
    return hierarchy

def decompose_to_atomic(C):
    if C.is_atomic:
        return C.owl
    elif C.form == 'ObjectComplementOf':
        concept_a = C.concept_a
        return Not(decompose_to_atomic(concept_a))
    elif C.form == 'ObjectUnionOf':
        return decompose_to_atomic(C.concept_a) | decompose_to_atomic(C.concept_b)
    elif C.form == 'ObjectIntersectionOf':
        return decompose_to_atomic(C.concept_a) & decompose_to_atomic(C.concept_b)
    elif C.form == 'ObjectSomeValuesFrom':
        return C.role.some(decompose_to_atomic(C.filler))
    elif C.form == 'ObjectAllValuesFrom':
        return C.role.only(decompose_to_atomic(C.filler))
    else:
        print('Someting wrong')
        print(C)
        raise ValueError
