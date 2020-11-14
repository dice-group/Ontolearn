from owlready2 import Ontology, World, Thing, Nothing
from typing import Tuple, Dict, Iterable
from .concept import Concept
from .abstracts import BaseNode
from collections import deque
from owlready2 import Thing, get_ontology
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
    T = Concept(Thing, kwargs={'form': 'Class'}, world=onto.world)
    bottom = Concept(Nothing, kwargs={'form': 'Class'}, world=onto.world)
    # TODO: Think about removing owlready2 instances and use string representation.
    for i in onto.classes():
        # i.namespace.base_iri = namespace_base_iri
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


def add_classes_to_ontology(node: BaseNode, onto):
    """

    @param node:
    @param onto:
    @return:
    """
    current_defined_classes = list(onto.classes())
    if node.concept.owl == Thing:
        return
    base_nodes = list(i for i in retrieve_concept_chain(node))
    assert base_nodes[0].concept.owl == node.concept.owl  # must be equal to itself.
    assert base_nodes[-1].concept.owl == Thing  # must be equal to T.

    if len(base_nodes) == 1:
        assert base_nodes[0].concept.owl == Thing
        return
    base_nodes.pop(0)  # itself.
    base_nodes.pop(-1)  # Thing.
    for b in base_nodes:
        if b.concept.owl not in current_defined_classes:
            print(b)
            add_classes_to_ontology(b, onto)

    # bases
    bases = tuple([i.concept.owl for i in base_nodes] + [Thing])

    with onto:  # Add into TBOX.
        new_concept = types.new_class(name=node.concept.str, bases=bases)
        # new_concept.equivalent_to = node.concept.owl.equivalent_to


def dl_syntax_to_word(x):
    """
    Given dl syntax, convert it to workds.
    @param x:
    @return:
    """
    return x


def add_concept_to_onto(name_of_concept, node, onto, classes_, classes_str_iri):
    x = classes_[classes_str_iri.index(get_full_iri(node.concept.owl))]
    # For sanity checking.
    assert get_full_iri(x) == get_full_iri(node.concept.owl)

    #concept_chain_of_h = list(retrieve_concept_chain(node))[1:-1]  # first is h. and last is Thing
    #bases = tuple([classes_[classes_str_iri.index(get_full_iri(i.concept.owl))] for i in concept_chain_of_h])
    #if len(bases) == 0:
    bases = (Thing,)

    with onto:
        w = types.new_class(name=name_of_concept+node.concept.str, bases=bases)
        w.label.append(dl_syntax_to_word(node.concept.str))
        w.equivalent_to.extend(x.equivalent_to)
        w.f1_score = [node.quality]
