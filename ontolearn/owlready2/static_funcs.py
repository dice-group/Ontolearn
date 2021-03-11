import time
import types

from owlready2 import Thing, Not


# def apply_type_enrichment_from_iterable(concepts: Iterable[Concept], world: World) -> None:
#     """
#     Extend ABOX by
#     (1) Obtaining all instances of selected concepts.
#     (2) For all instances in (1) include type information into ABOX.
#     @param world:
#     @param concepts:
#     @return:
#     """
#     for c in concepts:
#         for ind in c.owl.instances(world=world):
#             ind.is_a.append(c.owl)


# def apply_type_enrichment(concept: Concept) -> None:
#     for ind in concept.instances:
#         ind.is_a.append(concept.owl)


# def type_enrichment(instances, new_concept):
#     for i in instances:
#         i.is_a.append(new_concept)


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


def export_concepts(kb, concepts, path: str = 'concepts.owl', rdf_format: str = 'rdfxml') -> None:
    """
    @param kb: Knowledge Base object
    @param concepts:
    @param path:
    @param rdf_format: serialization format. “rdf/xml” or “ntriples”.
    @return:
    """
    o1 = kb.world.get_ontology('https://dice-research.org/predictions/' + str(time.time()))
    o1.imported_ontologies.append(kb._ontology)
    print('Number of concepts to be generated:', len(concepts))
    for ith, h in enumerate(concepts):
        with o1:
            w = types.new_class(name='Pred_' + str(ith), bases=(Thing,))
            w.is_a.remove(Thing)
            w.label.append(h.concept.str)
            try:
                w.equivalent_to.append(decompose_to_atomic(h.concept))
            except AttributeError as e:
                print('Attribute Error.')
                print(e)
                print('exiting.')
    o1.save(file=path, format=rdf_format)
    print('Concepts are saved.')
