from rdflib import Graph, RDF, RDFS, OWL, URIRef
from itertools import product, combinations
from typing import List, Set
from owlready2 import *
from owlapy.owl_property import OWLObjectInverseOf
# from owlapy.owl_property import get_inverse_property

def check_abox(path_kb):

    kb = get_ontology(path_kb).load()

    # Get all named concepts (classes)
    all_concepts = list(kb.classes())

    # Get all individuals (instances) in the ABox
    all_individuals = list(kb.individuals())

    # Collect all concepts that are referenced by individuals in the ABox
    referenced_concepts = set()
    for individual in all_individuals:
        for concept in individual.is_a:
            referenced_concepts.add(concept)

    # Find concepts that are not in the ABox
    concepts_not_in_abox = [concept for concept in all_concepts if concept not in referenced_concepts]

    concepts_not_in_abox = [c.iri for c in concepts_not_in_abox]

    return concepts_not_in_abox



def generate_class_expressions(kb, concept_type: str):


    concepts_not_in_abox = check_abox(kb.path)

    # print(concepts_not_in_abox)
    # exit(0)
    
    #  1. all named classes
    named_concepts = [c.str.split("#")[-1] for c in kb.ontology.classes_in_signature() if c.str not in concepts_not_in_abox]

   
    # 2: Negate all concepts obtained from step 1 (---> length 2)
    negated_concepts = [f"¬{c}" for c in named_concepts]

    # 3a: Union and intersect all concepts: should be obtained from step 1 and step 2 (---> length 3 and 4)
    unions = [f"{c1} ⊔ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]
    intersections = [f"{c1} ⊓ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]

    # 3b: For all properties, generate ∃ and ∀ with fillers that is a union of step 1 and step 2 (---> length 3 and 4)
    object_properties = [r.str.split("#")[-1] for r in kb.ontology.object_properties_in_signature()]

    existential_restrictions = [f"∃ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts + unions + intersections]
    universal_restrictions = [f"∀ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts + unions + intersections]

    # 4. Generate cardinality restrictions (---> length 3 and 4)

    cardinality_values = [1,2,3]  
    min_cardinality_restrictions = [f"≥ {n} {p}.{c}" for n in cardinality_values for p in object_properties for c in named_concepts + negated_concepts]
    max_cardinality_restrictions = [f"≤ {n} {p}.{c}" for n in cardinality_values for p in object_properties for c in named_concepts + negated_concepts]
    exact_cardinality_restrictions = [f"= {n} {p}.{c}" for n in cardinality_values for p in object_properties for c in named_concepts + negated_concepts]

    # 5. Generate inverse property restrictions (---> length 3 and 4) 

    existential_inverse_restrictions = [f"∃ {p}⁻.{c}" for p in object_properties for c in named_concepts + negated_concepts]
    universal_inverse_restrictions = [f"∀ {p}⁻.{c}" for p in object_properties for c in named_concepts + negated_concepts]
    min_cardinality_inverse_restrictions = [f"≥ {n} {p}⁻.{c}" for n in cardinality_values for p in object_properties for c in named_concepts + negated_concepts]
    max_cardinality_inverse_restrictions = [f"≤ {n} {p}⁻.{c}" for n in cardinality_values for p in object_properties for c in named_concepts + negated_concepts]
    exact_cardinality_inverse_restrictions = [f"= {n} {p}⁻.{c}" for n in cardinality_values for p in object_properties for c in named_concepts + negated_concepts]

    # 7. Combine everything (---> length 1, 2, 3, 4)

    all_concepts = (named_concepts + negated_concepts + unions + intersections + 
                    existential_restrictions + universal_restrictions + exact_cardinality_restrictions+
                    min_cardinality_restrictions + max_cardinality_restrictions +
                    existential_inverse_restrictions + universal_inverse_restrictions + exact_cardinality_inverse_restrictions+
                    min_cardinality_inverse_restrictions + max_cardinality_inverse_restrictions)

    if "name" in concept_type:
        return named_concepts, len(named_concepts)
    if "nega" in concept_type:
        return negated_concepts, len(negated_concepts)
    if "union" in concept_type:
        return unions, len(unions)
    if "intersect" in concept_type:
        return intersections, len(intersections)
    
    if "exist" == concept_type:
        return existential_restrictions, len(existential_restrictions)
    if "universal" == concept_type:
        return universal_restrictions, len(universal_restrictions)
    if "min_card" == concept_type:
        return min_cardinality_restrictions, len(min_cardinality_restrictions)
    if "max_card" == concept_type:
        return max_cardinality_restrictions, len(max_cardinality_restrictions)
    if "exact_card" == concept_type:
        return exact_cardinality_restrictions, len(exact_cardinality_restrictions)
    
    if "exact_card_inv" in concept_type:
        return exact_cardinality_inverse_restrictions, len(exact_cardinality_inverse_restrictions) 
    if "exist_inv" in concept_type:
        return existential_inverse_restrictions, len(existential_inverse_restrictions)
    if "universal_inv" in concept_type:
        return universal_inverse_restrictions, len(universal_inverse_restrictions)
    if "min_card_inv" in concept_type:
        return min_cardinality_inverse_restrictions, len(min_cardinality_inverse_restrictions)
    if "max_card_inv" in concept_type:
        return max_cardinality_inverse_restrictions, len(max_cardinality_inverse_restrictions)
    else:
        return all_concepts, len(all_concepts)




