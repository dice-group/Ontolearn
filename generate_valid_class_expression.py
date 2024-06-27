from rdflib import Graph, RDF, RDFS, OWL, URIRef
from itertools import product, combinations
from typing import List, Set
from owlapy.owl_property import OWLObjectInverseOf
# from owlapy.owl_property import get_inverse_property

def load_ontology(file_path):
    g = Graph()
    g.parse(file_path, format='xml')
    return g


def generate_class_expressions(kb, concept_type: str):

    #  1. all named classes
    named_concepts = [c.str.split("#")[-1] for c in kb.ontology.classes_in_signature() if c.str not in [
        "http://www.benchmark.org/family#PersonWithASibling",
        "http://www.benchmark.org/family#Child",
        "http://www.benchmark.org/family#Parent",
        "http://www.benchmark.org/family#Grandparent",
        "http://www.benchmark.org/family#Grandchild"]]

    # 2: Negate all concepts obtained from step 1 (---> length 2)
    negated_concepts = [f"¬{c}" for c in named_concepts]

    # 3a: Union and intersect all concepts: should be obtained from step 1 and step 2 (---> length 3 and 4)
    unions = [f"{c1} ⊔ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]
    intersections = [f"{c1} ⊓ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]

    # 3b: For all properties, generate ∃ and ∀ with fillers that is a union of step 1 and step 2 (---> length 3 and 4)
    object_properties = [r.str.split("#")[-1] for r in kb.ontology.object_properties_in_signature()]

    existential_restrictions = [f"∃ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]
    universal_restrictions = [f"∀ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]

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
                    existential_restrictions + universal_restrictions +
                    min_cardinality_restrictions + max_cardinality_restrictions +
                    existential_inverse_restrictions + universal_inverse_restrictions +
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




# def generate_class_expressions(kb, n):
#     # Helper function to count the length of an expression
#     def count_length(expr):
#         # Basic named concept
#         if isinstance(expr, str) and " " not in expr:
#             return 1
#         # Negation
#         if expr.startswith("¬"):
#             return 1 + count_length(expr[1:])
#         # Union or Intersection
#         if " ⊔ " in expr or " ⊓ " in expr:
#             parts = expr.split(" ⊔ ") if " ⊔ " in expr else expr.split(" ⊓ ")
#             return sum(count_length(part) for part in parts) + 1
#         # Existential or Universal Restriction
#         if expr.startswith("∃ ") or expr.startswith("∀ "):
#             _, rest = expr.split(" ", 1)
#             prop, concept = rest.split(".", 1)
#             return 1 + count_length(concept)

#     #  1. All named classes
#     named_concepts = [c.str for c in kb.ontology.classes_in_signature()]

#     # 2: Negate all concepts obtained from step 1 (---> length 2)
#     negated_concepts = [f"¬{c}" for c in named_concepts]

#     # 3a: Union and intersect all concepts: should be obtained from step 1 and step 2 (---> length 3 and 4)
#     unions = [f"{c1} ⊔ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]
#     intersections = [f"{c1} ⊓ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]

#     # 3b: For all properties, generate ∃ and ∀ with fillers that is a union of step 1 and step 2 (---> length 3 and 4)
#     object_properties = [r.str for r in kb.ontology.object_properties_in_signature()]

#     existential_restrictions = [f"∃ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]
#     universal_restrictions = [f"∀ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]

#     # Combine everything
#     all_concepts = named_concepts + negated_concepts + unions + intersections + existential_restrictions + universal_restrictions

    # Filter by length
    # return [expr for expr in all_concepts if count_length(expr) == n]

# Example usage:
# kb = your_knowledge_base_object
# n = desired_length_of_class_expression
# class_expressions = generate_class_expressions(kb, n)










# def generate_class_expressions(kb):
    
#     #  1. all named classes

#     # named_concepts = [str(c) for c in kb.subjects(RDF.type, OWL.Class) if "#" in str(c)]
#     named_concepts = [c.str.split("#")[-1] for c in kb.ontology.classes_in_signature()]

#     # 2: Negate all concepts obtained from step 1 (---> length 2)
#     negated_concepts = [f"¬{c}" for c in named_concepts]

#     # 3a: Union and intersect all concepts: should be obtained from step 1 and step 2 (---> length 3 and 4)
#     unions = [f"{c1} ⊔ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]
#     intersections = [f"{c1} ⊓ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]

#     # 3b: For all properties, generate ∃ and ∀ with fillers that is a union of step 1 and step 2 (---> length 3 and 4)

#     # object_properties = [str(p) for p in kb.subjects(RDF.type, OWL.ObjectProperty)]

#     object_properties = [r.str.split("#")[-1] for r in kb.ontology.object_properties_in_signature()]

#     existential_restrictions = [f"∃ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]
#     universal_restrictions = [f"∀ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]

#     # # Combine everything (---> length 1, 2, 3, 4)
#     all_concepts = named_concepts + negated_concepts + unions + intersections +  existential_restrictions + universal_restrictions #

#     return all_concepts


# def get_ground_truth_instances(kb: Graph, class_uri: str) -> Set[str]:
#     """
#     Get ground truth instances of a given class from the KB.
#     :param kb: The RDF Graph of the knowledge base.
#     :param class_uri: The URI of the class.
#     :return: A set of instance URIs.
#     """
#     class_ref = URIRef(class_uri)
#     query = f"""
#     SELECT ?individual WHERE {{
#         ?individual a <{class_ref}> .
#     }}
#     """
#     results = kb.query(query)
#     return {str(row[0]) for row in results}

# def is_valid_iri(iri: str) -> bool:
#     # A simple regex to check if the IRI is well-formed
#     return re.match(r"^https?://[^\s]+$", iri) is not None

