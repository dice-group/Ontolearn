from rdflib import Graph, RDF, RDFS, OWL, URIRef
from itertools import product, combinations
from typing import List, Set
import re

def load_ontology(file_path):
    g = Graph()
    g.parse(file_path, format='xml')
    return g

def generate_class_expressions(kb):
    
    #  1. all named classes

    # named_concepts = [str(c) for c in kb.subjects(RDF.type, OWL.Class) if "#" in str(c)]
    named_concepts = [c.str.split("#")[-1] for c in kb.ontology.classes_in_signature()]

    # 2: Negate all concepts obtained from step 1 (---> length 2)
    negated_concepts = [f"¬{c}" for c in named_concepts]

    # 3a: Union and intersect all concepts: should be obtained from step 1 and step 2 (---> length 3 and 4)
    unions = [f"{c1} ⊔ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]
    intersections = [f"{c1} ⊓ {c2}" for c1 in named_concepts + negated_concepts for c2 in named_concepts + negated_concepts if c1 != c2]

    # 3b: For all properties, generate ∃ and ∀ with fillers that is a union of step 1 and step 2 (---> length 3 and 4)

    # object_properties = [str(p) for p in kb.subjects(RDF.type, OWL.ObjectProperty)]

    object_properties = [r.str.split("#")[-1] for r in kb.ontology.object_properties_in_signature()]

    existential_restrictions = [f"∃ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]
    universal_restrictions = [f"∀ {p}.{c}" for p in object_properties for c in named_concepts + negated_concepts]

    # # Combine everything (---> length 1, 2, 3, 4)
    all_concepts = named_concepts + negated_concepts + unions + intersections +  existential_restrictions + universal_restrictions #

    return all_concepts


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

