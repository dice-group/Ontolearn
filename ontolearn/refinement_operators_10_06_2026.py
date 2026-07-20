# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""Refinement Operators for refinement-based concept learners."""
from collections import defaultdict
from typing import FrozenSet, Tuple, Dict
from ordered_set import OrderedSet
from itertools import chain
import random
from typing import DefaultDict, Dict, Set, Optional, Iterable, List, Type, Final, Generator, Tuple

from owlapy import owl_expression_to_sparql_with_confusion_matrix
from ontolearn.utils.static_funcs import compute_f1_score_from_confusion_matrix

from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectIntersectionOf, \
    OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, OWLObjectUnionOf, OWLClass, \
    OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLDataSomeValuesFrom, \
    OWLDatatypeRestriction, OWLDataHasValue, OWLObjectExactCardinality, OWLObjectOneOf, OWLDataOneOf
from owlapy.owl_individual import OWLIndividual
from owlapy.owl_literal import OWLLiteral, TopOWLDatatype
from owlapy.owl_object import OWLObject
from owlapy.owl_property import OWLObjectPropertyExpression, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataPropertyExpression, OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer

from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter
from owlapy.providers import owl_datatype_max_inclusive_restriction, owl_datatype_min_inclusive_restriction
from owlapy.vocab import OWLFacet
from owlready2 import owl_object_property
from sqlalchemy.util import OrderedSet
from sympy import true
from transformers.models.llava_next import processing_llava_next

from .abstracts import BaseRefinement, AbstractKnowledgeBase
from .concept_generator import ConceptGenerator
from .knowledge_base import KnowledgeBase
# from .nero_utils import AtomicExpression, ExistentialQuantifierExpression, ClassExpression, \
#   ComplementOfAtomicExpression, UniversalQuantifierExpression, UnionClassExpression, IntersectionClassExpression, Role
from .search import OENode
from owlapy import owl_expression_to_sparql
import requests
from requests.adapters import HTTPAdapter
from owlapy.iri import IRI
from owlapy.marked_entity_generator_converter_10_06_2026 import (
    CONTEXT_POSITION_MARKER,
    owl_expression_to_class_query,
    owl_expression_to_property_query as property_query_with_counts,
    owl_expression_to_inverse_property_query as inverse_property_query_with_counts,
    owl_expression_to_negated_class_query,
    owl_expression_to_data_property_query as data_property_query_with_counts,
    owl_expression_to_data_property_value_range_query,
    owl_expression_to_qualified_cardinality_query,
)

from owlapy.utils import get_top_level_dnf


def evaluate_with_confusion_matrix(concept: OWLClassExpression, kb: AbstractKnowledgeBase, pos: Set, neg: Set):
    """
    Evaluate a concept using confusion matrix metrics (F1, precision, recall).

    Args:
        concept: OWL class expression to evaluate
        kb: Knowledge base for querying
        pos: Set of positive examples (OWLIndividuals)
        neg: Set of negative examples (OWLIndividuals)

    Returns:
        Tuple of (f1, precision, recall, tp, fp)
    """
    # Special case: owl:Thing may not be explicitly asserted as rdf:type owl:Thing
    if concept == OWLThing:
        tp = len(pos)
        fp = len(neg)
        fn = 0
        tn = 0
        confusion_matrix = {
            "tp": tp,
            "fp": fp,
            "fn": 0,
            "tn": 0,
        }

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = compute_f1_score_from_confusion_matrix(confusion_matrix=confusion_matrix)

    else:
        sparql_query = owl_expression_to_sparql_with_confusion_matrix(expression=concept,
                                                                      positive_examples=pos,
                                                                      negative_examples=neg)
        response = kb.query(sparql_query)
        
        if response.status_code != 200:
            return 0,0,0,0,0
        
        bindings = response.json()["results"]["bindings"]
        assert len(bindings) == 1
        bindings = bindings.pop()

        confusion_matrix = {k: v["value"] for k, v in bindings.items()}
        tp = int(confusion_matrix["tp"])
        fp = int(confusion_matrix["fp"])
        fn = int(confusion_matrix["fn"])
        tn = int(confusion_matrix["tn"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = compute_f1_score_from_confusion_matrix(confusion_matrix=confusion_matrix)

    return f1, precision, recall, tp, fp


class PruneCELBasedRefinement(BaseRefinement):
    """
    Recursive refinement operator based on PruneCEL
    Properties:
        - Recursive: Each refinement method can call rho_star() on sub-expressions
        - Coverage-guided: Only generates concepts covering positive examples
        - Context-aware: Uses position marker μ to track refinement location
        - Prevents redundancy: Checks for duplicate operands (A⊓A, A⊔A) and nested same properties
    """

    def __init__(self, knowledge_base: AbstractKnowledgeBase,
                 max_concepts: int = 100,
                 random_seed: Optional[int] = None,
                 sparql_endpoint: Optional[str] = None,
                 length_penalty: float = 0.01,
                 enable_negation: bool = True,
                 enable_inverse_roles: bool = True,
                 enable_data_properties: bool = True,
                 enable_qualified_cardinality: bool = True
                 ):
        """
        Initialize PruneCEL recursive refinement operator.

        Args:
            knowledge_base: Knowledge base containing the ontology and instances
            max_concepts: Maximum number of concepts to generate per refinement
            random_seed: Random seed for reproducibility
            sparql_endpoint: Optional SPARQL endpoint URL for smart suggestion.
                           If provided, uses SPARQL-based candidate selection.
                           If None, falls back to local KB-based selection.
            length_penalty: Penalty for concept length in refinement score (default: 0.01)
            enable_inverse_roles: If True, enable generation of inverse property expressions (I)
                                m*(T, ∃ r⁻ . ⊤) (default: True)
            enable_data_properties: If True, enable generation of data property expressions (D)
                                  m*(T, ∃ r . rdfs:literal) (default: True)
        """
        super().__init__(knowledge_base)
        self.max_concepts = max_concepts
        self.random_seed = random_seed
        self.sparql_endpoint = sparql_endpoint
        self.length_penalty = length_penalty
        self.concept_generator = ConceptGenerator()
        self.pos = None
        self.neg = None
        self.top_refinements = set()  # Pool of previously refined concepts for complex fillers

        # negation, inverse(i), data properties(D), quantifier restriction(Q)
        self.enable_negation = enable_negation # I: m*(T, ¬A)
        self.enable_inverse_roles = enable_inverse_roles  # I: m*(T, ∃ r⁻ . ⊤)
        self.enable_data_properties = enable_data_properties  # D: m*(T, ∃ r . rdfs:literal)
        self.enable_qualified_cardinality = enable_qualified_cardinality # Q: m*(T, ≥n R.C) and m*(T, ≤n R.C)
        
        # High-precision fragment collection
        # self.high_precision_fragments = []  # Store high-precision, low-recall concepts
        self.covered_positives = set()  # Track which positives are covered by fragments

        # Renderer for debug output
        self._renderer = DLSyntaxObjectRenderer()

        # Define template as a global variable for now (can be improved by passing it through methods)
        # self.template = CONTEXT_POSITION_MARKER
        self._score_cache = {}

        # Performance toggles (flip these to compare speed/quality tradeoffs).
        self.enable_sparql_cache = True
        self._sparql_cache = {}
        self._sparql_headers = {
            "Accept": "application/sparql-results+json"
        }
        self._sparql_timeout = 600
        self._http_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self._http_session.mount("http://", adapter)
        self._http_session.mount("https://", adapter)

        # Template-based result caches for oracle methods
        self._class_template_cache = {}       # template_key -> list of (class, f1, precision, recall, pos_hits, neg_hits)
        self._role_template_cache = {}        # template_key -> list of (role, f1, precision, recall, pos_hits, neg_hits)
        self._inversed_role_template_cache = {}  # template_key -> list of (inversed_role, f1, precision, recall, pos_hits, neg_hits)
        self._qualified_cardinality_template_cache = {}  # template_key -> list of (restriction, f1, precision, recall, pos_hits, neg_hits)

        self._data_property_role_template_cache = {}
        self._neg_template_cache = {}         # template_key -> list of (neg_class, f1, precision, recall, pos_hits, neg_hits)
        # self._data_template_cache = {}        # template_key -> list of (filler, f1, precision, recall, pos_hits, neg_hits)


    def sparql(self, query: str, timeout: Optional[float] = None) -> List[Dict[str, str]]:

        cache_key = None
        if self.enable_sparql_cache:
            # Normalize whitespace so semantically identical queries map to one key.
            normalized_query = " ".join(query.split())
            cache_key = (normalized_query)
            cached = self._sparql_cache.get(cache_key)
            if cached is not None:
                return cached

        effective_timeout = timeout if timeout is not None else self._sparql_timeout
        
        try:
            response = self._http_session.post(self.sparql_endpoint,
                                               data={"query": query},
                                               headers=self._sparql_headers,
                                               timeout=effective_timeout)
        
            response.raise_for_status()
            bindings = response.json()["results"]["bindings"]
            if cache_key is not None:
                self._sparql_cache[cache_key] = bindings
            
        except Exception as e:
            print(e)
            bindings = []
        return bindings

    def close(self):
        if hasattr(self, "_http_session") and self._http_session is not None:
            self._http_session.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_input_examples(self, pos: frozenset, neg: frozenset):
        """Set positive and negative examples for coverage-guided refinement."""
        assert isinstance(pos, frozenset)
        self.pos = {i for i in pos}
        self.neg = {i for i in neg}
        self.set_examples = self.pos.union(self.neg)
        self.top_refinements = set()

        # Clear template caches when examples change (cached results depend on pos/neg)
        self._class_template_cache.clear()
        self._role_template_cache.clear()
        self._inversed_role_template_cache.clear()
        self._data_property_role_template_cache.clear()
        self._neg_template_cache.clear()
        # self._data_template_cache.clear()



    def _contains_context_marker(self, expr: OWLClassExpression) -> bool:
        if expr == CONTEXT_POSITION_MARKER:
            return True
        if isinstance(expr, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
            return any(self._contains_context_marker(op) for op in expr.operands())
        if isinstance(expr, (OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom)):
            return self._contains_context_marker(expr.get_filler())
        if isinstance(expr, OWLDataSomeValuesFrom):   # data property scope also carries the marker
            return self._contains_context_marker(expr.get_filler())
        if isinstance(expr, OWLObjectComplementOf):
            return self._contains_context_marker(expr.get_operand())
        return False

    def oracle_class_suggestor(self, template):
        # return class expression, f1, precision, recall, poshit, neghit

        # Check template cache
        template_key = str(template)
        cached = self._class_template_cache.get(template_key)
        if cached is not None:
            return list(cached)

        query = owl_expression_to_class_query(
            context=template,
            positive_examples=self.pos,
            negative_examples=self.neg,
        )

        res = self.sparql(query)

        results = []

        for row in res:
            if "class" not in row or "value" not in row["class"]:
                continue
            uri = row["class"]["value"]

            if "posHits" not in row or "value" not in row["posHits"]:
                # owl_class = OWLClass(IRI.create(uri))
                # results.append((owl_class, 0, 0, 0, 0, 0))
                continue
            pos_hits = int(row["posHits"]["value"])

            if "negHits" not in row or "value" not in row["negHits"]:
                continue

            neg_hits = int(row["negHits"]["value"])

            if not uri.startswith(("http://", "https://")):
                continue

            if uri in {"http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                       "http://www.w3.org/2002/07/owl#NamedIndividual",
                       "http://www.w3.org/2002/07/owl#Thing", "http://www.w3.org/2002/07/owl#Ontology",
                       "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                       "http://www.w3.org/2002/07/owl#ObjectProperty",
                       "http://www.w3.org/2000/01/rdf-schema#Class",
                       "http://www.w3.org/2002/07/owl#Class",
                       "http://www.w3.org/2002/07/owl#TransitiveProperty",
                       "http://www.w3.org/2002/07/owl#DatatypeProperty",
                       "http://www.w3.org/2002/07/owl#"}:
                continue

            if pos_hits + neg_hits == 0:
                continue

            tp = pos_hits
            fp = neg_hits
            fn = len(self.pos) - pos_hits
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            owl_class = OWLClass(IRI.create(uri))
            results.append((owl_class,f1,precision,recall,pos_hits,neg_hits))

        # Store in cache as tuple (immutable)
        self._class_template_cache[template_key] = tuple(results)
        return results


    def oracle_data_role_suggestor(self, template: OWLClassExpression) -> Set[OWLObjectProperty]:
        # return role, f1, precision, recall, poshit, neghit

        # Check template cache
        template_key = str(template)
        cached = self._data_property_role_template_cache.get(template_key)
        if cached is not None:
            return list(cached)

        query = data_property_query_with_counts(
            context=template,
            positive_examples=self.pos,
            negative_examples=self.neg,
        )

        res = self.sparql(query)
        results = []

        for row in res:
            if "prop" not in row or "value" not in row["prop"]:
                continue

            uri = row["prop"]["value"]
            pos_hits = int(row["posHits"]["value"])
            neg_hits = int(row["negHits"]["value"])

            if not uri.startswith(("http://", "https://")):
                continue

            if uri in {"http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                       "http://www.w3.org/2002/07/owl#NamedIndividual",
                       "http://www.w3.org/2002/07/owl#Thing",
                       "http://www.w3.org/2002/07/owl#Ontology",
                       "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                       "http://www.w3.org/2002/07/owl#ObjectProperty",
                       "http://www.w3.org/2000/01/rdf-schema#Class",
                       "http://www.w3.org/2002/07/owl#Class",
                       "http://www.w3.org/2002/07/owl#TransitiveProperty",
                       "http://www.w3.org/2002/07/owl#DatatypeProperty",
                       "http://www.w3.org/2002/07/owl#"}:
                continue

            if pos_hits + neg_hits == 0:
                continue

            tp = pos_hits
            fp = neg_hits
            fn = len(self.pos) - pos_hits
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            owl_class = OWLDataProperty(IRI.create(uri))
            results.append((owl_class,f1,precision,recall,pos_hits,neg_hits))

        # Store in cache as tuple (immutable)
        self._data_property_role_template_cache[template_key] = tuple(results)
        return results


    def oracle_role_suggestor(self, template: OWLClassExpression) -> Set[OWLObjectProperty]:
        # return role, f1, precision, recall, poshit, neghit

        # Check template cache
        template_key = str(template)
        # print(f"Template: {self._renderer.render(template)} ")
        cached = self._role_template_cache.get(template_key)
        if cached is not None:
            return list(cached)

        query = property_query_with_counts(
            context=template,
            positive_examples=self.pos,
            negative_examples=self.neg,
        )

        res = self.sparql(query)
        results = []

        for row in res:
            if "prop" not in row or "value" not in row["prop"]:
                continue

            uri = row["prop"]["value"]
            pos_hits = int(row["posHits"]["value"])
            neg_hits = int(row["negHits"]["value"])

            if not uri.startswith(("http://", "https://")):
                continue

            if uri in {"http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                       "http://www.w3.org/2002/07/owl#NamedIndividual",
                       "http://www.w3.org/2002/07/owl#Thing",
                       "http://www.w3.org/2002/07/owl#Ontology",
                       "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                       "http://www.w3.org/2002/07/owl#ObjectProperty",
                       "http://www.w3.org/2000/01/rdf-schema#Class",
                       "http://www.w3.org/2002/07/owl#Class",
                       "http://www.w3.org/2002/07/owl#TransitiveProperty",
                       "http://www.w3.org/2002/07/owl#DatatypeProperty",
                       "http://www.w3.org/2002/07/owl#"}:
                continue

            if pos_hits + neg_hits == 0:
                continue

            tp = pos_hits
            fp = neg_hits
            fn = len(self.pos) - pos_hits
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            owl_class = OWLObjectProperty(IRI.create(uri))
            results.append((owl_class,f1,precision,recall,pos_hits,neg_hits))

        # Store in cache as tuple (immutable)
        self._role_template_cache[template_key] = tuple(results)
        return results



    def oracle_inverse_role_suggestor(self, template: OWLClassExpression) -> Set[OWLObjectProperty]:
        # return role, f1, precision, recall, poshit, neghit

        # Check template cache
        template_key = str(template)
        cached = self._inversed_role_template_cache.get(template_key)
        if cached is not None:
            return list(cached)

        query = inverse_property_query_with_counts(
            context=template,
            positive_examples=self.pos,
            negative_examples=self.neg,
        )

        res = self.sparql(query)
        results = []

        for row in res:
            if "prop" not in row or "value" not in row["prop"]:
                continue

            uri = row["prop"]["value"]
            pos_hits = int(row["posHits"]["value"])
            neg_hits = int(row["negHits"]["value"])

            if not uri.startswith(("http://", "https://")):
                continue

            if uri in {"http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                       "http://www.w3.org/2002/07/owl#NamedIndividual",
                       "http://www.w3.org/2002/07/owl#Thing",
                       "http://www.w3.org/2002/07/owl#Ontology",
                       "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                       "http://www.w3.org/2002/07/owl#ObjectProperty",
                       "http://www.w3.org/2000/01/rdf-schema#Class",
                       "http://www.w3.org/2002/07/owl#Class",
                       "http://www.w3.org/2002/07/owl#TransitiveProperty",
                       "http://www.w3.org/2002/07/owl#DatatypeProperty",
                       "http://www.w3.org/2002/07/owl#"}:
                continue

            if pos_hits + neg_hits == 0:
                continue

            tp = pos_hits
            fp = neg_hits
            fn = len(self.pos) - pos_hits
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            owl_class = OWLObjectInverseOf(OWLObjectProperty(IRI.create(uri)))
            results.append((owl_class, f1, precision, recall, pos_hits, neg_hits))

        # Store in cache as tuple (immutable)
        self._inversed_role_template_cache[template_key] = tuple(results)
        return results



    def oracle_negation_class_suggestor(self, template: OWLClassExpression) -> Set[OWLClassExpression]:
        # return class expression, f1, precision, recall, poshit, neghit

        # Check template cache
        template_key = str(template)
        cached = self._neg_template_cache.get(template_key)
        if cached is not None:
            return list(cached)

        query = owl_expression_to_negated_class_query(
            context=template,
            positive_examples=self.pos,
            negative_examples=self.neg)

        res = self.sparql(query)

        results = []

        for row in res:
            if "class" not in row or "value" not in row["class"]:
                continue

            uri = row["class"]["value"]
            pos_hits = int(row["posHits"]["value"])
            neg_hits = int(row["negHits"]["value"])

            if not uri.startswith(("http://", "https://")):
                continue

            if uri in {"http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                       "http://www.w3.org/2002/07/owl#NamedIndividual",
                       "http://www.w3.org/2002/07/owl#Thing", "http://www.w3.org/2002/07/owl#Ontology",
                       "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                       "http://www.w3.org/2002/07/owl#ObjectProperty",
                       "http://www.w3.org/2000/01/rdf-schema#Class",
                       "http://www.w3.org/2002/07/owl#Class",
                       "http://www.w3.org/2002/07/owl#TransitiveProperty",
                       "http://www.w3.org/2002/07/owl#DatatypeProperty",
                       "http://www.w3.org/2002/07/owl#"}:
                continue

            if pos_hits + neg_hits == 0:
                continue

            tp = pos_hits
            fp = neg_hits
            fn = len(self.pos) - pos_hits
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            owl_class = OWLClass(IRI.create(uri))
            results.append((owl_class,f1,precision,recall,pos_hits,neg_hits))

        # Store in cache as tuple (immutable)
        self._neg_template_cache[template_key] = tuple(results)
        return results

    def oracle_data_value_suggestor(self, template: OWLClassExpression) -> list:
        """Find the best separator for a datatype property - supports both numeric and boolean values.

        Returns list of (restriction, f1, precision, recall, tp, fp) tuples sorted by F1 score.

        For numeric properties: Uses the data property value range query to get min/max values per
        individual, then tries all (comparator, threshold) combinations (>=, <=).

        For boolean properties: Since minVal == maxVal for booleans, we take one value and calculate
        F1 scores for both True and False matching options.

        Supports refinements like:
        - Numeric: ∃ charge.⊤ → ∃ charge.[≥ 0.5]
        - Boolean: ∃ hasFeature.⊤ → ∃ hasFeature.{true} or ∃ hasFeature.{false}
        """
        # Generate query to get min/max values per individual
        query = owl_expression_to_data_property_value_range_query(
            context=template,
            positive_examples=self.pos,
            negative_examples=self.neg,
        )
        
        res = self.sparql(query)
        
        # Parse results: individual -> {min, max, is_positive}
        individual_ranges = {}
        for row in res:
            if "ind" not in row or "value" not in row["ind"]:
                continue
            if "isPos" not in row or "value" not in row["isPos"]:
                continue
            if "minVal" not in row or "value" not in row["minVal"]:
                continue
            if "maxVal" not in row or "value" not in row["maxVal"]:
                continue

            try:
                ind_uri = row["ind"]["value"]
                is_pos = row["isPos"]["value"].lower() == "true"
                min_val_str = row["minVal"]["value"]
                max_val_str = row["maxVal"]["value"]

                # Check if this is a boolean value (minVal will be "true" or "false")
                is_bool = min_val_str.lower() in ['true', 'false']

                if is_bool:
                    min_bool = min_val_str.lower() == 'true'
                    max_bool = max_val_str.lower() == 'true'

                    # Store both min and max boolean values
                    individual_ranges[ind_uri] = {
                        'is_positive': is_pos,
                        'min_bool': min_bool,
                        'max_bool': max_bool,
                        'is_boolean': True
                    }
                else:
                    # Parse as numeric (original logic)
                    min_val = float(min_val_str)
                    max_val = float(max_val_str)

                    individual_ranges[ind_uri] = {
                        'is_positive': is_pos,
                        'min': min_val,
                        'max': max_val,
                        'is_boolean': False
                    }
            except (ValueError, KeyError, TypeError):
                continue

        if not individual_ranges:
            return []
        
        # Determine if we're dealing with booleans or numbers
        is_all_boolean = all(data.get('is_boolean', False) for data in individual_ranges.values())

        results = []

        # Determine if we're dealing with booleans or numbers
        is_all_boolean = all(data.get('is_boolean', False) for data in individual_ranges.values())

        results = []

        if is_all_boolean:
            # BOOLEAN VALUE HANDLING
            # Try both True and False values and calculate F1 for each
            for bool_target in [True, False]:
                tp = 0
                fp = 0

                for data in individual_ranges.values():
                    is_pos = data['is_positive']
                    min_bool = data.get('min_bool', False)
                    max_bool = data.get('max_bool', False)

                    # Check if individual has the target value
                    if min_bool == max_bool:
                        # Case 1: Individual has only one consistent value (all same)
                        matches = (min_bool == bool_target)
                    else:
                        # Case 2: Individual has BOTH true and false
                        # Matches both {true} AND {false}
                        matches = True

                    if matches:
                        if is_pos:
                            tp += 1
                        else:
                            fp += 1

                fn = len(self.pos) - tp

                if tp + fp == 0:
                    continue

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                # Create restriction using OWLDataOneOf for exact boolean matching
                bool_literal = OWLLiteral(bool_target)
                final_bool = OWLDataOneOf(bool_literal)

                results.append((final_bool, f1, precision, recall, tp, fp))
        else:
            # NUMERIC VALUE HANDLING (original logic - completely unchanged)
            # Extract all threshold candidates from min/max values
            thresholds = set()
            for data in individual_ranges.values():
                thresholds.add(data['min'])
                thresholds.add(data['max'])

            thresholds = sorted(thresholds)

            # for all ">=", "<="  cases "range" doesn't support..
            comparators = [">=", "<="]  # Add range comparator

            for comparator in comparators:

                if comparator == "range":
                    # Generate range candidates from all threshold pairs (not only consecutive ones)
                    for i in range(len(thresholds) - 1):
                        for j in range(i + 1, len(thresholds)):
                            min_threshold = thresholds[i]
                            max_threshold = thresholds[j]

                            tp = 0
                            fp = 0

                            for data in individual_ranges.values():
                                is_pos = data['is_positive']
                                min_v = data['min']
                                max_v = data['max']

                                # Check if range overlaps with [min_threshold, max_threshold]
                                matches = not (max_v < min_threshold or min_v > max_threshold)

                                if matches:
                                    if is_pos:
                                        tp += 1
                                    else:
                                        fp += 1

                            fn = len(self.pos) - tp

                            if tp + fp == 0:
                                continue

                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                            # Create range restriction [min, max]
                            min_restriction = owl_datatype_min_inclusive_restriction(min_threshold)
                            max_restriction = owl_datatype_max_inclusive_restriction(max_threshold)
                            restriction = OWLObjectIntersectionOf([min_restriction, max_restriction])

                            results.append((restriction, f1, precision, recall, tp, fp))

                else:
                    # Original >= and <= logic
                    for threshold in thresholds:
                        tp = 0
                        fp = 0

                        for data in individual_ranges.values():
                            is_pos = data['is_positive']
                            min_v = data['min']
                            max_v = data['max']

                            matches = False
                            if comparator == ">=":
                                matches = max_v >= threshold
                            elif comparator == "<=":
                                matches = min_v <= threshold

                            if matches:
                                if is_pos:
                                    tp += 1
                                else:
                                    fp += 1

                        fn = len(self.pos) - tp

                        if tp + fp == 0:
                            continue

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                        if comparator == ">=":
                            from owlapy.providers import owl_datatype_min_inclusive_restriction
                            restriction = owl_datatype_min_inclusive_restriction(threshold)
                        else:
                            from owlapy.providers import owl_datatype_max_inclusive_restriction
                            restriction = owl_datatype_max_inclusive_restriction(threshold)

                        results.append((restriction, f1, precision, recall, tp, fp))

            # Sort by F1 score descending, then by precision
            # results.sort(key=lambda x: (-x[1], -x[2]))

            if results:
                best_f1 = max(r[1] for r in results)
                results = [r for r in results if r[1] == best_f1]

        return results



    def oracle_qualified_cardinality_suggestor(self, template: OWLClassExpression, role: OWLObjectProperty, filler: OWLClassExpression,negated) -> list:
        """
        Discover best qualified cardinality restrictions: ≥n r.X, ≤n r.X for a specific role and filler
        
        Args:
            template: The context template
            role: The specific role to analyze (e.g., from ∃ r . X)
            filler: The filler class/expression to consider (e.g., X from ∃ r . X)
        
        Returns list of (expr, f1, precision, recall, pos_hits, neg_hits) tuples
        sorted by F1 score descending
        """
        
        # Check template cache first - cache key includes role and filler
        # Handle both regular properties and inverse properties
        if isinstance(role, OWLObjectInverseOf):
            # For inverse roles, get the wrapped property
            role_str = f"Inverse({role.get_inverse_property().to_string_id()})"
        else:
            role_str = role.to_string_id()
        
        template_key = f"{str(template)}_{role_str}_{str(filler)}"
        cached = self._qualified_cardinality_template_cache.get(template_key)
        if cached is not None:
            return list(cached)
        
        results = []
        negated_results = []
        all_results = []

        
        # Only process if we have positive/negative examples
        if not self.pos or not self.neg:
            return results
        
        try:
            # Run SPARQL query to discover qualified cardinalities for this specific role+filler
            query = owl_expression_to_qualified_cardinality_query(
                context=template,
                positive_examples=self.pos,
                negative_examples=self.neg,
                role=role,
                filler=filler,
            )
            
            res = self.sparql(query,3)
            
            # Parse all results first
            valid_rows = []
            for row in res:
                try:
                    if "cardinality" not in row or "posHits" not in row or "negHits" not in row:
                        continue
                    
                    cardinality = int(row["cardinality"]["value"])
                    pos_hits = int(row["posHits"]["value"])
                    neg_hits = int(row["negHits"]["value"])
                    
                    # Skip invalid cardinalities
                    if cardinality <= 0:
                        continue
                    
                    if pos_hits + neg_hits == 0:
                        continue
                    
                    valid_rows.append({
                        'cardinality': cardinality,
                        'posHits': pos_hits,
                        'negHits': neg_hits
                    })
                except Exception:
                    continue
            
            # Calculate instances with cardinality 0 (not in query results)
            total_pos_with_cardinality_ge_1 = sum(r['posHits'] for r in valid_rows)
            total_neg_with_cardinality_ge_1 = sum(r['negHits'] for r in valid_rows)
            pos_with_cardinality_0 = len(self.pos) - total_pos_with_cardinality_ge_1
            neg_with_cardinality_0 = len(self.neg) - total_neg_with_cardinality_ge_1
            
            # Now process with proper aggregation
            for row in valid_rows:
                cardinality = row['cardinality']
                
                # ===== COMPUTE F1 FOR >= cardinality (AGGREGATED) =====
                tp_ge = 0
                fp_ge = 0
                for r in valid_rows:
                    if r['cardinality'] >= cardinality:
                        tp_ge += r['posHits']
                        fp_ge += r['negHits']
                
                fn_ge = len(self.pos) - tp_ge
                precision_ge = tp_ge / (tp_ge + fp_ge) if (tp_ge + fp_ge) > 0 else 0.0
                recall_ge = tp_ge / len(self.pos) if len(self.pos) > 0 else 0.0
                f1_ge = 2 * precision_ge * recall_ge / (precision_ge + recall_ge) if (precision_ge + recall_ge) > 0 else 0.0
                
                # Create >= cardinality expression
                min_card_expr = OWLObjectMinCardinality(cardinality=cardinality, property=role, filler=filler)
                expr_min = self.m_star(template, min_card_expr)
                results.append((expr_min, f1_ge, precision_ge, recall_ge, tp_ge, fp_ge))
                
                # ===== COMPUTE F1 FOR <= cardinality (AGGREGATED) =====
                # IMPORTANT: Include cardinality 0 instances (not in query results)
                tp_le = pos_with_cardinality_0  # Start with cardinality 0 instances
                fp_le = neg_with_cardinality_0
                for r in valid_rows:
                    if r['cardinality'] <= cardinality:
                        tp_le += r['posHits']
                        fp_le += r['negHits']
                
                fn_le = len(self.pos) - tp_le
                precision_le = tp_le / (tp_le + fp_le) if (tp_le + fp_le) > 0 else 0.0
                recall_le = tp_le / len(self.pos) if len(self.pos) > 0 else 0.0
                f1_le = 2 * precision_le * recall_le / (precision_le + recall_le) if (precision_le + recall_le) > 0 else 0.0
                
                # Create <= cardinality expression
                max_card_expr = OWLObjectMaxCardinality(cardinality=cardinality, property=role, filler=filler)
                expr_max = self.m_star(template, max_card_expr)
                results.append((expr_max, f1_le, precision_le, recall_le, tp_le, fp_le))

            if negated:
                for child in results:
                    concept_expr, f1, precision, recall, pos_hit, neg_hit = child

                    negated_concept = OWLObjectComplementOf(concept_expr)

                    # Recalculate coverage for negated concept (complement coverage)
                    negated_pos_hit = len(self.pos) - pos_hit  # ¬C covers positives that C didn't
                    negated_neg_hit = len(self.neg) - neg_hit  # ¬C covers negatives that C didn't

                    # Now calculate confusion matrix for ¬C
                    neg_tp = negated_pos_hit
                    neg_fp = negated_neg_hit
                    neg_fn = pos_hit
                    neg_tn = neg_hit

                    # Recalculate precision, recall, and F1 based on new confusion matrix
                    neg_precision = neg_tp / (neg_tp + neg_fp) if (neg_tp + neg_fp) > 0 else 0.0
                    neg_recall = neg_tp / (neg_tp + neg_fn) if (neg_tp + neg_fn) > 0 else 0.0
                    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (
                                                                                                          neg_precision + neg_recall) > 0 else 0.0

                    negated_results.append(
                        (negated_concept, neg_f1, neg_precision, neg_recall, negated_pos_hit, negated_neg_hit))

            # Sort by F1 descending, then by precision
            all_results = results + negated_results
            all_results.sort(key=lambda x: (-x[1], -x[1]))
            
            # Cache results (limit to top 50 to avoid memory bloat)
            self._qualified_cardinality_template_cache[template_key] = tuple(all_results[:1])
            
            return all_results[:1]
            
        except Exception as e:
            print('Warning! Error in oracle_qualified_cardinality_suggestor' + str(e))
            return all_results


    def _literal_from_binding(value: str, datatype_iri: Optional[str]) -> Optional[OWLLiteral]:
        """Convert SPARQL binding to OWLLiteral for basic ALC(D) datatypes."""
        try:
            if datatype_iri is None:
                return OWLLiteral(value)
            if datatype_iri.endswith("#boolean"):
                return OWLLiteral(value.lower() in {"true", "1"})
            if datatype_iri.endswith("#int") or datatype_iri.endswith("#integer"):
                return OWLLiteral(int(value))
            if datatype_iri.endswith("#decimal") or datatype_iri.endswith("#double") or datatype_iri.endswith("#float"):
                return OWLLiteral(float(value))
            return OWLLiteral(value)
        except Exception:
            return None

    @staticmethod
    def _numeric_from_binding(value: str, datatype_iri: Optional[str]):
        if datatype_iri is None:
            return None
        try:
            if datatype_iri.endswith("#int") or datatype_iri.endswith("#integer"):
                return int(value)
            if datatype_iri.endswith("#decimal") or datatype_iri.endswith("#double") or datatype_iri.endswith("#float"):
                return float(value)
        except Exception:
            return None
        return None


    def g(self, template: OWLClassExpression) -> Set[OWLClassExpression]:

        template = get_top_level_dnf(template)

        # Handel the case Top
        classes = []
        roles = []
        neg_classes = []
        results = []

        # Since the orcal can not provide the correct tp and fp for the class expression which contain union and position_marker, we need to handle this case specially by decomposing the template into two parts:
        # Normalise the template to DNF like: {X(contain_position_marker) ⊔ A ⊔ B ⊔ C}
        # Caculate the tp1 and fp1 for {A ⊔ B ⊔ C}
        # Caculate the tp2 and fp2 for the {X(contain_position_marker) ⊓ ¬A ⊓ ¬B ⊓ ¬C} part
        # The final tp and fp for {X(contain_position_marker) ⊔ A ⊔ B ⊔ C} is tp1 + tp2 and fp1 + fp2

        if isinstance(template, OWLObjectUnionOf):
            tp_out_of_positionparker, fp_out_of_positionparker = 0, 0
            ops = list(template.operands())

            marker_ops = [op for op in ops if self._contains_context_marker(op)]
            fixed_ops = [op for op in ops if not self._contains_context_marker(op)]

            if marker_ops:
                marker_context = marker_ops[0] if len(marker_ops) == 1 else OWLObjectUnionOf(marker_ops)

                if fixed_ops:
                    fixed_union = fixed_ops[0] if len(fixed_ops) == 1 else OWLObjectUnionOf(fixed_ops)
                    _, _, _, tp_out_of_positionparker, fp_out_of_positionparker = evaluate_with_confusion_matrix(
                        fixed_union, self.kb, self.pos, self.neg
                    )

                negated_fixed = [OWLObjectComplementOf(o) for o in fixed_ops]
                combined = OWLObjectIntersectionOf([marker_context] + negated_fixed) if negated_fixed else marker_context

                classes = self.oracle_class_suggestor(combined)
                neg_classes = self.oracle_negation_class_suggestor(combined) if self.enable_negation else []
                roles = self.oracle_role_suggestor(combined)
                inversed_roles = self.oracle_inverse_role_suggestor(combined) if self.enable_inverse_roles else []
                data_roles = self.oracle_data_role_suggestor(combined) if self.enable_data_properties else []


            else:
                classes = self.oracle_class_suggestor(template)
                neg_classes = self.oracle_negation_class_suggestor(template) if self.enable_negation else []
                roles = self.oracle_role_suggestor(template)
                inversed_roles = self.oracle_inverse_role_suggestor(template) if self.enable_inverse_roles else []
                data_roles = self.oracle_data_role_suggestor(template) if self.enable_data_properties else []



            for i, c in enumerate(classes):
                tp_new = c[4] + tp_out_of_positionparker
                fp_new = c[5] + fp_out_of_positionparker
                precision_new = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0.0
                recall_new = tp_new / len(self.pos) if len(self.pos) > 0 else 0.0
                f1_new = 2 * precision_new * recall_new / (precision_new + recall_new) if (precision_new + recall_new) > 0 else 0.0
                classes[i] = (c[0], f1_new, precision_new, recall_new, tp_new, fp_new)

            for i, nc in enumerate(neg_classes):
                tp_new = nc[4] + tp_out_of_positionparker
                fp_new = nc[5] + fp_out_of_positionparker
                precision_new = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0.0
                recall_new = tp_new / len(self.pos) if len(self.pos) > 0 else 0.0
                f1_new = 2 * precision_new * recall_new / (precision_new + recall_new) if (precision_new + recall_new) > 0 else 0.0
                neg_classes[i] = (nc[0], f1_new, precision_new, recall_new, tp_new, fp_new)

            for i, r in enumerate(roles):
                tp_new = r[4] + tp_out_of_positionparker
                fp_new = r[5] + fp_out_of_positionparker
                precision_new = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0.0
                recall_new = tp_new / len(self.pos) if len(self.pos) > 0 else 0.0
                f1_new = 2 * precision_new * recall_new / (precision_new + recall_new) if (precision_new + recall_new) > 0 else 0.0
                roles[i] = (r[0], f1_new, precision_new, recall_new, tp_new, fp_new)

            for i, ir in enumerate(inversed_roles):
                tp_new = ir[4] + tp_out_of_positionparker
                fp_new = ir[5] + fp_out_of_positionparker
                precision_new = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0.0
                recall_new = tp_new / len(self.pos) if len(self.pos) > 0 else 0.0
                f1_new = 2 * precision_new * recall_new / (precision_new + recall_new) if (precision_new + recall_new) > 0 else 0.0
                inversed_roles[i] = (ir[0], f1_new, precision_new, recall_new, tp_new, fp_new)

            for i, dr in enumerate(data_roles):
                tp_new = dr[4] + tp_out_of_positionparker
                fp_new = dr[5] + fp_out_of_positionparker
                precision_new = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0.0
                recall_new = tp_new / len(self.pos) if len(self.pos) > 0 else 0.0
                f1_new = 2 * precision_new * recall_new / (precision_new + recall_new) if (precision_new + recall_new) > 0 else 0.0
                data_roles[i] = (dr[0], f1_new, precision_new, recall_new, tp_new, fp_new)

        else:
            classes = self.oracle_class_suggestor(template)
            neg_classes = self.oracle_negation_class_suggestor(template) if self.enable_negation else []
            roles = self.oracle_role_suggestor(template)
            inversed_roles = self.oracle_inverse_role_suggestor(template) if self.enable_inverse_roles else []
            data_roles = self.oracle_data_role_suggestor(template) if self.enable_data_properties else []



        # m*(T, D) - Class
        for c in classes:
            expr = self.m_star(template, c[0])
            results.append((expr, c[1], c[2], c[3], c[4], c[5]))

        # m*(T, ¬D) - Negation
        if self.enable_negation:
             for c in neg_classes:
                 neg_c = OWLObjectComplementOf(c[0])
                 expr = self.m_star(template, neg_c)
                 results.append((expr, c[1], c[2], c[3], c[4], c[5]))

        # m*(T, ∃ r . ⊤) - Object Property Expressions
        for r in roles:
             exists = OWLObjectSomeValuesFrom(property=r[0], filler=OWLThing)
             expr = self.m_star(template, exists)
             results.append((expr, r[1], r[2], r[3], r[4], r[5]))

        # I: m*(T, ∃ r⁻ . ⊤) - Inverse Role Expressions
        if self.enable_inverse_roles:
             for r in inversed_roles:
                 exists = OWLObjectSomeValuesFrom(property=r[0], filler=OWLThing)
                 expr = self.m_star(template, exists)
                 results.append((expr, r[1], r[2], r[3], r[4], r[5]))

        # D: m*(T, ∃ r . rdfs:literal) - Data Property Expressions
        if self.enable_data_properties:
             for r in data_roles:
                 exists = OWLDataSomeValuesFrom(property=r[0], filler=TopOWLDatatype)
                 expr = self.m_star(template, exists)
                 results.append((expr, r[1], r[2], r[3], r[4], r[5]))



        return results


    def g_for_data(self, template: OWLClassExpression, negated) -> Set[OWLClassExpression]:

        template = get_top_level_dnf(template)

        # Handel the case Top
        data_exprs = []
        results = []
        negated_results = []
        all_results = []

        # Since the orcal can not provide the correct tp and fp for the class expression which contain union and position_marker, we need to handle this case specially by decomposing the template into two parts:
        # Normalise the template to DNF like: {X(contain_position_marker) ⊔ A ⊔ B ⊔ C}
        # Caculate the tp1 and fp1 for {A ⊔ B ⊔ C}
        # Caculate the tp2 and fp2 for the {X(contain_position_marker) ⊓ ¬A ⊓ ¬B ⊓ ¬C} part
        # The final tp and fp for {X(contain_position_marker) ⊔ A ⊔ B ⊔ C} is tp1 + tp2 and fp1 + fp2

        if isinstance(template, OWLObjectUnionOf):
            tp_out_of_positionparker, fp_out_of_positionparker = 0, 0
            ops = list(template.operands())

            marker_ops = [op for op in ops if self._contains_context_marker(op)]
            fixed_ops = [op for op in ops if not self._contains_context_marker(op)]

            if marker_ops:
                marker_context = marker_ops[0] if len(marker_ops) == 1 else OWLObjectUnionOf(marker_ops)

                if fixed_ops:
                    fixed_union = fixed_ops[0] if len(fixed_ops) == 1 else OWLObjectUnionOf(fixed_ops)
                    _, _, _, tp_out_of_positionparker, fp_out_of_positionparker = evaluate_with_confusion_matrix(
                        fixed_union, self.kb, self.pos, self.neg
                    )

                negated_fixed = [OWLObjectComplementOf(o) for o in fixed_ops]
                combined = OWLObjectIntersectionOf([marker_context] + negated_fixed) if negated_fixed else marker_context

                if template != CONTEXT_POSITION_MARKER:
                    data_exprs = self.oracle_data_value_suggestor(combined)
            else:
                if template != CONTEXT_POSITION_MARKER:
                    data_exprs = self.oracle_data_value_suggestor(template)

            if template != CONTEXT_POSITION_MARKER:
                for i, de in enumerate(data_exprs):
                    tp_new = de[4] + tp_out_of_positionparker
                    fp_new = de[5] + fp_out_of_positionparker
                    precision_new = tp_new / (tp_new + fp_new) if (tp_new + fp_new) > 0 else 0.0
                    recall_new = tp_new / len(self.pos) if len(self.pos) > 0 else 0.0
                    f1_new = 2 * precision_new * recall_new / (precision_new + recall_new) if (precision_new + recall_new) > 0 else 0.0
                    data_exprs[i] = (de[0], f1_new, precision_new, recall_new, tp_new, fp_new)

        else:
            if template != CONTEXT_POSITION_MARKER:
                data_exprs = self.oracle_data_value_suggestor(template)

        # m(T, D data restrictions)
        if template != CONTEXT_POSITION_MARKER:

            for d in data_exprs:
                expr = self.m_star(template, d[0])
                # Keep current oracle flow but restore semantically correct
                # data existentials before returning candidates.
                expr = self._restore_data_existentials(expr)
                results.append((expr, d[1], d[2], d[3], d[4], d[5]))

            if negated: # For now, need to change because now we just negated the best F1 score conceot, need to check all the possiable candidate concept to find the one that after negation can return the best F1 score, need to change in oracle_data_value_suggestor function!
                for child in results:
                    concept_expr, f1, precision, recall, pos_hit, neg_hit = child

                    negated_concept = OWLObjectComplementOf(concept_expr)

                    # Recalculate coverage for negated concept (complement coverage)
                    negated_pos_hit = len(self.pos) - pos_hit  # ¬C covers positives that C didn't
                    negated_neg_hit = len(self.neg) - neg_hit  # ¬C covers negatives that C didn't

                    # Now calculate confusion matrix for ¬C
                    neg_tp = negated_pos_hit
                    neg_fp = negated_neg_hit
                    neg_fn = pos_hit
                    neg_tn = neg_hit

                    # Recalculate precision, recall, and F1 based on new confusion matrix
                    neg_precision = neg_tp / (neg_tp + neg_fp) if (neg_tp + neg_fp) > 0 else 0.0
                    neg_recall = neg_tp / (neg_tp + neg_fn) if (neg_tp + neg_fn) > 0 else 0.0
                    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (
                                                                                                          neg_precision + neg_recall) > 0 else 0.0

                    negated_results.append(
                        (negated_concept, neg_f1, neg_precision, neg_recall, negated_pos_hit, negated_neg_hit))

            # Sort by F1 descending, then by precision
            all_results = results + negated_results
            all_results.sort(key=lambda x: (-x[1], -x[1]))

        return all_results[:1]


    def m_star(self,
               template: OWLClassExpression,
               expr: OWLClassExpression) -> OWLClassExpression:
        """
        Replace CONTEXT_POSITION_MARKER (μ) in template with expr.
        """

        # Base case: found the marker
        if template == CONTEXT_POSITION_MARKER:
            return expr

        # Atomic classes (nothing to replace)neg
        if isinstance(template, OWLClass):
            return template

        # Complement
        if self.enable_negation and isinstance(template, OWLObjectComplementOf):
            return OWLObjectComplementOf(
                self.m_star(template.get_operand(), expr)
            )

        # Intersection
        if isinstance(template, OWLObjectIntersectionOf):
            return OWLObjectIntersectionOf([
                self.m_star(op, expr)
                for op in template.operands()
            ])

        # Union
        if isinstance(template, OWLObjectUnionOf):
            return OWLObjectUnionOf([
                self.m_star(op, expr)
                for op in template.operands()
            ])

        # Existential owl object property
        if isinstance(template, OWLObjectSomeValuesFrom):
            return OWLObjectSomeValuesFrom(
                property=template.get_property(),
                filler=self.m_star(template.get_filler(), expr)
            )

        # Existential data property
        if isinstance(template, OWLDataSomeValuesFrom):
            return OWLDataSomeValuesFrom(
                property=template.get_property(),
                filler=self.m_star(template.get_filler(), expr)
            )

        # Universal
        if isinstance(template, OWLObjectAllValuesFrom):
            return OWLObjectAllValuesFrom(
                property=template.get_property(),
                filler=self.m_star(template.get_filler(), expr)
            )

        # Fallback
        return template

    @staticmethod
    def _is_data_filler(expr) -> bool:
        """Return True when *expr* is a datatype-oriented filler."""
        return expr == TopOWLDatatype or isinstance(expr, (OWLDatatypeRestriction, OWLDataOneOf))

    def _restore_data_existentials(self, expr):
        """Rewrite invalid OWLObjectSomeValuesFrom(prop, dataRange) back to OWLDataSomeValuesFrom.

        This preserves your current marker-based oracle generation path while ensuring
        emitted candidates use the correct constructor for data properties.
        """
        if isinstance(expr, OWLObjectSomeValuesFrom):
            repaired_filler = self._restore_data_existentials(expr.get_filler())
            prop = expr.get_property()
            if isinstance(prop, OWLObjectProperty) and self._is_data_filler(repaired_filler):
                return OWLDataSomeValuesFrom(property=OWLDataProperty(prop.iri), filler=repaired_filler)
            return OWLObjectSomeValuesFrom(property=prop, filler=repaired_filler)

        if isinstance(expr, OWLDataSomeValuesFrom):
            repaired_filler = self._restore_data_existentials(expr.get_filler())
            return OWLDataSomeValuesFrom(property=expr.get_property(), filler=repaired_filler)

        if isinstance(expr, OWLObjectIntersectionOf):
            return OWLObjectIntersectionOf([self._restore_data_existentials(op) for op in expr.operands()])

        if isinstance(expr, OWLObjectUnionOf):
            return OWLObjectUnionOf([self._restore_data_existentials(op) for op in expr.operands()])

        if isinstance(expr, OWLObjectComplementOf):
            return OWLObjectComplementOf(self._restore_data_existentials(expr.get_operand()))

        return expr

    def rho_star(self, concept: OWLClassExpression, template: OWLClassExpression):

        results = []

        # TOP
        if concept == OWLThing:
            results.extend(self.g(template))

        # ∃ r . X
        if isinstance(concept, OWLObjectSomeValuesFrom):
            r = concept.get_property()
            X = concept.get_filler()

            new_template = self.m_star(template,
                                       OWLObjectSomeValuesFrom(r, CONTEXT_POSITION_MARKER))

            # recursive refinement
            results.extend(self.rho_star(X, new_template))

            # special case: ∀ r . X
            additional_concept_forall = self.m_star(template,OWLObjectAllValuesFrom(r, X))
            f1, precision, recall, tp, fp = evaluate_with_confusion_matrix(additional_concept_forall, self.kb, self.pos, self.neg)
            results.append((additional_concept_forall, f1, precision, recall, tp, fp))

            # Q: qualified cardinality - find best N for ≥N r.X
            if self.enable_qualified_cardinality:

                qualified_cards = self.oracle_qualified_cardinality_suggestor(template, r, X, True)
                results.extend(qualified_cards)

        # ∀ r . X
        if isinstance(concept, OWLObjectAllValuesFrom):
            r = concept.get_property()
            X = concept.get_filler()

            new_template = self.m_star(template,
                                       OWLObjectAllValuesFrom(r, CONTEXT_POSITION_MARKER))

            results.extend(self.rho_star(X, new_template))

        # ¬ X
        if self.enable_negation and isinstance(concept, OWLObjectComplementOf):
            X = concept.get_operand()

            new_template = self.m_star(template,
                                       OWLObjectComplementOf(CONTEXT_POSITION_MARKER))

            results.extend(self.rho_star(X, new_template))

        # X1 ⊓ X2 ⊓ ... ⊓ Xn
        if isinstance(concept, OWLObjectIntersectionOf):
            operands = list(concept.operands())
            for i, operand in enumerate(operands):
                other_ops = [op for j, op in enumerate(operands) if j != i]
                local_template = CONTEXT_POSITION_MARKER if not other_ops else OWLObjectIntersectionOf(
                    [CONTEXT_POSITION_MARKER] + other_ops
                )
                results.extend(self.rho_star(
                    operand,
                    self.m_star(template, local_template)
                ))

        # X1 ⊔ X2 ⊔ ... ⊔ Xn
        if isinstance(concept, OWLObjectUnionOf):
            operands = list(concept.operands())
            for i, operand in enumerate(operands):
                other_ops = [op for j, op in enumerate(operands) if j != i]
                local_template = CONTEXT_POSITION_MARKER if not other_ops else OWLObjectUnionOf(
                    [CONTEXT_POSITION_MARKER] + other_ops
                )
                results.extend(self.rho_star(
                    operand,
                    self.m_star(template, local_template)
                ))

        # not ⊤ or ⊥ concepts
        if not self.is_top_or_bottom(concept):

            results.extend(self.g(
                self.m_star(template,
                            OWLObjectIntersectionOf([concept, CONTEXT_POSITION_MARKER]))
            ))

            results.extend(self.g(
                self.m_star(template,
                            OWLObjectUnionOf([concept, CONTEXT_POSITION_MARKER]))
            ))

        # ∃ r . X for data property
        if isinstance(concept, OWLDataSomeValuesFrom):
            r = concept.get_property()
            r_for_template = OWLObjectProperty(r.iri.str)
            X = concept.get_filler()

            if X == TopOWLDatatype:
                new_template = self.m_star(template,
                                           OWLObjectSomeValuesFrom(r_for_template, CONTEXT_POSITION_MARKER))
                results.extend(self.g_for_data(new_template,negated=True))

        return results


    def refine(self, concept: OWLClassExpression):
        total_refinement = 0
        concept = get_top_level_dnf(concept)
        template = CONTEXT_POSITION_MARKER

        candidates = self.rho_star(concept, template)

        parent_key = str(concept)
        if parent_key not in self._score_cache:
            parent_f1, parent_precision, parent_recall, tp, fp = evaluate_with_confusion_matrix(
                concept, self.kb, self.pos, self.neg
            )
            self._score_cache[parent_key] = parent_f1
        else:
            parent_f1 = self._score_cache[parent_key]

        refinements = []

        for ref in candidates:

            # print(f"Refinement candidate: {self._renderer.render(ref[0])} F1-score: {ref[1]} r-score: {self.refinement_score(ref[0], ref[1], ref[4])} tp: {ref[4]} fp: {ref[5]}")
            total_refinement = total_refinement + 1
            child_f1 = ref[1]
            child_rscore = self.refinement_score(ref[0], ref[1], ref[4])


            # f1_REAL, precision_REAL, recall_REAL, covered_pos, _ = evaluate_with_confusion_matrix(ref[0], self.kb, self.pos, self.neg)
            # if child_f1 == f1_REAL:
            #     print(f"F1 matched! concept: {self._renderer.render(ref[0])} cscore: {child_f1} rscore: {child_rscore}")
            # else:
            #     REAL_rscore = self.refinement_score(ref[0], f1_REAL, covered_pos)
            #     REAL_cscore = f1_REAL
            #     print(f"F1 not matched! concept: {self._renderer.render(ref[0])} cscore: {child_f1} rscore: {child_rscore} REAL cscore: {REAL_cscore}  REAL rscore: {REAL_rscore}")

            child_key = str(ref[0])
            self._score_cache[child_key] = child_f1

            # only keep refinements whose F1 improves over the parent
            if child_f1 > parent_f1 or self.added_role(concept, ref[0]):
                # print(f"Refinement candidate: {self._renderer.render(ref[0])} F1-score: {ref[1]} r-score: {self.refinement_score(ref[0], ref[1], ref[4])} tp: {ref[4]} fp: {ref[5]}")
                ref = ref[:2] + (child_rscore,) + ref[2:]
                refinements.append(ref)

        return refinements, total_refinement

    def concept_length(self, concept: OWLClassExpression) -> int:
        """Calculate concept length (number of symbols)."""
        # ALC
        if isinstance(concept, OWLClass):
            return 1
        if isinstance(concept, OWLObjectComplementOf):
            return 1 + self.concept_length(concept.get_operand())
        if isinstance(concept, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
            return 1 + sum(self.concept_length(op) for op in concept.operands())
        if isinstance(concept, (OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom)):
            return 2 + self.concept_length(concept.get_filler())

        # Q
        if isinstance(concept, (OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality)):
            return 2 + self.concept_length(concept.get_filler())
        # D
        if isinstance(concept, OWLDataSomeValuesFrom):
            return 2 + self.concept_length(concept.get_filler())
       
        return 1

    def refinement_score(self, concept: OWLClassExpression, f1: float, tp: int) -> float:
        """
        Calculate refinement score = F1 - length * penalty.
        Filters out concepts with tp <= 1.
        """
        if tp <= 1:
            return 0.0
        length = self.concept_length(concept)
        return f1 - length * self.length_penalty

    def count_quantifiers(self, concept: OWLClassExpression) -> int:
        """Count number of existential/universal quantifiers in a concept."""
        if isinstance(concept, (OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom)):
            return 1 + self.count_quantifiers(concept.get_filler())
        if isinstance(concept, OWLDataSomeValuesFrom):
            return 1
        if isinstance(concept, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
            return sum(self.count_quantifiers(op) for op in concept.operands())
        if isinstance(concept, OWLObjectComplementOf):
            return self.count_quantifiers(concept.get_operand())
        return 0

    def added_role(self, parent: OWLClassExpression, child: OWLClassExpression) -> bool:
        """Check if child was derived from parent by adding a role (∃r or ∀r)."""
        return self.count_quantifiers(child) > self.count_quantifiers(parent)

    def is_top_or_bottom(self, concept: OWLClassExpression) -> bool:
        """Check if concept is ⊤ or ⊥ (including the data top type rdfs:Literal)."""
        return concept == OWLThing or concept == OWLNothing or concept == TopOWLDatatype

    def compute_f1(self, concept: OWLClassExpression, kb: KnowledgeBase,
                   pos: Set, neg: Set) -> Tuple[float, float, float, int]:
        """Compute F1, precision, recall, and coverage counts for a concept."""
        try:
            instances = kb.individuals_set(concept)
            tp = len(instances.intersection(pos))
            fp = len(instances.intersection(neg))
            fn = len(pos - instances)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return f1, precision, recall, tp, fp
        except Exception as e:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
    
    def evaluate_with_confusion_matrix(self, concept: OWLClassExpression, pos: Set, neg: Set):

        # Special case: owl:Thing may not be explicitly asserted as rdf:type owl:Thing
        if concept == OWLThing:
            tp = len(pos)
            fp = len(neg)
            fn = 0
            tn = 0
            confusion_matrix = {
                "tp": tp,
                "fp": fp,
                "fn": 0,
                "tn": 0,
            }

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = compute_f1_score_from_confusion_matrix(confusion_matrix=confusion_matrix)

        else:
            sparql_query = owl_expression_to_sparql_with_confusion_matrix(expression=concept,
                                                                          positive_examples=pos,
                                                                          negative_examples=neg)
            bindings = self.kb.query(sparql_query).json()["results"]["bindings"]
            assert len(bindings) == 1
            bindings = bindings.pop()

            confusion_matrix = {k: v["value"] for k, v in bindings.items()}
            tp = tp=int(confusion_matrix["tp"])
            fp = fp=int(confusion_matrix["fp"])
            fn = fn=int(confusion_matrix["fn"])
            tn = tn=int(confusion_matrix["tn"])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = compute_f1_score_from_confusion_matrix(confusion_matrix=confusion_matrix)

        return f1, precision, recall, tp, fp
