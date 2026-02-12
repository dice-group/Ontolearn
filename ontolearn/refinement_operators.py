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

from itertools import chain
import random
from typing import DefaultDict, Dict, Set, Optional, Iterable, List, Type, Final, Generator, Tuple

from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectIntersectionOf, \
    OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, OWLObjectUnionOf, OWLClass, \
    OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLDataSomeValuesFrom, \
    OWLDatatypeRestriction, OWLDataHasValue, OWLObjectExactCardinality, OWLObjectOneOf
from owlapy.owl_individual import OWLIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLObjectPropertyExpression, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataPropertyExpression, OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer

from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter
from owlapy.providers import owl_datatype_max_inclusive_restriction, owl_datatype_min_inclusive_restriction
from owlapy.vocab import OWLFacet

from .abstracts import BaseRefinement, AbstractKnowledgeBase
from .concept_generator import ConceptGenerator
from .knowledge_base import KnowledgeBase
from .nero_utils import AtomicExpression, ExistentialQuantifierExpression, ClassExpression, \
    ComplementOfAtomicExpression, UniversalQuantifierExpression, UnionClassExpression, IntersectionClassExpression, Role
from .search import OENode
from owlapy import owl_expression_to_sparql
import requests





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
                 length_penalty: float = 0.01):
        """
        Initialize PruneCEL recursive refinement operator.
        
        Args:
            knowledge_base: Knowledge base containing the ontology and instances
            max_concepts: Maximum number of concepts to generate per refinement
            random_seed: Random seed for reproducibility
            use_inverse: Whether to use inverse properties in refinements
            use_negation: Whether to generate negated concepts
            use_all_constructor: Whether to use universal restrictions (∀r.C)
            use_card_restrictions: Whether to use cardinality restrictions (≥n r.C)
            min_cardinality_restriction: Minimum cardinality value
            max_cardinality_restriction: Maximum cardinality value
            sparql_endpoint: Optional SPARQL endpoint URL for smart suggestion.
                           If provided, uses SPARQL-based candidate selection.
                           If None, falls back to local KB-based selection.
            length_penalty: Penalty for concept length in refinement score (default: 0.01)
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
        
        # High-precision fragment collection
        self.high_precision_fragments = []  # Store high-precision, low-recall concepts
        self.covered_positives = set()  # Track which positives are covered by fragments
        self.precision_threshold = 1.0  # Minimum precision for a fragment
        self.recall_threshold = 0.5  # Maximum recall for a fragment (low recall)
        
        # Refinement score caching
        self._score_cache = {}  # Cache for parent concept scores
        
        # Renderer for debug output
        self._renderer = DLSyntaxObjectRenderer()

    def sparql(self, query: str) -> List[Dict[str, str]]:
        headers = {
        "Accept": "application/sparql-results+json"
        }

        response = requests.post(
            self.sparql_endpoint,
            data={"query": query},
            headers=headers
        )

        response.raise_for_status()
        return response.json()["results"]["bindings"]


    def set_input_examples(self, pos: frozenset, neg: frozenset):
        """Set positive and negative examples for coverage-guided refinement."""
        assert isinstance(pos, frozenset)
        self.pos = {i for i in pos}
        self.neg = {i for i in neg}
        # Clear pool and fragments when setting new examples
        self.top_refinements = set()
        self.high_precision_fragments = []
        self.covered_positives = set()

    
    def refine_top_concept(self, concept: OWLClassExpression):

        if concept != OWLThing:
            return

        examples = list(self.pos) + list(self.neg)
        if not examples:
            return

        values_clause = " ".join(f"<{ind.iri.as_str()}>" for ind in examples)

        refinements = []

        # -----------------------------------
        # Named atomic concepts
        # -----------------------------------
        class_query = f"""
        SELECT DISTINCT ?class
        WHERE {{
            VALUES ?ind {{ {values_clause} }}
            ?ind a ?class .
            FILTER(?class != <http://www.w3.org/2002/07/owl#NamedIndividual>)
            # FILTER(?class != <http://www.w3.org/2002/07/owl#Thing>)
        }}
        """


        class_results = self.sparql(class_query)

        for row in class_results:
            cls = OWLClass(row["class"]["value"])
            
            # Check coverage of this class
            instances = set(self.kb.individuals(cls))
            covers_pos = len(instances & self.pos) > 0
            covers_neg = len(instances & self.neg) > 0
            
            if covers_pos:
                # If it covers positives, add the class as-is
                refinements.append(cls)
            elif covers_neg:
                # If it only covers negatives, add its negation (which will cover positives)
                refinements.append(OWLObjectComplementOf(cls))

        # -----------------------------------
        # Existential restrictions ∃ μ . ⊤
        # -----------------------------------
        property_query = f"""
        SELECT DISTINCT ?prop
        WHERE {{
            VALUES ?ind {{ {values_clause} }}
            ?ind ?prop ?obj .
            FILTER(?prop != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
            FILTER(isIRI(?obj))
        }}
        """

        prop_results = self.sparql(property_query)

        for row in prop_results:
            prop = OWLObjectProperty(row["prop"]["value"])
            refinements.append(
                OWLObjectSomeValuesFrom(prop, OWLThing)
            )

        return refinements
    
    def refine_existential(self, concept: OWLClassExpression):
        """
        Refine existential restrictions ∃r.X
        
        Strategy:
        1. Recursively refine the filler X to get ∃r.X' (when X=⊤, gets ∃r.A, etc.)
        2. Add conjunctions: A ⊓ ∃r.X
        3. Add nested properties: ∃r.(∃s.⊤)
        """
        if not isinstance(concept, OWLObjectSomeValuesFrom):
            return
        
        prop = concept.get_property()
        filler = concept.get_filler()
        refinements = []
        
        examples = list(self.pos) + list(self.neg)
        if not examples:
            return
        
        values_clause = " ".join(f"<{ind.iri.as_str()}>" for ind in examples)
        
        # -----------------------------------
        # 1. ∃r.X' - Recursively refine filler
        # When filler=⊤, this queries for classes of property objects
        # When filler≠⊤, this refines the filler concept
        # -----------------------------------
        if filler == OWLThing:
            # Query for classes that objects of this property belong to
            class_query = f"""
            SELECT DISTINCT ?class
            WHERE {{
                VALUES ?ind {{ {values_clause} }}
                ?ind <{prop.iri.as_str()}> ?obj .
                ?obj a ?class .
                FILTER(?class != <http://www.w3.org/2002/07/owl#NamedIndividual>)
            }}
            """
            
            class_results = self.sparql(class_query)
            for row in class_results:
                cls = OWLClass(row["class"]["value"])
                new_concept = OWLObjectSomeValuesFrom(prop, cls)
                
                # Check coverage - only add if it covers at least one positive
                instances = set(self.kb.individuals(new_concept))
                covers_pos = len(instances & self.pos) > 0
                
                if covers_pos:
                    refinements.append(new_concept)
        else:
            # Recursively refine the filler
            refined_filler = self.refine(filler)
            # print(list(self._renderer.render(i) for i in refined_filler))  # Debug output
            # exit(0)
            for refined_filler in self.refine(filler):
                new_concept = OWLObjectSomeValuesFrom(prop, refined_filler)
                
                # Check coverage - only add if it covers at least one positive
                instances = set(self.kb.individuals(new_concept))
                covers_pos = len(instances & self.pos) > 0
                
                if covers_pos:
                    refinements.append(new_concept)
        
        # -----------------------------------
        # 2. A ⊓ ∃r.X - conjunction with named classes
        # -----------------------------------
        top_class_query = f"""
        SELECT DISTINCT ?class
        WHERE {{
            VALUES ?ind {{ {values_clause} }}
            ?ind a ?class .
            FILTER(?class != <http://www.w3.org/2002/07/owl#NamedIndividual>)
        }}
        """
        
        top_class_results = self.sparql(top_class_query)
        
        # Get original concept coverage
        original_instances = set(self.kb.individuals(concept))
        
        for row in top_class_results:
            cls = OWLClass(row["class"]["value"])
            new_concept = OWLObjectIntersectionOf([cls, concept])
            
            # Check coverage - only add if it covers positives AND changes coverage
            instances = set(self.kb.individuals(new_concept))
            if len(instances & self.pos) > 0 and instances != original_instances:
                refinements.append(new_concept)


        # -----------------------------------
        # 2. A ⊓ ∃r.X - conjunction with named classes
        # -----------------------------------
        top_class_query = f"""
        SELECT DISTINCT ?class
        WHERE {{
            VALUES ?ind {{ {values_clause} }}
            ?ind a ?class .
            FILTER(?class != <http://www.w3.org/2002/07/owl#NamedIndividual>)
        }}
        """
        
        top_class_results = self.sparql(top_class_query)
        
        # Get original concept coverage
        original_instances = set(self.kb.individuals(concept))
        
        for row in top_class_results:
            cls = OWLClass(row["class"]["value"])
            new_concept = OWLObjectUnionOf([cls, concept])
            
            # Check coverage - only add if it covers positives AND changes coverage
            instances = set(self.kb.individuals(new_concept))
            if len(instances & self.pos) > 0 and instances != original_instances:
                refinements.append(new_concept)

        
        
        # -----------------------------------
        # 3. ∃r.(Filler ⊓ ∃s.⊤) - nested existential with filler preserved
        # Query for properties that objects of r have
        # This adds nested properties while preserving the original filler constraint
        # -----------------------------------
        # nested_prop_query = f"""
        # SELECT DISTINCT ?prop2
        # WHERE {{
        #     VALUES ?ind {{ {values_clause} }}
        #     ?ind <{prop.iri.as_str()}> ?obj .
        #     ?obj ?prop2 ?obj2 .
        #     FILTER(?prop2 != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
        #     FILTER(?prop2 != <{prop.iri.as_str()}>)  # Avoid nested same property
        #     FILTER(isIRI(?obj2))
        # }}
        # """
        
        # nested_prop_results = self.sparql(nested_prop_query)
        # for row in nested_prop_results:
        #     inner_prop = OWLObjectProperty(row["prop2"]["value"])
        #     inner_exist = OWLObjectSomeValuesFrom(inner_prop, OWLThing)
        #     # Combine original filler with nested property: Filler ⊓ ∃s.⊤
        #     combined_filler = OWLObjectIntersectionOf([filler, inner_exist])
        #     new_concept = OWLObjectSomeValuesFrom(prop, combined_filler)
            
        #     # Check coverage - only add if it covers at least one positive
        #     instances = set(self.kb.individuals(new_concept))
        #     covers_pos = len(instances & self.pos) > 0
            
        #     if covers_pos:
                # refinements.append(new_concept)


        # -----------------------------------
        # 4. ∃r.(∃s.Filler) - nested existential with filler preserved
        # Query for properties that objects of r have
        # This adds nested properties while preserving the original filler constraint
        # -----------------------------------
        nested_prop_query = f"""
        SELECT DISTINCT ?prop2
        WHERE {{
            VALUES ?ind {{ {values_clause} }}
            ?ind <{prop.iri.as_str()}> ?obj .
            ?obj ?prop2 ?obj2 .
            FILTER(?prop2 != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
            FILTER(?prop2 != <{prop.iri.as_str()}>)  # Avoid nested same property
            FILTER(isIRI(?obj2))
        }}
        """
        
        nested_prop_results = self.sparql(nested_prop_query)
        for row in nested_prop_results:
            inner_prop = OWLObjectProperty(row["prop2"]["value"])
            inner_exist = OWLObjectSomeValuesFrom(inner_prop, filler)
            new_concept = OWLObjectSomeValuesFrom(prop, inner_exist)
            
            # Check coverage - only add if it covers at least one positive
            instances = set(self.kb.individuals(new_concept))
            covers_pos = len(instances & self.pos) > 0
            
            if covers_pos:
                refinements.append(new_concept)
        
        return refinements
    

    def refine_intersection(self, concept: OWLClassExpression):
        """Refine intersection concepts A⊓B by refining each operand."""
        if not isinstance(concept, OWLObjectIntersectionOf):
            return
        
        refinements = []
        for operand in concept.operands():
            for refined_operand in self.refine(operand):
                # Replace the operand with its refinement
                new_operands = [refined_operand if op == operand else op for op in concept.operands()]
                new_concept = OWLObjectIntersectionOf(new_operands)
                
                # Check coverage - only add if it covers at least one positive
                instances = set(self.kb.individuals(new_concept))
                covers_pos = len(instances & self.pos) > 0
                
                if covers_pos:
                    refinements.append(new_concept)
        
        return refinements
    
    def refine_union(self, concept: OWLClassExpression):
        """Refine union concepts A⊔B by refining each operand."""
        if not isinstance(concept, OWLObjectUnionOf):
            return
        
        refinements = []
        for operand in concept.operands():
            for refined_operand in self.refine(operand):
                # Replace the operand with its refinement
                new_operands = [refined_operand if op == operand else op for op in concept.operands()]
                new_concept = OWLObjectUnionOf(new_operands)
                
                # Check coverage - only add if it covers at least one positive
                instances = set(self.kb.individuals(new_concept))
                covers_pos = len(instances & self.pos) > 0
                
                if covers_pos:
                    refinements.append(new_concept)
        
        return refinements
    
    def refine_any_concept(self, concept: OWLClassExpression):
        """Refine named atomic concepts A by adding conjunctions and disjunctions."""
        # if not isinstance(concept, OWLClass):
        #     return
        
        refinements = []
        
        examples = list(self.pos) + list(self.neg)
        if not examples:
            return
        
        values_clause = " ".join(f"<{ind.iri.as_str()}>" for ind in examples)
        
        # Build filter to exclude current concept (only for OWLClass)
        concept_filter = ""
        if isinstance(concept, OWLClass):
            concept_filter = f"FILTER(?class != <{concept.iri.as_str()}>)"
        
        # Add conjunctions with other classes
        class_query = f"""
        SELECT DISTINCT ?class
        WHERE {{
            VALUES ?ind {{ {values_clause} }}
            ?ind a ?class .
            FILTER(?class != <http://www.w3.org/2002/07/owl#NamedIndividual>)
            FILTER(?class != <http://www.w3.org/2002/07/owl#Thing>)
            {concept_filter}
        }}
        """
        
        class_results = self.sparql(class_query)
        
        # Get coverage of original concept for comparison
        original_instances = set(self.kb.individuals(concept))
        
        for row in class_results:
            cls = OWLClass(row["class"]["value"])

            # Try conjunction: concept ⊓ cls
            new_concept = OWLObjectIntersectionOf([concept, cls])
            instances = set(self.kb.individuals(new_concept))
            # Only add if it covers positives AND changes the coverage
            if len(instances & self.pos) > 0 and instances != original_instances:
                refinements.append(new_concept)

            # Try disjunction: concept ⊔ cls  
            new_concept = OWLObjectUnionOf([concept, cls])
            instances = set(self.kb.individuals(new_concept))
            # Only add if it covers positives AND changes the coverage
            if len(instances & self.pos) > 0 and instances != original_instances:
                refinements.append(new_concept)

        #BUG if we limit at to the above version, we will never get out of the loop of refining the same named classes over and over (A⊓B⊓C⊓D, A⊔C⊔B⊔D, etc.)
        #  because we never add new properties to break out of it. We need to add a refinement that adds a new property restriction (∃r.⊤) to create 
        # new patterns that can then be refined further.  T
        
        # -----------------------------------
        # Add concept ⊓ ∃r.⊤ to break out of intersection/union loop
        # -----------------------------------
        property_query = f"""
        SELECT DISTINCT ?prop
        WHERE {{
            VALUES ?ind {{ {values_clause} }}
            ?ind ?prop ?obj .
            FILTER(?prop != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
            FILTER(isIRI(?obj))
        }}
        """
        
        prop_results = self.sparql(property_query)
        
        for row in prop_results:
            prop = OWLObjectProperty(row["prop"]["value"])
            exist = OWLObjectSomeValuesFrom(prop, OWLThing)
            
            # Try conjunction: concept ⊓ ∃r.⊤
            new_concept = OWLObjectIntersectionOf([concept, exist])
            instances = set(self.kb.individuals(new_concept))
            if len(instances & self.pos) > 0 and instances != original_instances:
                refinements.append(new_concept)
            
            # Try disjunction: concept ⊔ ∃r.⊤
            new_concept = OWLObjectUnionOf([concept, exist])
            instances = set(self.kb.individuals(new_concept))
            if len(instances & self.pos) > 0 and instances != original_instances:
                refinements.append(new_concept)
        
        # -----------------------------------
        # MOST IMPORTANT - Generate ∃r.concept and ∃r.¬concept
        # This creates patterns like ∃hasChild.Male (has a male child)
        # Query for inverse relationships - which individuals have relationships TO our examples
        # -----------------------------------
        inverse_property_query = f"""
        SELECT DISTINCT ?prop
        WHERE {{
            VALUES ?target {{ {values_clause} }}
            ?subject ?prop ?target .
            FILTER(?prop != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
            FILTER(isIRI(?subject))
        }}
        """
        
        inv_prop_results = self.sparql(inverse_property_query)
        
        for row in inv_prop_results:
            prop = OWLObjectProperty(row["prop"]["value"])
            
            # Generate ∃r.concept (e.g., ∃hasChild.Male)
            exist_concept = OWLObjectSomeValuesFrom(prop, concept)
            instances = set(self.kb.individuals(exist_concept))
            if len(instances & self.pos) > 0:
                refinements.append(exist_concept)
            
            # Generate ∃r.¬concept (e.g., ∃hasChild.¬Male)
            neg_concept = OWLObjectComplementOf(concept)
            exist_neg_concept = OWLObjectSomeValuesFrom(prop, neg_concept)
            instances = set(self.kb.individuals(exist_neg_concept))
            if len(instances & self.pos) > 0:
                refinements.append(exist_neg_concept)
        
        return refinements


    def concept_length(self, concept: OWLClassExpression) -> int:
        """Calculate concept length (number of symbols)."""
        if isinstance(concept, OWLClass):
            return 1
        if isinstance(concept, OWLObjectComplementOf):
            return 1 + self.concept_length(concept.get_operand())
        if isinstance(concept, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
            return 1 + sum(self.concept_length(op) for op in concept.operands())
        if isinstance(concept, (OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom)):
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
        if isinstance(concept, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
            return sum(self.count_quantifiers(op) for op in concept.operands())
        if isinstance(concept, OWLObjectComplementOf):
            return self.count_quantifiers(concept.get_operand())
        return 0
    
    def added_role(self, parent: OWLClassExpression, child: OWLClassExpression) -> bool:
        """Check if child was derived from parent by adding a role (∃r or ∀r)."""
        return self.count_quantifiers(child) > self.count_quantifiers(parent)


    def refine(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """
        Main refinement method that dispatches to type-specific refinements.
        Uses PruneCEL-S filtering: only yield child if:
        (1) refinement_score(child) > refinement_score(parent), OR
        (2) child was derived by adding a role (∃r or ∀r)
        """
        # Calculate parent score (cache it)
        parent_key = str(concept)
        if parent_key not in self._score_cache:
            parent_f1, parent_precision, parent_recall, parent_coverage = self.compute_f1(concept, self.kb, self.pos, self.neg)
            parent_instances = set(self.kb.individuals(concept))
            parent_tp = len(parent_instances & self.pos)
            parent_score = self.refinement_score(concept, parent_f1, parent_tp)
            self._score_cache[parent_key] = parent_score
        else:
            parent_score = self._score_cache[parent_key]
        
        # Generate refinements
        if concept == OWLThing:
            results = list(self.refine_top_concept(concept))
        elif isinstance(concept, OWLObjectSomeValuesFrom):
            results = self.refine_existential(concept)
            if results:
                results = list(results)
            else:
                results = []
        elif isinstance(concept, OWLObjectIntersectionOf):
            results = self.refine_intersection(concept)
            if results:
                results = list(results)
            else:
                results = []
        elif isinstance(concept, OWLObjectUnionOf):
            results = self.refine_union(concept)
            if results:
                results = list(results)
            else:
                results = []
        else:
            # For other types (atomic concepts, complements, etc.)
            results = self.refine_any_concept(concept)
            if results:
                results = list(results)
            else:
                results = []
        
        # Filter and yield results based on refinement score
        for ref in results:
            # Compute metrics for child
            f1, precision, recall, coverage = self.compute_f1(ref, self.kb, self.pos, self.neg)
            instances = set(self.kb.individuals(ref))
            tp = len(instances & self.pos)
            
            # Calculate child refinement score
            child_score = self.refinement_score(ref, f1, tp)
            
            # Cache child score for future use
            child_key = str(ref)
            self._score_cache[child_key] = child_score
            
            # PruneCEL-S filter: only yield if score improved OR added a role
            if child_score > parent_score or self.added_role(concept, ref):
                # Check for high-precision fragment
                if precision >= self.precision_threshold and recall < self.recall_threshold:
                    covered = instances & self.pos
                    covered_sig = frozenset(covered)
                    
                    # Check if we haven't already stored a fragment with this exact coverage
                    duplicate = False
                    for existing_frag in self.high_precision_fragments:
                        existing_covered = set(self.kb.individuals(existing_frag)) & self.pos
                        if frozenset(existing_covered) == covered_sig:
                            duplicate = True
                            break
                    
                    if not duplicate:
                        print(f"Fragment: P={precision:.3f}, R={recall:.3f} | {self._renderer.render(ref)[:80]}")
                        self.high_precision_fragments.append(ref)
                        self.covered_positives.update(covered)
                
                # Yield concept that passed the filter
                yield ref
    
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
            
            return f1, precision, recall, len(instances)
        except Exception as e:
            return 0.0, 0.0, 0.0, 0
    
    def get_union_of_fragments(self) -> Optional[OWLClassExpression]:
        """
        Get the union of all collected high-precision fragments.
        
        Returns:
            Union concept if fragments exist, None otherwise
        """
        if not self.high_precision_fragments:
            return None
        
        if len(self.high_precision_fragments) == 1:
            return self.high_precision_fragments[0]
        
        return OWLObjectUnionOf(self.high_precision_fragments)
    
    def get_fragment_stats(self) -> Dict:
        """Get statistics about collected fragments."""
        if not self.high_precision_fragments:
            return {
                'num_fragments': 0,
                'covered_positives': 0,
                'total_positives': len(self.pos) if self.pos else 0,
                'coverage_ratio': 0.0
            }
        
        total_pos = len(self.pos) if self.pos else 0
        return {
            'num_fragments': len(self.high_precision_fragments),
            'covered_positives': len(self.covered_positives),
            'total_positives': total_pos,
            'coverage_ratio': len(self.covered_positives) / total_pos if total_pos > 0 else 0.0
        }


class LengthBasedRefinement(BaseRefinement):
    """ A top-down length based ("no semantic information leveraged) refinement operator in ALC."""

    def __init__(self, knowledge_base: AbstractKnowledgeBase,
                 use_inverse: bool = True,
                 use_data_properties: bool = False,
                 use_card_restrictions: bool = True,
                 use_nominals: bool = True,
                 min_cardinality_restriction: int = 2,
                 max_cardinality_restriction: int = 5):
        """

        """
        super().__init__(knowledge_base)

        self.use_inverse = use_inverse
        self.use_data_properties = use_data_properties
        self.use_card_restrictions = use_card_restrictions
        self.use_nominals = use_nominals
        self.top_refinements: set = None
        self.pos = None
        self.neg = None
        if self.use_card_restrictions:
            assert max_cardinality_restriction > min_cardinality_restriction > 0, \
                f"max_cardinality_restriction:{max_cardinality_restriction} must be greater than min_cardinality_restriction:{min_cardinality_restriction}"
            self.min_cardinality_restriction = min_cardinality_restriction
            self.max_cardinality_restriction = max_cardinality_restriction

    def set_input_examples(self, pos: frozenset, neg: frozenset):
        assert isinstance(pos, frozenset)
        self.pos = {i for i in pos}
        self.neg = {i for i in neg}
        """
        for i in self.pos:
            for s, p, o in self.kb.abox(i):
                if isinstance(o, OWLIndividual):
                    self.property_to_individuals_mapping.setdefault(p, set()).add(o)
        """

    def refine_top(self) -> Iterable:
        """ Refine Top Class Expression

        rho(T)

        1. Most General OWL Named Concepts
        2. Complement of Least General OWL Named Concepts General
        3- Intersection and Union of (1) and (2)
        4. Restriction: SomeValuesFrom and AllValuesFrom
        4.1. For each object property, return SomeValuesFrom AllValuesFrom with (1) / (2) as fillers
        4.2. For each inverse of object property, return SomeValuesFrom AllValuesFrom with (1) / (2) as fillers
        4.3. For each object property, return OWLObjectMinCardinality
        4.4. For each inverse of object property, return OWLObjectMinCardinality





        """
        # (1) Most General OWL Named Concepts
        most_general_concepts = [i for i in self.kb.most_general_classes()]
        yield from most_general_concepts
        # (2) Complement of Least General OWL Named Concepts General
        neg_concepts = [OWLObjectComplementOf(i) for i in self.kb.least_general_named_concepts()]
        yield from neg_concepts
        # (3) Intersection and Union of (1) and (2)
        yield from self.from_iterables(cls=OWLObjectUnionOf,
                                       a_operands=most_general_concepts,
                                       b_operands=most_general_concepts)
        yield from self.from_iterables(cls=OWLObjectUnionOf, a_operands=most_general_concepts, b_operands=neg_concepts)

        # (4) Restriction: SomeValuesFrom and AllValuesFrom
        restrictions = []
        for c in most_general_concepts + [OWLThing, OWLNothing] + neg_concepts:
            owl_obj_property: OWLObjectProperty
            for owl_obj_property in self.kb.get_object_properties():
                # TODO: CD: We need to ignore those expressions where the filler does not match with the range of the property
                # (4.1) For each object property, return SomeValuesFrom AllValuesFrom with (1) / (2) as fillers
                restrictions.append(OWLObjectSomeValuesFrom(property=owl_obj_property, filler=c))
                restrictions.append(OWLObjectAllValuesFrom(property=owl_obj_property, filler=c))
                if self.use_inverse:
                    # (4.2) For each inverse of object property, return SomeValuesFrom AllValuesFrom with (1) / (2) as fillers
                    inverse_owl_obj_property = owl_obj_property.get_inverse_property()
                    restrictions.append(OWLObjectSomeValuesFrom(property=inverse_owl_obj_property, filler=c))
                    restrictions.append(OWLObjectAllValuesFrom(property=inverse_owl_obj_property, filler=c))

                # Move the card limit into existantial restrictions.
                if self.use_card_restrictions:
                    for card in range(self.min_cardinality_restriction, self.max_cardinality_restriction):
                        temp_res = [OWLObjectMinCardinality(cardinality=card,
                                                            property=owl_obj_property,
                                                            filler=c)]
                        if self.use_inverse:
                            temp_res.extend([OWLObjectMinCardinality(cardinality=card,
                                                                     property=inverse_owl_obj_property,
                                                                     filler=c
                                                                     )])
                        restrictions.extend(temp_res)
                    del temp_res
        yield from restrictions

        # (2) OWLDataSomeValuesFrom over double values fillers
        # Two ce for each property returned. Mean value extracted-
        # TODO: Most general_double_data_pro
        if not isinstance(self.kb, KnowledgeBase):  # pragma: no cover
            for i in self.kb.get_double_data_properties():
                doubles = [i.parse_double() for i in self.kb.get_values_of_double_data_property(i)]
                mean_doubles = sum(doubles) / len(doubles)
                yield OWLDataSomeValuesFrom(property=i,
                                            filler=owl_datatype_min_inclusive_restriction(
                                                min_=OWLLiteral(mean_doubles)))
                yield OWLDataSomeValuesFrom(property=i,
                                            filler=owl_datatype_max_inclusive_restriction(
                                                max_=OWLLiteral(mean_doubles)))
        # (3) Boolean Valued OWLDataHasValue: TODO: Most general_boolean_data_pro
        for i in self.kb.get_boolean_data_properties():
            yield OWLDataHasValue(property=i, value=OWLLiteral(True))
            yield OWLDataHasValue(property=i, value=OWLLiteral(False))

    def refine_atomic_concept(self, class_expression: OWLClass) -> Generator[OWLObjectIntersectionOf, None, None]:
        assert isinstance(class_expression, OWLClass), class_expression
        for i in self.top_refinements:
            if i.is_owl_nothing() is False:
                if isinstance(i, OWLClass) and self.kb.are_owl_concept_disjoint(class_expression, i) is False:
                    yield OWLObjectIntersectionOf((class_expression, i))
                else:
                    yield OWLObjectIntersectionOf((class_expression, i))

    def refine_complement_of(self, class_expression: OWLObjectComplementOf) -> Generator[
        OWLObjectComplementOf, None, None]:
        assert isinstance(class_expression, OWLObjectComplementOf)
        # not Father => Not Person given Father subclass of Person
        yield from (OWLObjectComplementOf(i) for i in self.kb.get_direct_parents(class_expression.get_operand()))
        yield OWLObjectIntersectionOf((class_expression, OWLThing))

    def refine_object_some_values_from(self, class_expression: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectSomeValuesFrom)
        # Given \exists r. C
        yield OWLObjectIntersectionOf((class_expression, OWLThing))
        owl_obj_property = class_expression.get_property()
        yield from (OWLObjectSomeValuesFrom(property=owl_obj_property,
                                            filler=C) for C in self.refine(class_expression.get_filler()))
        if self.use_nominals:
            for owl_named_individual in self.kb.individuals(class_expression):
                assert isinstance(owl_named_individual, OWLIndividual)
                yield OWLObjectUnionOf(

                    (class_expression,
                     OWLObjectSomeValuesFrom(property=owl_obj_property, filler=OWLObjectOneOf(owl_named_individual))))


    def refine_object_all_values_from(self, class_expression: OWLObjectAllValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectAllValuesFrom)
        yield OWLObjectIntersectionOf((class_expression, OWLThing))
        owl_obj_property = class_expression.get_property()
        yield from (OWLObjectAllValuesFrom(property=owl_obj_property, filler=C) for C in
                    self.refine(class_expression.get_filler()))
        if self.use_nominals:
            for owl_named_individual in self.kb.individuals(class_expression):
                assert isinstance(owl_named_individual, OWLIndividual)
                yield OWLObjectUnionOf(

                    (class_expression,
                     OWLObjectAllValuesFrom(property=owl_obj_property, filler=OWLObjectOneOf(owl_named_individual))))


    def refine_object_union_of(self, class_expression: OWLObjectUnionOf) -> Iterable[OWLClassExpression]:
        """ Refine OWLObjectUnionof by refining each operands:"""
        assert isinstance(class_expression, OWLObjectUnionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        # Refine each operant
        for i, concept in enumerate(operands):
            for refinement_of_concept in self.refine(concept):
                if refinement_of_concept == class_expression:
                    continue
                yield OWLObjectUnionOf(operands[:i] + [refinement_of_concept] + operands[i + 1:])

        yield OWLObjectIntersectionOf((class_expression, OWLThing))

    def refine_object_intersection_of(self, class_expression: OWLObjectIntersectionOf) -> Iterable[OWLClassExpression]:
        """ Refine OWLObjectIntersectionOf by refining each operands:"""
        assert isinstance(class_expression, OWLObjectIntersectionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        # Refine each operant
        for i, concept in enumerate(operands):
            for refinement_of_concept in self.refine(concept):
                if refinement_of_concept == class_expression:
                    continue
                yield OWLObjectIntersectionOf(operands[:i] + [refinement_of_concept] + operands[i + 1:])

        yield OWLObjectIntersectionOf((class_expression, OWLThing))

    def refine(self, class_expression) -> Iterable[OWLClassExpression]:  # pragma: no cover
        assert isinstance(class_expression, OWLClassExpression)
        # (1) Initialize top refinement if it has not been initialized.
        if self.top_refinements is None:
            self.top_refinements = set()
            for i in self.refine_top():
                self.top_refinements.add(i)
                yield i
        if class_expression.is_owl_thing():
            yield from self.top_refinements
        elif isinstance(class_expression, OWLClass):
            yield from self.refine_atomic_concept(class_expression)
        elif class_expression.is_owl_nothing():
            yield from {class_expression}
        elif isinstance(class_expression, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(class_expression)
        elif isinstance(class_expression, OWLObjectComplementOf):
            yield from self.refine_complement_of(class_expression)
        elif isinstance(class_expression, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectUnionOf):
            yield from self.refine_object_union_of(class_expression)
        elif isinstance(class_expression, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectMaxCardinality):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectExactCardinality):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectMinCardinality):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLDataSomeValuesFrom):
            """unclear how to refine OWLDataHasValue via refining a the property
            We may need to modify the literal little bit right little bit left fashion
            ∃ lumo.xsd:double[≤ -1.6669212962962956] 
            
            ∃ lumo.xsd:double[≥ -1.6669212962962956] 
            """
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLDataHasValue):
            yield from (OWLObjectIntersectionOf((class_expression, i)) for i in self.top_refinements)
        elif isinstance(class_expression, OWLObjectOneOf):
            yield class_expression
        else:
            raise ValueError(f"{type(class_expression)} objects are not yet supported")

    @staticmethod
    def from_iterables(cls, a_operands, b_operands):
        assert (isinstance(a_operands, Generator) is False) and (isinstance(b_operands, Generator) is False)
        seen = set()
        results = set()
        for i in a_operands:
            for j in b_operands:
                #if i == j:
                #    results.add(i)
                if (i, j) in seen:
                    continue
                else:
                    i_and_j = cls((i, j))
                    seen.add((i, j))
                    seen.add((j, i))
                    results.add(i_and_j)
        return results


class ModifiedCELOERefinement(BaseRefinement[OENode]):
    """
     A top down/downward refinement operator in SHIQ(D).
    """
    __slots__ = 'max_child_length', 'use_negation', 'use_all_constructor', 'use_inverse', 'use_card_restrictions', \
        'max_nr_fillers', 'card_limit', 'use_numeric_datatypes', 'use_boolean_datatype', 'dp_splits', \
        'value_splitter', 'use_time_datatypes', 'generator'

    _Node: Final = OENode

    kb: AbstractKnowledgeBase
    value_splitter: Optional[AbstractValueSplitter]
    max_child_length: int
    use_negation: bool
    use_all_constructor: bool
    use_card_restrictions: bool
    use_numeric_datatypes: bool
    use_time_datatypes: bool
    use_boolean_datatype: bool
    card_limit: int

    max_nr_fillers: DefaultDict[OWLObjectPropertyExpression, int]
    dp_splits: Dict[OWLDataPropertyExpression, List[OWLLiteral]]
    generator: ConceptGenerator

    def __init__(self,
                 knowledge_base: AbstractKnowledgeBase,
                 value_splitter: Optional[AbstractValueSplitter] = None,
                 max_child_length: int = 10,
                 use_negation: bool = True,
                 use_all_constructor: bool = True,
                 use_inverse: bool = True,
                 use_card_restrictions: bool = True,
                 use_numeric_datatypes: bool = True,
                 use_time_datatypes: bool = True,
                 use_boolean_datatype: bool = True,
                 card_limit: int = 10):
        self.value_splitter = value_splitter
        self.max_child_length = max_child_length
        self.use_negation = use_negation
        self.use_all_constructor = use_all_constructor
        self.use_inverse = use_inverse
        self.use_card_restrictions = use_card_restrictions
        self.use_numeric_datatypes = use_numeric_datatypes
        self.use_time_datatypes = use_time_datatypes
        self.use_boolean_datatype = use_boolean_datatype
        self.card_limit = card_limit
        self.generator = ConceptGenerator()
        super().__init__(knowledge_base)
        self._setup()

    def _setup(self):
        if self.value_splitter is None:
            self.value_splitter = BinningValueSplitter()

        if self.use_card_restrictions:
            obj_properties = list(self.kb.get_object_properties())
            if self.use_inverse:
                obj_properties.extend(list(map(OWLObjectInverseOf, obj_properties)))

            self.max_nr_fillers = defaultdict(int)
            for prop in obj_properties:
                for ind in self.kb.individuals():
                    num = sum(1 for _ in zip(self.kb.get_object_property_values(ind, prop), range(self.card_limit)))
                    self.max_nr_fillers[prop] = max(self.max_nr_fillers[prop], num)
                    if num == self.card_limit:
                        break

        split_dps = []
        if self.use_numeric_datatypes:
            split_dps.extend(self.kb.get_numeric_data_properties())

        if self.use_time_datatypes:
            split_dps.extend(self.kb.get_time_data_properties())

        if len(split_dps) > 0:
            self.dp_splits = self.value_splitter.compute_splits_properties(self.kb.reasoner, split_dps)

    def _operands_len(self, _Type: Type[OWLNaryBooleanClassExpression],
                      ops: List[OWLClassExpression]) -> int:
        """Calculate the length of a OWL Union or Intersection with operands ops.

        Args:
            _Type: Type of class expression (OWLObjectUnionOf or OWLObjectIntersectionOf)
            ops: list of operands.

        Returns:
            Length of expression.
        """
        length = 0
        if len(ops) == 1:
            length += self.len(ops[0])
        elif ops:
            length += self.len(_Type(ops))
        return length

    def _get_dp_restrictions(self, data_properties: Iterable[OWLDataProperty]) -> List[OWLDataSomeValuesFrom]:
        restrictions = []
        for dp in data_properties:
            splits = self.dp_splits[dp]
            if len(splits) > 0:
                restrictions.append(self.generator.data_existential_restriction(
                    filler=owl_datatype_min_inclusive_restriction(splits[0]), property=dp))
                restrictions.append(self.generator.data_existential_restriction(
                    filler=owl_datatype_max_inclusive_restriction(splits[-1]), property=dp))
        return restrictions

    def _get_current_domain(self, property_: OWLObjectPropertyExpression) -> OWLClassExpression:
        func = self.kb.get_object_property_domains \
            if isinstance(property_, OWLObjectInverseOf) else self.kb.get_object_property_ranges
        return func(property_.get_named_property())

    def refine_atomic_concept(self, ce: OWLClass, max_length: int,
                              current_domain: Optional[OWLClassExpression] = None) -> Iterable[OWLClassExpression]:
        """Refinement operator implementation in CELOE-DL-learner,
        distinguishes the refinement of atomic concepts and start concept(they called Top concept).
        [1] Concept learning, Lehmann et. al

            (1) Generate all subconcepts given C, Denoted by (SH_down(C)),
            (2) Generate {A AND C | A \\in SH_down(C)},
            (2) Generate {A OR C | A \\in SH_down(C)},
            (3) Generate {\\not A | A \\in SH_down(C) AND_logical \\not \\exist B in T : B \\sqsubset A},
            (4) Generate restrictions,
            (5) Intersect and union (1),(2),(3),(4),
            (6) Create negation of all leaf_concepts.

                        (***) The most general relation is not available.

        Args:
            ce: Atomic concept to refine.
            max_length: Refine up to this concept length.
            current_domain: Domain.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLClass)

        if current_domain is None:
            current_domain = OWLThing

        iter_container: List[Iterable[OWLClassExpression]] = []
        # (1) Generate all_sub_concepts. Note that originally CELOE obtains only direct subconcepts
        iter_container.append(self.kb.get_direct_sub_concepts(ce))

        if self.use_negation:
            # TODO probably not correct/complete
            if max_length >= 2 and (self.len(ce) + 1 <= self.max_child_length):
                # (2.2) Create negation of all leaf_concepts
                iter_container.append(self.generator.negation_from_iterables(self.kb.class_hierarchy.leaves(of=ce)))

        if max_length >= 3 and (self.len(ce) + 2 <= self.max_child_length):
            # (2.3) Create ∀.r.T and ∃.r.T where r is the most general relation.
            iter_container.append(self.kb.most_general_existential_restrictions(domain=current_domain))
            if self.use_all_constructor:
                iter_container.append(self.kb.most_general_universal_restrictions(domain=current_domain))
            if self.use_inverse:
                iter_container.append(self.kb.most_general_existential_restrictions_inverse(domain=current_domain))
                if self.use_all_constructor:
                    iter_container.append(self.kb.most_general_universal_restrictions_inverse(domain=current_domain))
            if self.use_numeric_datatypes:
                iter_container.append(self._get_dp_restrictions(
                    self.kb.most_general_numeric_data_properties(domain=current_domain)))
            if self.use_time_datatypes:
                iter_container.append(self._get_dp_restrictions(
                    self.kb.most_general_time_data_properties(domain=current_domain)))
            if self.use_boolean_datatype:
                bool_res = []
                for bool_dp in self.kb.most_general_boolean_data_properties(domain=current_domain):
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(True), property=bool_dp))
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(False),
                                                                              property=bool_dp))
                iter_container.append(bool_res)

        if self.use_card_restrictions and max_length >= 4 and (self.max_child_length >= self.len(ce) + 3):
            card_res = []
            for prop in self.kb.most_general_object_properties(domain=current_domain):
                max_ = self.max_nr_fillers[prop]
                if max_ > 1 or (self.use_negation and max_ > 0):
                    card_res.append(self.generator.max_cardinality_restriction(self.generator.thing, prop, max_ - 1))
            iter_container.append(card_res)

        refs = []
        for i in chain.from_iterable(iter_container):
            yield i
            refs.append(i)

        # Compute all possible combinations of the disjunction and conjunctions.
        mem = set()
        for i in refs:
            # assert i is not None
            i_inds = None
            for j in refs:
                # assert j is not None
                if (i, j) in mem or i == j:
                    continue
                mem.add((j, i))
                mem.add((i, j))
                length = self.len(i) + self.len(j) + 1

                if (max_length >= length) and (self.max_child_length >= length + 1):
                    if not i.is_owl_thing() and not j.is_owl_thing():
                        # TODO: remove individuals_set calls
                        if i_inds is None:
                            i_inds = self.kb.individuals_set(i)
                        j_inds = self.kb.individuals_set(j)
                        if not j_inds.difference(i_inds):
                            # already contained
                            continue
                        else:
                            yield self.generator.union((i, j))

                        if not j_inds.intersection(i_inds):
                            # empty
                            continue
                        else:
                            yield self.generator.intersection((i, j))

    def refine_complement_of(self, ce: OWLObjectComplementOf) -> Iterable[OWLClassExpression]:
        """ Refine owl:complementOf.

        Args:
            ce (OWLObjectComplementOf): owl:complementOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectComplementOf)

        if self.use_negation:
            parents = self.kb.get_direct_parents(ce.get_operand())
            yield from self.generator.negation_from_iterables(parents)
        else:
            yield from {}

    def refine_object_some_values_from(self, ce: OWLObjectSomeValuesFrom, max_length: int) \
            -> Iterable[OWLClassExpression]:
        """ Refine owl:someValuesFrom.

        Args:
            ce (OWLObjectSomeValuesFrom): owl:someValuesFrom class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectSomeValuesFrom)
        assert isinstance(ce.get_filler(), OWLClassExpression)

        # rule 1: EXISTS r.D = > EXISTS r.E
        domain = self._get_current_domain(ce.get_property())
        for i in self.refine(ce.get_filler(), max_length=max_length - 2, current_domain=domain):
            if i is not None:
                yield self.generator.existential_restriction(i, ce.get_property())

        for more_special_op in self.kb.object_property_hierarchy. \
                more_special_roles(ce.get_property().get_named_property()):
            yield self.generator.existential_restriction(ce.get_filler(), more_special_op)

        if self.use_all_constructor:
            yield self.generator.universal_restriction(ce.get_filler(), ce.get_property())

        length = self.len(ce)
        if self.use_card_restrictions and length < max_length and \
                length < self.max_child_length and self.max_nr_fillers[ce.get_property()] > 1:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(), 2)

    def refine_object_all_values_from(self, ce: OWLObjectAllValuesFrom, max_length: int) \
            -> Iterable[OWLObjectAllValuesFrom]:
        """Refine owl:allValuesFrom.

        Args:
            ce (OWLObjectAllValuesFrom): owl:allValuesFrom - class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectAllValuesFrom)

        if self.use_all_constructor:
            # rule 1: Forall r.D = > Forall r.E
            domain = self._get_current_domain(ce.get_property())
            for i in self.refine(ce.get_filler(), max_length=max_length - 2, current_domain=domain):
                if i is not None:
                    yield self.generator.universal_restriction(i, ce.get_property())
            # if not concept.get_filler().is_owl_nothing() and concept.get_filler().isatomic and (len(refs) == 0):
            #    # TODO find a way to include nothing concept
            #    refs.update(self.kb.universal_restriction(i, concept.get_property()))
            for more_special_op in self.kb.object_property_hierarchy. \
                    more_special_roles(ce.get_property().get_named_property()):
                yield self.generator.universal_restriction(ce.get_filler(), more_special_op)
        else:
            yield from {}

    def refine_object_min_card_restriction(self, ce: OWLObjectMinCardinality, max_length: int) \
            -> Iterable[OWLObjectMinCardinality]:
        """Refine owl:minCardinality.

        Args:
            ce (OWLObjectMinCardinality): owl:minCardinality - class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMinCardinality)
        assert ce.get_cardinality() >= 0

        domain = self._get_current_domain(ce.get_property())
        for i in self.refine(ce.get_filler(), max_length=max_length - 3, current_domain=domain):
            if i is not None:
                yield self.generator.min_cardinality_restriction(i, ce.get_property(), ce.get_cardinality())

        if ce.get_cardinality() < self.max_nr_fillers[ce.get_property()]:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() + 1)

    def refine_object_max_card_restriction(self, ce: OWLObjectMaxCardinality, max_length: int) \
            -> Iterable[OWLObjectMaxCardinality]:
        """Refine owl:maxCardinality.

        Args:
            ce (OWLObjectMaxCardinality): owl:maxCardinality - class expression.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMaxCardinality)
        assert ce.get_cardinality() >= 0

        domain = self._get_current_domain(ce.get_property())
        for i in self.refine(ce.get_filler(), max_length=max_length - 3, current_domain=domain):
            if i is not None:
                yield self.generator.max_cardinality_restriction(i, ce.get_property(), ce.get_cardinality())

        if ce.get_cardinality() > 1 or (self.use_negation and ce.get_cardinality() > 0):
            yield self.generator.max_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() - 1)

    def refine_object_union_of(self, ce: OWLObjectUnionOf, max_length: int,
                               current_domain: Optional[OWLClassExpression]) -> Iterable[OWLObjectUnionOf]:
        """Refine owl:unionOf.

        Given a node corresponding a concepts that comprises union operation:
        1) Obtain two concepts A, B,
        2) Refine A and union refinements with B,
        3) Repeat (2) for B.

        Args:
            ce (OWLObjectUnionOf): owl:unionOf - class expression.
            current_domain (OWLClassExpression): Domain.
            max_length (int): Refine up to this concept length.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectUnionOf)

        operands: List[OWLClassExpression] = list(ce.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            concept_length = self.len(concept)

            for ref_concept in self.refine(concept,
                                           max_length=max_length - self.len(ce) + concept_length,
                                           current_domain=current_domain):
                union = self.generator.union(concept_left + [ref_concept] + concept_right)
                if max_length >= self.len(union):
                    yield union

    def refine_object_intersection_of(self, ce: OWLObjectIntersectionOf, max_length: int,
                                      current_domain: Optional[OWLClassExpression]) \
            -> Iterable[OWLObjectIntersectionOf]:
        """Refine owl:intersectionOf.

        Args:
            ce (OWLObjectIntersectionOf): owl:intersectionOf - class expression.
            current_domain (int): Domain.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectIntersectionOf)
        # TODO: Add sanity check method for intersections as in DL-Learner?

        operands: List[OWLClassExpression] = list(ce.operands())
        for i in range(len(operands)):
            concept_left, concept, concept_right = operands[:i], operands[i], operands[i + 1:]
            concept_length = self.len(concept)

            for ref_concept in self.refine(concept,
                                           max_length=max_length - self.len(ce) + concept_length,
                                           current_domain=current_domain):
                intersection = self.generator.intersection(concept_left + [ref_concept] + concept_right)
                if max_length >= self.len(ref_concept):
                    # if other_concept.instances.isdisjoint(ref_concept.instances):
                    #    continue
                    yield intersection

    def refine_data_some_values_from(self, ce: OWLDataSomeValuesFrom) -> Iterable[OWLDataSomeValuesFrom]:
        """Refine owl:someValuesFrom for data properties.

        Args:
            ce (OWLDataSomeValuesFrom): owl:someValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataSomeValuesFrom)
        datarange = ce.get_filler()
        if isinstance(datarange, OWLDatatypeRestriction) and ce.get_property() in self.dp_splits:
            splits = self.dp_splits[ce.get_property()]
            if len(splits) > 0 and len(datarange.get_facet_restrictions()) > 0:
                facet_res = datarange.get_facet_restrictions()[0]
                val = facet_res.get_facet_value()
                idx = splits.index(val)

                if facet_res.get_facet() == OWLFacet.MIN_INCLUSIVE and (next_idx := idx + 1) < len(splits):
                    yield self.generator.data_existential_restriction(
                        owl_datatype_min_inclusive_restriction(splits[next_idx]), ce.get_property())
                elif facet_res.get_facet() == OWLFacet.MAX_INCLUSIVE and (next_idx := idx - 1) >= 0:
                    yield self.generator.data_existential_restriction(
                        owl_datatype_max_inclusive_restriction(splits[next_idx]), ce.get_property())

    def refine_data_has_value(self, ce: OWLDataHasValue) -> Iterable[OWLDataHasValue]:
        """ Refine owl:hasValue.

        Args:
            ce (OWLDataHasValue): owl:hasValue - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataHasValue)

        for more_special_dp in self.kb.data_property_hierarchy.more_special_roles(ce.get_property()):
            yield self.generator.data_has_value_restriction(ce.get_filler(), more_special_dp)

    def refine(self, ce: OWLClassExpression, max_length: int, current_domain: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLClassExpression]:
        """Refine a given concept.

        Args:
            ce: Concept to refine.
            max_length: Refine up to this concept length.
            current_domain: Domain.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLClassExpression)
        if isinstance(ce, OWLClass):
            yield from self.refine_atomic_concept(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectComplementOf):
            yield from self.refine_complement_of(ce)
        elif isinstance(ce, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(ce, max_length)
        elif isinstance(ce, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(ce, max_length)
        elif isinstance(ce, OWLObjectMinCardinality):
            yield from self.refine_object_min_card_restriction(ce, max_length)
        elif isinstance(ce, OWLObjectMaxCardinality):
            yield from self.refine_object_max_card_restriction(ce, max_length)
        elif isinstance(ce, OWLObjectUnionOf):
            yield from self.refine_object_union_of(ce, max_length, current_domain)
        elif isinstance(ce, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(ce, max_length, current_domain)
        elif isinstance(ce, OWLDataSomeValuesFrom):
            yield from self.refine_data_some_values_from(ce)
        elif isinstance(ce, OWLDataHasValue):
            yield from self.refine_data_has_value(ce)
        else:
            raise ValueError(f"{type(ce)} objects are not yet supported")


class ExpressRefinement(ModifiedCELOERefinement):
    """ A top-down refinement operator in ALCHIQ(D)."""

    __slots__ = 'expressivity', 'downsample', 'sample_fillers_count', 'generator'

    expressivity: float
    downsample: bool
    sample_fillers_count: int
    generator: ConceptGenerator

    def __init__(self, knowledge_base: AbstractKnowledgeBase,
                 downsample: bool = True,
                 expressivity: float = 0.8,
                 sample_fillers_count: int = 5,
                 value_splitter: Optional[AbstractValueSplitter] = None,
                 max_child_length: int = 10,
                 use_inverse: bool = True,
                 use_card_restrictions: bool = True,
                 use_numeric_datatypes: bool = True,
                 use_time_datatypes: bool = True,
                 use_boolean_datatype: bool = True,
                 card_limit: int = 10):
        self.downsample = downsample
        self.sample_fillers_count = sample_fillers_count
        self.expressivity = expressivity
        self.generator = ConceptGenerator()
        super().__init__(knowledge_base,
                         value_splitter=value_splitter,
                         max_child_length=max_child_length,
                         use_inverse=use_inverse,
                         use_card_restrictions=use_card_restrictions,
                         use_numeric_datatypes=use_numeric_datatypes,
                         use_time_datatypes=use_time_datatypes,
                         use_boolean_datatype=use_boolean_datatype,
                         card_limit=card_limit)
        self._setup()

    def refine_atomic_concept(self, ce: OWLClass) -> Iterable[OWLClassExpression]:
        """Refine atomic concept.
        Args:
            ce: Atomic concept to refine.

        Returns:
            Iterable of refined concepts.
        """
        if ce.is_owl_nothing():
            yield OWLNothing
        else:
            any_refinement = False
            # Get all subconcepts
            iter_container_sub = list(self.kb.get_all_sub_concepts(ce))
            if len(iter_container_sub) == 0:
                iter_container_sub = [ce]
            iter_container_restrict = []
            # Get negations of all subconcepts
            iter_container_neg = list(self.generator.negation_from_iterables(iter_container_sub))
            # (3) Create ∀.r.C and ∃.r.C where r is the most general relation and C in Fillers
            fillers: Set[OWLClassExpression] = {OWLThing, OWLNothing}
            if len(iter_container_sub) >= self.sample_fillers_count:
                fillers = fillers | set(random.sample(iter_container_sub, k=self.sample_fillers_count)) | \
                          set(random.sample(iter_container_neg, k=self.sample_fillers_count))
            for c in fillers:
                if self.len(c) + 2 <= self.max_child_length:
                    iter_container_restrict.append(
                        set(self.kb.most_general_universal_restrictions(domain=ce, filler=c)))
                    iter_container_restrict.append(
                        set(self.kb.most_general_existential_restrictions(domain=ce, filler=c)))
                    if self.use_inverse:
                        iter_container_restrict.append(
                            set(self.kb.most_general_existential_restrictions_inverse(domain=ce, filler=c)))
                        iter_container_restrict.append(
                            set(self.kb.most_general_universal_restrictions_inverse(domain=ce, filler=c)))

            if self.use_numeric_datatypes:
                iter_container_restrict.append(set(self._get_dp_restrictions(
                    self.kb.most_general_numeric_data_properties(domain=ce))))
            if self.use_time_datatypes:
                iter_container_restrict.append(set(self._get_dp_restrictions(
                    self.kb.most_general_time_data_properties(domain=ce))))
            if self.use_boolean_datatype:
                bool_res = []
                for bool_dp in self.kb.most_general_boolean_data_properties(domain=ce):
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(True), property=bool_dp))
                    bool_res.append(self.generator.data_has_value_restriction(value=OWLLiteral(False),
                                                                              property=bool_dp))
                iter_container_restrict.append(set(bool_res))

            if self.use_card_restrictions and (self.max_child_length >= self.len(ce) + 3):
                card_res = []
                for prop in self.kb.most_general_object_properties(domain=ce):
                    max_ = self.max_nr_fillers[prop]
                    if max_ > 1:
                        card_res.append(self.generator.max_cardinality_restriction(self.generator.thing,
                                                                                   prop, max_ - 1))
                iter_container_restrict.append(set(card_res))

            iter_container_restrict = list(set(chain.from_iterable(iter_container_restrict)))
            container = iter_container_restrict + iter_container_neg + iter_container_sub
            if self.downsample:  # downsampling is necessary if no enough computation resources
                assert self.expressivity < 1, "When downsampling, the expressivity must be less than 1"
                m = int(self.expressivity * len(container))
                container = random.sample(container, k=max(m, 1))
            else:
                self.expressivity = 1.
            if ce.is_owl_thing():  # If this is satisfied then all possible refinements are subconcepts
                if iter_container_neg + iter_container_restrict:
                    any_refinement = True
                    yield from iter_container_neg + iter_container_restrict
            del iter_container_restrict, iter_container_neg
            # Yield all subconcepts
            if iter_container_sub:
                any_refinement = True
                yield from iter_container_sub
            for sub in iter_container_sub:
                for other_ref in container:
                    if sub != other_ref and self.len(sub) + self.len(other_ref) < self.max_child_length:
                        if ce.is_owl_thing() or (other_ref in iter_container_sub):
                            union = self.generator.union([sub, other_ref])
                            yield union
                            any_refinement = True
                        elif other_ref not in iter_container_sub:
                            union = self.generator.union([sub, other_ref])
                            union = self.generator.intersection([ce, union])
                            if self.len(union) <= self.max_child_length:
                                yield union
                                any_refinement = True
                        intersect = self.generator.intersection([sub, other_ref])
                        if self.len(intersect) <= self.max_child_length:
                            yield intersect
                            any_refinement = True
            if not any_refinement:
                yield ce

    def refine_complement_of(self, ce: OWLObjectComplementOf) -> Iterable[OWLClassExpression]:
        """ Refine owl:complementOf.

        Args:
            ce (OWLObjectComplementOf): owl:complementOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectComplementOf)
        any_refinement = False
        parents = self.kb.get_direct_parents(self.generator.negation(ce))
        for ref in self.generator.negation_from_iterables(parents):
            if self.len(ref) <= self.max_child_length:
                any_refinement = True
                yield ref
        if not any_refinement:
            yield ce

    def refine_object_some_values_from(self, ce: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        """ Refine owl:someValuesFrom.

        Args:
            ce (OWLObjectSomeValuesFrom): owl:someValuesFrom class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectSomeValuesFrom)
        assert isinstance(ce.get_filler(), OWLClassExpression)
        any_refinement = False
        for ref in self.refine(ce.get_filler()):
            if 2 + self.len(ref) <= self.max_child_length:
                any_refinement = True
                reft = self.generator.existential_restriction(ref, ce.get_property())
                yield reft
        if self.len(ce) <= self.max_child_length:
            any_refinement = True
            reft = self.generator.universal_restriction(ce.get_filler(), ce.get_property())
            yield reft

        for more_special_op in self.kb.object_property_hierarchy. \
                more_special_roles(ce.get_property().get_named_property()):
            if self.len(ce) <= self.max_child_length:
                yield self.generator.existential_restriction(ce.get_filler(), more_special_op)
                any_refinement = True

        if self.use_card_restrictions and self.len(ce) <= self.max_child_length and \
                self.max_nr_fillers[ce.get_property()] > 1:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(), 2)
            any_refinement = True
        if not any_refinement:
            yield ce

    def refine_object_all_values_from(self, ce: OWLObjectAllValuesFrom) -> Iterable[OWLClassExpression]:
        """Refine owl:allValuesFrom.

        Args:
            ce (OWLObjectAllValuesFrom): owl:allValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectAllValuesFrom)
        assert isinstance(ce.get_filler(), OWLClassExpression)
        any_refinement = False
        for ref in self.refine(ce.get_filler()):
            if 2 + self.len(ref) <= self.max_child_length:
                any_refinement = True
                reft = self.generator.universal_restriction(ref, ce.get_property())
                yield reft
        for more_special_op in self.kb.object_property_hierarchy. \
                more_special_roles(ce.get_property().get_named_property()):
            if 2 + self.len(ce.get_filler()) <= self.max_child_length:
                yield self.generator.universal_restriction(ce.get_filler(), more_special_op)
                any_refinement = True
        if not any_refinement and not ce.get_filler().is_owl_nothing():
            yield ce
        elif not any_refinement and ce.get_filler().is_owl_nothing():
            yield OWLNothing

    def refine_object_min_card_restriction(self, ce: OWLObjectMinCardinality) \
            -> Iterable[OWLObjectMinCardinality]:  # pragma: no cover
        """Refine owl:allValuesFrom.

        Args:
            ce (OWLObjectAllValuesFrom): owl:allValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMinCardinality)
        assert ce.get_cardinality() >= 0

        for ref in self.refine(ce.get_filler()):
            if ref is not None:
                yield self.generator.min_cardinality_restriction(ref, ce.get_property(), ce.get_cardinality())

        if self.use_card_restrictions and ce.get_cardinality() < self.max_nr_fillers[ce.get_property()]:
            yield self.generator.min_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() + 1)

    def refine_object_max_card_restriction(self, ce: OWLObjectMaxCardinality) \
            -> Iterable[OWLObjectMaxCardinality]:  # pragma: no cover
        """Refine owl:maxCardinality.

        Args:
            ce (OWLObjectMaxCardinality): owl:maxCardinality - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectMaxCardinality)
        assert ce.get_cardinality() >= 0

        for ref in self.refine(ce.get_filler()):
            if ref is not None:
                yield self.generator.max_cardinality_restriction(ref, ce.get_property(), ce.get_cardinality())

        if ce.get_cardinality() > 1:
            yield self.generator.max_cardinality_restriction(ce.get_filler(), ce.get_property(),
                                                             ce.get_cardinality() - 1)

    def refine_object_union_of(self, ce: OWLObjectUnionOf) -> Iterable[OWLClassExpression]:
        """Refine owl:unionOf.

        Args:
            ce (OWLObjectUnionOf): owl:unionOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectUnionOf)
        any_refinement = False
        for op in ce.operands():
            if self.len(op) <= self.max_child_length:
                yield op
                any_refinement = True
        operands = list(ce.operands())
        for i in range(len(operands)):
            ce_left, ce_, ce_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectUnionOf, ce_left + ce_right)
            for ref_ce in self.refine(ce_):
                if self.max_child_length >= other_length + self.len(ref_ce):
                    yield self.generator.union(ce_left + [ref_ce] + ce_right)
                    any_refinement = True
        if not any_refinement:
            yield ce

    def refine_object_intersection_of(self, ce: OWLObjectIntersectionOf) -> Iterable[OWLClassExpression]:
        """Refine owl:intersectionOf.

        Args:
            ce (OWLObjectIntersectionOf): owl:intersectionOf - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLObjectIntersectionOf)
        any_refinement = False
        operands = list(ce.operands())
        for i in range(len(operands)):
            ce_left, ce, ce_right = operands[:i], operands[i], operands[i + 1:]
            other_length = self._operands_len(OWLObjectIntersectionOf, ce_left + ce_right)
            for ref_ce in self.refine(ce):
                if self.max_child_length >= other_length + self.len(ref_ce):
                    yield self.generator.intersection(ce_left + [ref_ce] + ce_right)
                    any_refinement = True
        if not any_refinement:
            yield ce

    def refine_data_some_values_from(self, ce: OWLDataSomeValuesFrom) -> Iterable[OWLDataSomeValuesFrom]:  # pragma: no cover
        """Refine owl:someValuesFrom for data properties.

        Args:
            ce (OWLDataSomeValuesFrom): owl:someValuesFrom - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataSomeValuesFrom)
        any_refinement = False
        datarange = ce.get_filler()
        if isinstance(datarange, OWLDatatypeRestriction) and ce.get_property() in self.dp_splits:
            splits = self.dp_splits[ce.get_property()]
            if len(splits) > 0 and len(datarange.get_facet_restrictions()) > 0:
                facet_res = datarange.get_facet_restrictions()[0]
                val = facet_res.get_facet_value()
                idx = splits.index(val)

                if facet_res.get_facet() == OWLFacet.MIN_INCLUSIVE and (next_idx := idx + 1) < len(splits):
                    yield self.generator.data_existential_restriction(
                        owl_datatype_min_inclusive_restriction(splits[next_idx]), ce.get_property())
                    any_refinement = True
                elif facet_res.get_facet() == OWLFacet.MAX_INCLUSIVE and (next_idx := idx - 1) >= 0:
                    yield self.generator.data_existential_restriction(
                        owl_datatype_max_inclusive_restriction(splits[next_idx]), ce.get_property())
                    any_refinement = True
        if not any_refinement:
            yield ce

    def refine_data_has_value(self, ce: OWLDataHasValue) -> Iterable[OWLDataHasValue]:  # pragma: no cover
        """ Refine owl:hasValue.

        Args:
            ce (OWLDataHasValue): owl:hasValue - class expression.

        Returns:
            Iterable of refined concepts.
        """
        assert isinstance(ce, OWLDataHasValue)
        any_refinement = False
        for more_special_dp in self.kb.data_property_hierarchy.more_special_roles(ce.get_property()):
            yield self.generator.data_has_value_restriction(ce.get_filler(), more_special_dp)
            any_refinement = True

        if not any_refinement:
            yield ce

    def refine(self, ce, **kwargs) -> Iterable[OWLClassExpression]:  # pragma: no cover
        """Refine a given concept.

        Args:
            ce: Concept to refine

        Returns:
            Iterable of refined concepts.
        """
        # we ignore additional arguments like "max_length" or "current_domain" that might be supplied by the learning
        # algorithm by using **kwargs
        assert isinstance(ce, OWLClassExpression)
        if self.len(ce) == 1:
            yield from self.refine_atomic_concept(ce)
        elif isinstance(ce, OWLObjectComplementOf):
            yield from self.refine_complement_of(ce)
        elif isinstance(ce, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(ce)
        elif isinstance(ce, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(ce)
        elif isinstance(ce, OWLObjectMinCardinality):
            yield from self.refine_object_min_card_restriction(ce)
        elif isinstance(ce, OWLObjectMaxCardinality):
            yield from self.refine_object_max_card_restriction(ce)
        elif isinstance(ce, OWLObjectUnionOf):
            yield from self.refine_object_union_of(ce)
        elif isinstance(ce, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(ce)
        elif isinstance(ce, OWLDataSomeValuesFrom):
            yield from self.refine_data_some_values_from(ce)
        elif isinstance(ce, OWLDataHasValue):
            yield from self.refine_data_has_value(ce)
        else:
            raise ValueError(f"{type(ce)} objects are not yet supported")


class NERORefinement(BaseRefinement):
    """
    A custom refinement operator for NERO that generates candidate class expressions.
    This is a faithful implementation of SimpleRefinement from the original NERO.
    """
    def __init__(self, knowledge_base: KnowledgeBase, fast_init=False):
        super().__init__(knowledge_base)
        self.renderer = DLSyntaxObjectRenderer()
        self.N_I = {i.str for i in self.kb.individuals(OWLThing)}  # All individuals
        self.expressions = {}  # str -> ClassExpression
        self.str_nc_star = None
        self.dict_sh_direct_down = {}  # subsumption hierarchy down
        self.dict_sh_direct_up = {}  # subsumption hierarchy up
        self.fast_init = fast_init
        self.length_to_expression_str = {}
        self.str_ae_to_neg_ae = {}  # atomic expression name -> negated expression
        self.top_refinements = {}  # length -> set of expressions

        self._prepare()

    def _prepare(self):
        """Materialize KB to subsumption hierarchy and create all basic expressions."""
        # Get all named classes including Top and Bottom
        nc_top_bot = list(self.kb.class_hierarchy.sub_classes(OWLThing, direct=False))

        # Initialize length dictionaries
        self.length_to_expression_str.setdefault(1, set())
        self.length_to_expression_str.setdefault(2, set())
        self.length_to_expression_str.setdefault(3, set())

        # (1) Create atomic expressions for all named classes + Top + Bottom
        for atomic_owl_class in [OWLThing, OWLNothing] + nc_top_bot:
            # Create atomic expression
            target = AtomicExpression(
                name=self.renderer.render(atomic_owl_class),
                str_individuals={i.str for i in self.kb.individuals(atomic_owl_class)},
                expression_chain=tuple(
                    self.renderer.render(x) for x in self.kb.get_direct_parents(atomic_owl_class)
                )
            )
            self._store(target, length=1)

            # (1.3) Create universal restrictions: ∀r.C
            for mgur in self.kb.most_general_universal_restrictions(domain=OWLThing, filler=atomic_owl_class):
                filler = self.expressions[self.renderer.render(mgur.get_filler())]
                role = Role(name=self.renderer.render(mgur.get_property()))
                target = UniversalQuantifierExpression(
                    name=self.renderer.render(mgur),
                    role=role,
                    filler=filler,
                    str_individuals={i.str for i in self.kb.individuals(mgur)},
                    expression_chain=(self.renderer.render(OWLThing),)
                )
                self._store(target, length=3)

            # (1.4) Create existential restrictions: ∃r.C
            for mger in self.kb.most_general_existential_restrictions(domain=OWLThing, filler=atomic_owl_class):
                filler = self.expressions[self.renderer.render(mger.get_filler())]
                role = Role(name=self.renderer.render(mger.get_property()))
                target = ExistentialQuantifierExpression(
                    name=self.renderer.render(mger),
                    role=role,
                    filler=filler,
                    str_individuals={i.str for i in self.kb.individuals(mger)},
                    expression_chain=(self.renderer.render(OWLThing),)
                )
                self._store(target, length=3)

        if not self.fast_init:
            # (2) Construct subsumption hierarchy (sh_down and sh_up)
            for k, v in self.expressions.items():
                self.dict_sh_direct_down.setdefault(k, set())
                if len(v.expression_chain) == 0:
                    v.expression_chain = (self.renderer.render(OWLThing),)
                assert len(v.expression_chain) > 0

                for x in v.expression_chain:
                    self.dict_sh_direct_down.setdefault(x, set()).add(k)
                    self.dict_sh_direct_up.setdefault(k, set()).add(x)

            # Remove self-loop: sh_down(T) => T
            if '⊤' in self.dict_sh_direct_down:
                self.dict_sh_direct_down['⊤'].discard('⊤')

            # Fill top refinements: direct children of Top
            if '⊤' in self.dict_sh_direct_down:
                for i in self.dict_sh_direct_down['⊤']:
                    ref = self.expressions[i]
                    self.top_refinements.setdefault(ref.length, set()).add(ref)

            # Create negations of leaf nodes (concepts with Bottom as child)
            for i in self._expression_given_length(1):
                if i.name in ['⊤', '⊥']:
                    continue
                if '⊥' in self.dict_sh_direct_down.get(i.name, set()):
                    neg_i = ComplementOfAtomicExpression(
                        name='¬' + i.name,
                        atomic_expression=i,
                        str_individuals=self.N_I.difference(i.str_individuals),
                        expression_chain=(self.renderer.render(OWLThing),)
                    )
                    self._store(neg_i, length=2)
                    self.str_ae_to_neg_ae[i.name] = neg_i
                    self.top_refinements.setdefault(neg_i.length, set()).add(neg_i)

            self.str_nc_star = set(self.expressions.keys())
        else:
            # Fast init: create all negations
            for i in self._expression_given_length(1):
                if i.name in ['⊤', '⊥']:
                    continue
                neg_i = ComplementOfAtomicExpression(
                    name='¬' + i.name,
                    atomic_expression=i,
                    str_individuals=self.N_I.difference(i.str_individuals),
                    expression_chain=(self.renderer.render(OWLThing),)
                )
                self._store(neg_i, length=2)

    def _store(self, target: ClassExpression, length: int) -> None:
        """Store an expression indexed by its length."""
        self.expressions[target.name] = target
        self.length_to_expression_str.setdefault(length, set()).add(target)

    def _expression_given_length(self, length: int) -> List[ClassExpression]:
        """Get all expressions of a given length."""
        if length in self.length_to_expression_str:
            return list(self.length_to_expression_str[length])
        else:
            return []

    def _negate_atomic_class(self, i: AtomicExpression) -> ComplementOfAtomicExpression:
        """Negate an atomic class expression."""
        neg_i = ComplementOfAtomicExpression(
            name='¬' + i.name,
            atomic_expression=i,
            str_individuals=self.N_I.difference(i.str_individuals),
            expression_chain=i.expression_chain
        )
        self._store(neg_i, length=2)
        self.str_ae_to_neg_ae[i.name] = neg_i
        return neg_i

    def _sh_down(self, atomic_class_expression: AtomicExpression) -> List[AtomicExpression]:
        """Return direct sub-concepts in subsumption hierarchy."""
        res = []
        for str_ in self.dict_sh_direct_down.get(atomic_class_expression.name, []):
            res.append(self.expressions[str_])
        return res

    def _sh_up(self, atomic_class_expression: AtomicExpression) -> List[AtomicExpression]:
        """Return direct super-concepts in subsumption hierarchy."""
        res = []
        for str_ in self.dict_sh_direct_up.get(atomic_class_expression.name, []):
            res.append(self.expressions[str_])
        return res

    def _intersect_with_top(self, res: List[ClassExpression]):
        """Intersect given expressions with refinements of Top."""
        for k_length_int, refinements_top in self.top_refinements.items():
            for ref_top in refinements_top:
                for j in res:
                    if (len(j.str_individuals) > 0 and
                        len(ref_top.str_individuals.intersection(j.str_individuals)) > 0):
                        ref_top_and_j = ref_top * j  # Use __mul__ operator
                        if ref_top_and_j.length == 3:
                            self.top_refinements.setdefault(ref_top_and_j.length, set()).add(ref_top_and_j)
                        yield ref_top_and_j

    def _refine_atomic_concept(self, atomic_class_expression: AtomicExpression):
        """Refine atomic concept: {A' | A' ∈ sh_down(A)} ∪ {A ⊓ D | D ∈ ρ(⊤)}"""
        yield from self._sh_down(atomic_class_expression)
        yield from self._intersect_with_top([atomic_class_expression])

    def _refine_complement_of(self, neg_atomic_expression: ComplementOfAtomicExpression):
        """Refine negation: {¬A' | A' ∈ sh_↑(A)} ∪ {¬A ⊓ D | D ∈ ρ(⊤)}"""
        atomic_class_expression = neg_atomic_expression.atomic_expression
        res = []
        for ae in self._sh_up(atomic_class_expression):
            if ae.name in self.str_ae_to_neg_ae:
                res.append(self.str_ae_to_neg_ae[ae.name])
            else:
                # Create negation on the fly if not exists
                neg_ae = self._negate_atomic_class(ae)
                res.append(neg_ae)
        yield from res
        yield from self._intersect_with_top(res)

    def _refine_universal_quantifier(self, uqe: UniversalQuantifierExpression):
        """Refine universal quantifier: {∀r.D ⊓ E | E ∈ ρ(⊤)}"""
        yield from self._intersect_with_top([uqe])

    def _refine_existential_quantifier(self, eqe: ExistentialQuantifierExpression):
        """Refine existential quantifier: {∃r.D ⊓ E | E ∈ ρ(⊤)}"""
        yield from self._intersect_with_top([eqe])

    def _refine_union_or_intersection(self, x, operator):
        """Refine union or intersection by refining operands."""
        a, b = x.concepts

        # Refine left operand
        for refined_a in self.refine(a):
            if (len(refined_a.str_individuals) > 0 and
                len(refined_a.str_individuals.intersection(b.str_individuals)) > 0):
                yield operator(refined_a, b)

        # Refine right operand
        for refined_b in self.refine(b):
            if (len(refined_b.str_individuals) > 0 and
                len(refined_b.str_individuals.intersection(a.str_individuals)) > 0):
                yield operator(a, refined_b)

    def _refine_union_expression(self, ue: UnionClassExpression):
        """Refine union expression."""
        yield from self._refine_union_or_intersection(ue, operator=lambda x, y: x + y)

    def _refine_intersection_expression(self, ie: IntersectionClassExpression):
        """Refine intersection expression."""
        yield from self._refine_union_or_intersection(ie, operator=lambda x, y: x * y)

    def refine(self, class_expression: ClassExpression) -> List[ClassExpression]:
        """
        Generate refinements for a given class expression.
        This implements the full refinement logic from the original NERO.
        """
        refinements = []

        if isinstance(class_expression, AtomicExpression):
            refinements = list(self._refine_atomic_concept(class_expression))
        elif isinstance(class_expression, ComplementOfAtomicExpression):
            refinements = list(self._refine_complement_of(class_expression))
        elif isinstance(class_expression, UniversalQuantifierExpression):
            refinements = list(self._refine_universal_quantifier(class_expression))
        elif isinstance(class_expression, ExistentialQuantifierExpression):
            refinements = list(self._refine_existential_quantifier(class_expression))
        elif isinstance(class_expression, UnionClassExpression):
            refinements = list(self._refine_union_expression(class_expression))
        elif isinstance(class_expression, IntersectionClassExpression):
            refinements = list(self._refine_intersection_expression(class_expression))
        else:
            # Unknown type, return empty list
            refinements = []

        return refinements

