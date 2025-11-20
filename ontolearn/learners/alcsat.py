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

"""ALCSAT Learner - SAT-based ALC concept learning."""

from typing import Optional, Set

from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.abstracts import AbstractOWLReasoner

from ontolearn.abstracts import AbstractKnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.learners.sat_base import SATBaseLearner

from ontolearn.learners.spell_kit.fitting_alc import FittingALC, STreeNode, NEG, AND, OR, EX, ALL
from ontolearn.learners.spell_kit.structures import Structure


class ALCSAT(SATBaseLearner):
    """
    ALCSAT: SAT-based ALC concept learner.
    
    This learner uses SAT solvers to find ALC concept expressions that fit positive and negative examples.
    It encodes the concept learning problem as a SAT problem and uses a Glucose SAT solver to find solutions.
    
    The algorithm incrementally searches for concepts of increasing size (tree depth k) that maximize
    the accuracy on the given examples.
    
    Attributes:
        kb (AbstractKnowledgeBase): The knowledge base that the concept learner is using.
        max_concept_size (int): Maximum size (depth) of concepts to search for.
        start_concept_size (int): Starting size for incremental search.
        operators (Set): Set of ALC operators to use (NEG, AND, OR, EX, ALL).
        tree_templates (bool): Whether to use tree templates for symmetry breaking.
        type_encoding (bool): Whether to use type encoding optimization.
        timeout (float): Timeout in seconds for the SAT solver (-1 for no timeout).
        _best_hypothesis (OWLClassExpression): Best found hypothesis.
        _best_hypothesis_accuracy (float): Accuracy of the best hypothesis.
        _structure (Structure): Internal structure representation of the knowledge base.
        _ind_to_owl (dict): Mapping from internal individual indices to OWL individuals.
        _owl_to_ind (dict): Mapping from OWL individuals to internal indices.
    """
    
    __slots__ = ('max_concept_size', 'start_concept_size', 'operators', 'tree_templates', 'type_encoding')

    name = 'alcsat'
    
    def __init__(self,
                 knowledge_base: AbstractKnowledgeBase,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 max_runtime: Optional[int] = 60,
                 max_concept_size: int = 10,
                 start_concept_size: int = 1,
                 operators: Optional[Set] = None,
                 tree_templates: bool = True,
                 type_encoding: bool = True):
        """
        Initialize ALCSAT learner.
        
        Args:
            knowledge_base: The knowledge base to use for learning.
            reasoner: Optional reasoner (if None, uses the KB's reasoner).
            max_runtime: Maximum allowed runtime in seconds.
            max_concept_size: Maximum concept tree depth to search. Defaults to 10.
            start_concept_size: Starting concept size for incremental search. Defaults to 1.
            operators: Set of ALC operators to use. Defaults to {NEG, AND, OR, EX, ALL}.
            tree_templates: Whether to use tree template optimization. Defaults to True.
            type_encoding: Whether to use type encoding optimization. Defaults to True.
        """
        super().__init__(knowledge_base, reasoner, max_runtime)

        self.max_concept_size = max_concept_size
        self.start_concept_size = start_concept_size
        self.operators = operators if operators is not None else {NEG, AND, OR, EX, ALL}
        self.tree_templates = tree_templates
        self.type_encoding = type_encoding

    def _tree_to_owl_expression(self, tree: STreeNode) -> OWLClassExpression:
        """
        Convert a syntax tree from FittingALC to an OWL class expression.
        
        Args:
            tree: STreeNode from FittingALC.
            
        Returns:
            OWL class expression.
        """
        from owlapy.class_expression import (
            OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectComplementOf,
            OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLClass, OWLThing, OWLNothing
        )
        from owlapy.owl_property import OWLObjectProperty
        from owlapy.iri import IRI
        
        node_label = tree.node[1]
        
        # Get base IRI for the ontology
        onto = self.kb.ontology
        if hasattr(onto, 'get_ontology_iri'):
            base_iri = onto.get_ontology_iri().as_str()
        else:
            # Fallback: try to extract from any class IRI
            try:
                some_class = next(iter(onto.classes_in_signature()))
                base_iri = some_class.iri.as_str().rsplit('#', 1)[0]
            except StopIteration:
                base_iri = "http://example.org/ontology"

        # Handle atomic concepts
        if node_label == "TOP":
            return OWLThing
        elif node_label == "BOT":
            return OWLNothing
        elif node_label.startswith("ex."):
            # Existential restriction
            role_name = node_label[3:]
            prop = OWLObjectProperty(IRI.create(base_iri + "#" + role_name))
            if tree.children:
                filler = self._tree_to_owl_expression(tree.children[0])
                return OWLObjectSomeValuesFrom(prop, filler)
            return OWLThing
        elif node_label.startswith("all."):
            # Universal restriction
            role_name = node_label[4:]
            prop = OWLObjectProperty(IRI.create(base_iri + "#" + role_name))
            if tree.children:
                filler = self._tree_to_owl_expression(tree.children[0])
                return OWLObjectAllValuesFrom(prop, filler)
            return OWLThing
        elif node_label == "NEG":
            # Negation
            if tree.children:
                child_expr = self._tree_to_owl_expression(tree.children[0])
                return OWLObjectComplementOf(child_expr)
            return OWLThing
        elif node_label == "AND" or node_label == "⊓":
            # Conjunction
            if len(tree.children) >= 2:
                operands = [self._tree_to_owl_expression(child) for child in tree.children]
                return OWLObjectIntersectionOf(operands)
            return OWLThing
        elif node_label == "OR" or node_label == "⊔":
            # Disjunction
            if len(tree.children) >= 2:
                operands = [self._tree_to_owl_expression(child) for child in tree.children]
                return OWLObjectUnionOf(operands)
            return OWLThing
        else:
            # Assume it's a concept name
            concept_iri = IRI.create(base_iri + "#" + node_label)
            return OWLClass(concept_iri)
    
    def fit(self, lp: PosNegLPStandard):
        """
        Find ALC concept expressions that explain positive and negative examples.
        
        Args:
            lp: Learning problem with positive and negative examples.

        Returns:
            self
        """
        import time
        
        self.clean()
        self.start_time = time.time()
        
        # Construct learning problem
        assert isinstance(lp, PosNegLPStandard)
        self._learning_problem = lp
        
        pos = set(self._learning_problem.pos)
        neg = set(self._learning_problem.neg)

        # Convert positive and negative examples to indices
        P = [self._owl_to_ind[ind] for ind in pos]
        N = [self._owl_to_ind[ind] for ind in neg]

        # Create FittingALC instance
        fitting = FittingALC(
            A=self._structure,
            k=self.max_concept_size,
            P=P,
            N=N,
            op=self.operators,
            tree_templates=self.tree_templates,
            type_encoding=self.type_encoding
        )
        
        # Run incremental search
        best_acc, final_k, best_sol = fitting.solve_incr_approx(
            max_k=self.max_concept_size,
            start_k=self.start_concept_size,
            min_n=len(P) + len(N),
            timeout=self.max_runtime if self.max_runtime else -1
        )
        
        # Convert solution to OWL expression
        if best_sol is not None:
            owl_expr = self._tree_to_owl_expression(best_sol)
            self._best_hypothesis = owl_expr
            self._best_hypothesis_accuracy = best_acc

        return self
