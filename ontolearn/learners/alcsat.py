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

from typing import Optional, Iterable, Set

from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.abstracts import AbstractOWLReasoner

from ontolearn.learners.base import BaseConceptLearner
from ontolearn.abstracts import AbstractScorer, AbstractKnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard


from ontolearn.learners.spell.fitting_alc import FittingALC, STreeNode, NEG, AND, OR, EX, ALL
from ontolearn.learners.spell.structures import Structure


class ALCSAT():
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
    
    __slots__ = ('kb','reasoner', 'max_concept_size', 'start_concept_size', 'operators', 'tree_templates',
                 'type_encoding', 'timeout', '_best_hypothesis', '_structure', '_ind_to_owl',
                 '_owl_to_ind', '_learning_problem', '_best_hypothesis_accuracy', 'start_time')
    
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
        
        self.max_concept_size = max_concept_size
        self.start_concept_size = start_concept_size
        self.operators = operators if operators is not None else {NEG, AND, OR, EX, ALL}
        self.tree_templates = tree_templates
        self.type_encoding = type_encoding
        self.timeout = max_runtime
        self.kb = knowledge_base
        if reasoner == None:
            self.reasoner = self.kb.reasoner
        
        self._best_hypothesis = None
        self._best_hypothesis_accuracy = None
        self._structure = None
        self._ind_to_owl = {}
        self._owl_to_ind = {}
        self._learning_problem = None
        self.start_time = None
    
    def clean(self):
        """Clear all states of the concept learner."""
        self._best_hypothesis = None
        self._best_hypothesis_accuracy = None
        self._structure = None
        self._ind_to_owl = {}
        self._owl_to_ind = {}
        self._learning_problem = None
    
    def _convert_kb_to_structure(self) -> Structure:
        """
        Convert knowledge base to spell Structure format.

            
        Returns:
            Structure object for use with FittingALC.
        """
        # Get all relevant individuals
        all_individuals = set(self.kb.individuals())
        max_ind = len(all_individuals)

        # Create mappings
        self._owl_to_ind = {ind: idx for idx, ind in enumerate(all_individuals)}
        self._ind_to_owl = {idx: ind for ind, idx in self._owl_to_ind.items()}

        # Extract concept names and their extensions
        cn_ext = {}
        for cls in self.kb.ontology.classes_in_signature():
            if cls.is_owl_thing() or cls.is_owl_nothing():
                continue
            cls_name = cls.iri.get_remainder()
            if cls_name:
                instances = set(self.reasoner.instances(cls))
                cn_ext[cls_name] = {self._owl_to_ind[ind] for ind in instances 
                                    if ind in self._owl_to_ind}
        
        # Extract role names and their extensions
        rn_ext = {i: set() for i in range(max_ind)}
        for prop in self.kb.ontology.object_properties_in_signature():
            prop_name = prop.iri.get_remainder()
            if prop_name:
                for ind in all_individuals:
                    if ind in self._owl_to_ind:
                        ind_idx = self._owl_to_ind[ind]
                        # Get property values for this individual
                        for value in self.reasoner.object_property_values(ind, prop):
                            if value in self._owl_to_ind:
                                value_idx = self._owl_to_ind[value]
                                rn_ext[ind_idx].add((value_idx, prop_name))

        # Create individual name mapping
        indmap = {ind.iri.as_str(): idx for ind, idx in self._owl_to_ind.items()}
        
        # Create namespace mapping
        nsmap = {}
        
        return Structure(
            max_ind=max_ind,
            cn_ext=cn_ext,
            rn_ext=rn_ext,
            indmap=indmap,
            nsmap=nsmap
        )
    
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
    
    def fit(self, lp:PosNegLPStandard):
        """
        Find ALC concept expressions that explain positive and negative examples.
        
        Args:
            *args: Either a PosNegLPStandard learning problem, or positive examples.
            **kwargs: May contain 'learning_problem', 'pos', 'neg', etc.
            
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
        
        # Convert knowledge base to Structure format
        self._structure = self._convert_kb_to_structure()
        
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
            timeout=self.timeout
        )
        
        # Convert solution to OWL expression
        if best_sol is not None:
            owl_expr = self._tree_to_owl_expression(best_sol)
            self._best_hypothesis = owl_expr
            self._best_hypothesis_accuracy = best_acc
    
    def best_hypothesis(self) -> Iterable[OWLClassExpression]:
        """
        Get the best found hypotheses.
        
        Args:
            n: Maximum number of results to return.
            
        Returns:
            Iterable of OWL class expressions.
        """
        if not self._best_hypothesis:
            return OWLThing
        
        # Return top n hypotheses
        return self._best_hypothesis

    def best_hypothesis_accuracy(self) -> Optional[float]:
        """
        Get the accuracy of the best found hypothesis.

        Returns:
            Accuracy as float, or None if no hypothesis found.
        """
        return self._best_hypothesis_accuracy

