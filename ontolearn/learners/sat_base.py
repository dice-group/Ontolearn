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

"""Base class for SAT-based concept learners."""

from typing import Optional
from abc import abstractmethod

from owlapy.class_expression import OWLClassExpression, OWLThing, OWLClass
from owlapy.class_expression import OWLObjectIntersectionOf, OWLObjectSomeValuesFrom
from owlapy.owl_property import OWLObjectProperty
from owlapy.iri import IRI
from owlapy.abstracts import AbstractOWLReasoner

from ontolearn.abstracts import AbstractKnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.learners.spell_kit.structures import Structure


class SATBaseLearner:
    """
    Base class for SAT-based concept learners that use the SPELL framework.
    
    This class provides common functionality for converting between OWL ontologies
    and the Structure format used by SPELL, as well as converting results back to
    OWL class expressions.
    """
    
    __slots__ = ('kb', 'reasoner', 'max_runtime', '_best_hypothesis', 
                 '_best_hypothesis_accuracy','cache_structure', '_structure', '_ind_to_owl',
                 '_owl_to_ind', '_learning_problem', 'start_time')
    
    def __init__(self,
                 knowledge_base: AbstractKnowledgeBase,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 max_runtime: Optional[int] = 60,
                 cache_structure: bool = True):
        """
        Initialize SAT-based learner.
        
        Args:
            knowledge_base: The knowledge base to use for learning.
            reasoner: Optional reasoner (if None, uses the KB's reasoner).
            max_runtime: Maximum allowed runtime in seconds.
            cache_structure: Whether to cache the structure representation. Set to False if you plan to modify the
                ontology between multiple calls to fit().
        """
        self.kb = knowledge_base
        self.reasoner = reasoner if reasoner is not None else self.kb.reasoner
        self.max_runtime = max_runtime
        self.cache_structure = cache_structure
        self.start_time = None
        
        self._best_hypothesis = None
        self._best_hypothesis_accuracy = None
        self._ind_to_owl = {}
        self._owl_to_ind = {}
        self._learning_problem = None
        self._structure = self._convert_kb_to_structure()
    
    def clean(self):
        """Clear all states of the concept learner."""
        self._best_hypothesis = None
        self._best_hypothesis_accuracy = None
        self._learning_problem = None
        if not self.cache_structure:
            # if ontology is modified, the structure will also reflect the changes on next the fit
            self._ind_to_owl = {}
            self._owl_to_ind = {}
            self._structure = self._convert_kb_to_structure()

    def _convert_kb_to_structure(self) -> Structure:
        """
        Convert knowledge base to SPELL Structure format.
        
        Returns:
            Structure object for use with SPELL solvers.
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
                        # Get property values for this individual.
                        # Guard against malformed ontologies (e.g. cyclic subclass
                        # definitions) where Owlready2 may return complex class
                        # expressions (Or, And, …) instead of named individuals,
                        # causing AttributeError: 'Or' object has no attribute 'iri'.
                        try:
                            values = list(self.reasoner.object_property_values(ind, prop))
                        except AttributeError:
                            continue
                        for value in values:
                            try:
                                if value in self._owl_to_ind:
                                    value_idx = self._owl_to_ind[value]
                                    rn_ext[ind_idx].add((value_idx, prop_name))
                            except (AttributeError, TypeError):
                                pass

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
    
    def _structure_to_owl_expression(self, query: Structure) -> OWLClassExpression:
        """
        Convert a Structure query result to an OWL class expression.
        
        Args:
            query: Structure object representing a query/concept.
            
        Returns:
            OWL class expression.
        """
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
        
        # Build concept expression from query structure
        # Start with the root node (index 0)
        return self._build_expression_from_query(query, 0, base_iri)
    
    def _build_expression_from_query(self, query: Structure, node_idx: int, base_iri: str) -> OWLClassExpression:
        """
        Recursively build an OWL expression from a query structure.
        
        Args:
            query: Structure query object.
            node_idx: Current node index in the query.
            base_iri: Base IRI for the ontology.
            
        Returns:
            OWL class expression for this node.
        """
        # Get concept assertions for this node
        concepts = []
        for cn_name, cn_inds in query.cn_ext.items():
            if node_idx in cn_inds:
                concept_iri = IRI.create(base_iri + "#" + cn_name)
                concepts.append(OWLClass(concept_iri))
        
        # Get role assertions for this node
        role_restrictions = []
        if node_idx in query.rn_ext:
            for target_idx, role_name in query.rn_ext[node_idx]:
                prop = OWLObjectProperty(IRI.create(base_iri + "#" + role_name))
                filler = self._build_expression_from_query(query, target_idx, base_iri)
                role_restrictions.append(OWLObjectSomeValuesFrom(prop, filler))
        
        # Combine all parts
        all_parts = concepts + role_restrictions
        
        if len(all_parts) == 0:
            return OWLThing
        elif len(all_parts) == 1:
            return all_parts[0]
        else:
            return OWLObjectIntersectionOf(all_parts)
    
    @abstractmethod
    def fit(self, lp: PosNegLPStandard):
        """
        Find concept expressions that explain positive and negative examples.
        
        Args:
            lp: Learning problem with positive and negative examples.
            
        Returns:
            self
        """
        pass
    
    def best_hypothesis(self) -> OWLClassExpression:
        """
        Get the best found hypothesis.
        
        Returns:
            OWL class expression, or OWLThing if no hypothesis found.
        """
        if not self._best_hypothesis:
            return OWLThing
        return self._best_hypothesis

    def best_hypothesis_accuracy(self) -> Optional[float]:
        """
        Get the accuracy of the best found hypothesis.

        Returns:
            Accuracy as float, or None if no hypothesis found.
        """
        return self._best_hypothesis_accuracy
