from abc import ABCMeta, abstractmethod
from owlready2 import get_ontology, World, rdfs, AnnotationPropertyClass

from .refinement_operators import ModifiedCELOERefinement
from .search import Node
from .search import CELOESearchTree
from .metrics import F1
from .heuristics import CELOEHeuristic
import types
from typing import List,AnyStr
#from .util import serialize_concepts
import numpy as np
import pandas as pd

class BaseConceptLearner(metaclass=ABCMeta):
    """
    Base class for Concept Learning approaches

    Learning problem definition, Let
        * K = (TBOX, ABOX) be a knowledge base.
        * \ALCConcepts be a set of all ALC concepts.
        * \hypotheses be a set of ALC concepts : \hypotheses \subseteq \ALCConcepts.

        * K_N be a set of all instances.
        * K_C be a set of concepts defined in TBOX: K_C \subseteq \ALCConcepts
        * K_R be a set of properties/relations.

        * E^+, E^- be a set of positive and negative instances and the followings hold
            ** E^+ \cup E^- \subseteq K_N
            ** E^+ \cap E^- = \emptyset

    ##################################################################################################
        The goal is to to learn a set of concepts $\hypotheses \subseteq \ALCConcepts$ such that
              ∀  H \in \hypotheses: { (K \wedge H \models E^+) \wedge  \neg( K \wedge H \models E^-) }.
    ##################################################################################################

    """

    @abstractmethod
    def __init__(self, knowledge_base=None, refinement_operator=None,
                 quality_func=None,
                 heuristic_func=None,
                 search_tree=None,
                 terminate_on_goal=True,
                 iter_bound=10,
                 max_child_length=10,
                 verbose=True, max_num_of_concepts_tested=None, ignored_concepts=None, root_concept=None):
        if ignored_concepts is None:
            ignored_concepts = {}
        assert knowledge_base
        self.kb = knowledge_base
        self.heuristic = heuristic_func
        self.quality_func = quality_func
        self.rho = refinement_operator
        self.search_tree = search_tree
        self.max_num_of_concepts_tested = max_num_of_concepts_tested

        self.concepts_to_ignore = ignored_concepts
        self.start_class = root_concept

        # Memoization
        self.concepts_to_nodes = dict()
        self.iter_bound = iter_bound
        self.terminate_on_goal = terminate_on_goal
        self.verbose = verbose

        if self.rho is None:
            self.rho = ModifiedCELOERefinement(self.kb, max_child_length=max_child_length)
        self.rho.set_concepts_node_mapping(self.concepts_to_nodes)

        if self.heuristic is None:
            self.heuristic = CELOEHeuristic()

        if self.quality_func is None:
            self.quality_func = F1()

        if self.search_tree is None:
            self.search_tree = CELOESearchTree(quality_func=self.quality_func, heuristic_func=self.heuristic)
        else:
            self.search_tree.set_quality_func(self.quality_func)
            self.search_tree.set_heuristic_func(self.heuristic)

        if self.start_class is None:
            self.start_class = self.kb.thing

        assert self.start_class
        assert self.search_tree is not None
        assert self.quality_func
        assert self.heuristic
        assert self.rho

    def get_metric_key(self, key: str):
        if key == 'quality':
            metric = self.quality_func.name
            attribute = key
        elif key == 'heuristic':
            metric = self.heuristic.name
            attribute = key
        elif key == 'length':
            metric = key
            attribute = key
        else:
            raise ValueError('Invalid key:{0}'.format(key))
        return metric, attribute

    def show_best_predictions(self, key='quality', top_n=10, serialize_name=None, rdf_format='xml'):
        raise NotImplementedError('Use best_hypotheses method to obtain predictions.')
        predictions = self.search_tree.show_best_nodes(top_n, key=key)
        if serialize_name is not None:
            if key == 'quality':
                metric = self.quality_func.name
                attribute = key
            elif key == 'heuristic':
                metric = self.heuristic.name
                attribute = key
            elif key == 'length':
                metric = key
                attribute = key
            else:
                raise ValueError

            print('Currently. Serialization is not available')
            serialize_concepts(concepts=predictions,
                               serialize_name=serialize_name,
                               metric=metric,
                               attribute=attribute, rdf_format=rdf_format)

    def extend_ontology(self, top_n_concepts=10, key='quality', rdf_format='xml'):
        """
        1) Obtain top N nodes from search tree.
        2) Extend ABOX by including explicit type information for all instances belonging to concepts (1)
        """
        raise NotImplementedError('Not yet implemented.')
        # This module needs to be tested
        # if World().get_ontology(self.path).load(reload=True) used
        # saving owlready ontology is not working.
        self.search_tree.sort_search_tree_by_decreasing_order(key=key)
        for (ith, node) in enumerate(self.search_tree):
            if ith <= top_n_concepts:
                self.kb.apply_type_enrichment(node.concept)
            else:
                break

        folder = self.kb.path[:self.kb.path.rfind('/')] + '/'
        kb_name = 'enriched_' + self.kb.name
        self.kb.save(folder + kb_name + '.owl', rdf_format=rdf_format)

    @abstractmethod
    def initialize_root(self):
        pass

    @abstractmethod
    def next_node_to_expand(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply_rho(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def best_hypotheses(self, n=10) -> List[Node]:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        return [i for i in self.search_tree.get_top_n_nodes(n)]

    def predict(self, individuals: List[AnyStr], hypotheses: List[Node] = None, n: int = None):
        assert isinstance(hypotheses, List)  # set would not work.
        if hypotheses:

            for ith, ind in enumerate(individuals):
                if isinstance(ind, str):
                    try:
                        individuals[ith] = self.kb.convert_uri_instance_to_obj(ind)
                    except KeyError:
                        print('Item in individuals: {0} can not be found in the ontology'.format(ind))
                elif isinstance(type(ind), ThingClass):
                    continue  # is
                else:
                    raise ValueError('Wrong format individual **{0}**,\t type:{1}'.format(ind, type(ind)))

            labels = np.zeros((len(individuals), len(hypotheses)))
            for ith_ind in range(len(individuals)):
                for jth_hypo in range(len(hypotheses)):
                    if individuals[ith_ind] in hypotheses[jth_hypo].concept.instances:
                        labels[ith_ind][jth_hypo] = 1
        else:
            try:
                assert isinstance(n, int) and n > 0
            except AssertionError:
                print('**n** must be positive integer.')
                exit(1)
            hypotheses = self.best_hypotheses(n)
            labels = np.zeros((len(individuals), len(hypotheses)))
            for ith_ind in range(len(individuals)):
                for jth_hypo in range(len(hypotheses)):
                    if individuals[ith_ind] in hypotheses[jth_hypo].concept.instances:
                        labels[ith_ind][jth_hypo] = 1

        return pd.DataFrame(labels, index=individuals, columns=[c.concept.str for c in hypotheses])

    @property
    def number_of_tested_concepts(self):
        return self.quality_func.applied

