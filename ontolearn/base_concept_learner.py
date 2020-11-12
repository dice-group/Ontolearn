from abc import ABCMeta, abstractmethod
from owlready2 import get_ontology, World, rdfs, AnnotationPropertyClass

from .refinement_operators import ModifiedCELOERefinement
from .search import Node
from .search import CELOESearchTree
from .metrics import F1
from .heuristics import CELOEHeuristic
import types
from typing import List, AnyStr, Set
from .util import create_experiment_folder, create_logger
import numpy as np
import pandas as pd
import time
import random


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
    def __init__(self, knowledge_base=None, refinement_operator=None, heuristic_func=None, quality_func=None,
                 search_tree=None, max_num_of_concepts_tested=None, terminate_on_goal=None, ignored_concepts=None,
                 iter_bound=None, max_child_length=None, root_concept=None, verbose=None, name=None):

        self.kb = knowledge_base
        self.rho = refinement_operator
        self.heuristic_func = heuristic_func
        self.quality_func = quality_func
        self.search_tree = search_tree
        self.max_num_of_concepts_tested = max_num_of_concepts_tested
        self.terminate_on_goal = terminate_on_goal
        self.concepts_to_ignore = ignored_concepts
        self.iter_bound = iter_bound
        self.start_class = root_concept
        self.max_child_length = max_child_length
        self.verbose = verbose
        self.start_time = None
        self.goal_found = False
        self.max_length = 5
        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name=name, p=self.storage_path)

        # Memoization
        self.concepts_to_nodes = dict()
        self.__default_values()
        self.__sanity_checking()

    def __default_values(self):
        """
        Fill all params with plausible default values.
        """
        if self.rho is None:
            self.rho = ModifiedCELOERefinement(self.kb)
        self.rho.set_concepts_node_mapping(self.concepts_to_nodes)

        if self.heuristic_func is None:
            self.heuristic_func = CELOEHeuristic()

        if self.quality_func is None:
            self.quality_func = F1()
        if self.search_tree is None:
            self.search_tree = CELOESearchTree(quality_func=self.quality_func, heuristic_func=self.heuristic)
        else:
            self.search_tree.clean()
            self.search_tree.set_quality_func(self.quality_func)
            self.search_tree.set_heuristic_func(self.heuristic_func)

        if self.start_class is None:
            self.start_class = self.kb.thing
        if self.iter_bound is None:
            self.iter_bound = 1000

        if self.max_num_of_concepts_tested is None:
            self.max_num_of_concepts_tested = 1000
        if self.terminate_on_goal is None:
            self.terminate_on_goal = True

        if self.concepts_to_ignore is None:
            self.concepts_to_ignore = set()
        if self.verbose is None:
            self.verbose = 0

    def __sanity_checking(self):
        assert self.start_class
        assert self.search_tree is not None
        assert self.quality_func
        assert self.heuristic_func
        assert self.rho
        assert self.kb

        self.add_ignored_concepts(self.concepts_to_ignore)

    def add_ignored_concepts(self, ignore: Set[AnyStr]):

        if ignore:
            owl_concepts_to_ignore = set()
            for i in ignore:  # iterate over string representations of ALC concepts.
                found = False
                for k, v in self.kb.concepts.items():
                    if (i == k) or (i == v.str):
                        found = True
                        owl_concepts_to_ignore.add(v)
                        break
                if found is False:
                    raise ValueError(
                        '{0} could not found in \n{1} \n{2}.'.format(i, [_.str for _ in self.kb.concepts.values()],
                                                                     [uri for uri in self.kb.concepts.keys()]))
            self.concepts_to_ignore = owl_concepts_to_ignore  # use ALC concept representation instead of URI.

    def initialize_learning_problem(self, pos: Set[AnyStr], neg: Set[AnyStr], all_instances, ignore: Set[AnyStr]):
        """
        Determine the learning problem and initialize the search.
        1) Convert the string representation of an individuals into the owlready2 representation.
        2) Sample negative examples if necessary.
        3) Initialize the root and search tree.
        """
        self.reset_state()
        assert isinstance(pos, set) and isinstance(neg, set) and isinstance(all_instances, set)
        assert 0 < len(pos) < len(all_instances) and len(all_instances) > len(neg)
        if self.verbose > 1:
            self.logger.info('E^+:[ {0} ]'.format(', '.join(pos)))
            self.logger.info('E^-:[ {0}] '.format(', '.join(neg)))
            if ignore:
                self.logger.info('Concepts to ignore:{0}'.format(' '.join(ignore)))
        self.add_ignored_concepts(ignore)

        owl_ready_pos = set(self.kb.convert_uri_instance_to_obj_from_iterable(pos))
        if len(neg) == 0:  # if negatives are not provided, randomly sample.
            owl_ready_neg = set(random.sample(all_instances, len(owl_ready_pos)))
        else:
            owl_ready_neg = set(self.kb.convert_uri_instance_to_obj_from_iterable(neg))

        assert len(owl_ready_pos) == len(pos)
        assert len(owl_ready_neg) == len(neg)

        unlabelled = all_instances.difference(owl_ready_pos.union(owl_ready_neg))
        self.quality_func.set_positive_examples(owl_ready_pos)
        self.quality_func.set_negative_examples(owl_ready_neg)

        self.heuristic_func.set_positive_examples(owl_ready_pos)
        self.heuristic_func.set_negative_examples(owl_ready_neg)
        self.heuristic_func.set_unlabelled_examples(unlabelled)

        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)
        self.search_tree.heuristic_func.apply(root)
        self.search_tree.add(root)
        assert len(self.search_tree) == 1

    def terminate(self):
        if self.verbose == 1:
            self.logger.info('Elapsed runtime: {0} seconds'.format(round(time.time() - self.start_time, 4)))
            self.logger.info('Number of concepts tested:{0}'.format(self.number_of_tested_concepts))
            if self.goal_found:
                t = 'A goal concept found:{0}'.format(self.goal_found)
            else:
                t = 'Current best concept:{0}'.format(self.best_hypotheses(n=1)[0])
            self.logger.info(t)
        if self.verbose > 1:
            self.search_tree.show_search_tree('Final')
        return self

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

    @staticmethod
    def assign_labels_to_individuals(*, individuals: List, hypotheses: List[Node]) -> np.ndarray:
        """
        individuals: A list of owlready individuals.
        hypotheses: A

        Use each hypothesis as a binary function and assign 1 or 0 to each individual.

        return matrix of |individuals| x |hypotheses|
        """
        labels = np.zeros((len(individuals), len(hypotheses)))
        for ith_ind in range(len(individuals)):
            for jth_hypo in range(len(hypotheses)):
                if individuals[ith_ind] in hypotheses[jth_hypo].concept.instances:
                    labels[ith_ind][jth_hypo] = 1
        return labels

    def predict(self, individuals: List[AnyStr], hypotheses: List[Node] = None, n: int = None) -> pd.DataFrame:
        """
        individuals: A list of individuals/instances where each item is a string.
        hypotheses: A list of ALC concepts.
        n: integer denoting number of ALC concepts to extract from search tree if hypotheses=None.
        """
        assert isinstance(hypotheses, List)  # set would not work.
        individuals = self.kb.convert_uri_instance_to_obj_from_iterable(individuals)
        if hypotheses is None:
            try:
                assert isinstance(n, int) and n > 0
            except AssertionError:
                raise AssertionError('**n** must be positive integer.')
            hypotheses = self.best_hypotheses(n)

        return pd.DataFrame(data=self.assign_labels_to_individuals(individuals=individuals, hypotheses=hypotheses),
                            index=individuals, columns=[c.concept.str for c in hypotheses])

    @property
    def number_of_tested_concepts(self):
        return self.quality_func.applied

    def reset_state(self):
        """
        At each problem initialization, we recent previous info if available.
        @return:
        """
        self.concepts_to_nodes.clear()
        self.concepts_to_ignore.clear()
        self.kb.clean()
        self.search_tree.clean()
        self.quality_func.clean()
        self.heuristic_func.clean()
