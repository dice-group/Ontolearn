import logging
import time
import types
from abc import ABCMeta, abstractmethod
from typing import List, Set, Tuple, Dict, Optional, Iterable, Generic, TypeVar, ClassVar, Final, cast

import numpy as np
import pandas as pd

from owlapy import IRI
from owlapy.model import OWLClassExpression, OWLNamedIndividual, OWLOntologyManager, OWLOntology, AddImport, \
    OWLImportsDeclaration, OWLClass, OWLEquivalentClassesAxiom, OWLAnnotationAssertionAxiom, OWLAnnotation, \
    OWLAnnotationProperty, OWLLiteral
from owlapy.owlready2 import OWLOntologyManager_Owlready2
from owlapy.render import DLSyntaxObjectRenderer
from .abstracts import BaseRefinement, AbstractScorer, AbstractNode, AbstractHeuristic, \
    AbstractKnowledgeBase, AbstractLearningProblem, AbstractConceptNode
from ontolearn.owlready2.static_funcs import decompose_to_atomic
from .utils import oplogging

_N = TypeVar('_N', bound=AbstractConceptNode)

logger = logging.getLogger(__name__)


class BaseConceptLearner(Generic[_N], metaclass=ABCMeta):
    """
    Base class for Concept Learning approaches

    Learning problem definition, Let
        * K = (TBOX, ABOX) be a knowledge base.
        * \\ALCConcepts be a set of all ALC concepts.
        * \\hypotheses be a set of ALC concepts : \\hypotheses \\subseteq \\ALCConcepts.

        * K_N be a set of all instances.
        * K_C be a set of concepts defined in TBOX: K_C \\subseteq \\ALCConcepts
        * K_R be a set of properties/relations.

        * E^+, E^- be a set of positive and negative instances and the followings hold
            ** E^+ \\cup E^- \\subseteq K_N
            ** E^+ \\cap E^- = \\emptyset

    ##################################################################################################
        The goal is to to learn a set of concepts $\\hypotheses \\subseteq \\ALCConcepts$ such that
              ∀  H \\in \\hypotheses: { (K \\wedge H \\models E^+) \\wedge  \\neg( K \\wedge H \\models E^-) }.
    ##################################################################################################

    """
    __slots__ = 'kb', 'lp', 'operator', 'heuristic_func', 'quality_func', 'max_num_of_concepts_tested', \
                'terminate_on_goal', 'max_child_length', 'goal_found', 'start_class', 'iter_bound', 'max_runtime', \
                'start_time', 'name'

    name: ClassVar[str]

    kb: AbstractKnowledgeBase
    lp: AbstractLearningProblem
    operator: BaseRefinement
    heuristic_func: AbstractHeuristic
    quality_func: AbstractScorer
    max_num_of_concepts_tested: Optional[int]
    terminate_on_goal: Optional[bool]
    max_child_length: Optional[int]
    goal_found: bool
    start_class: Optional[OWLClassExpression]
    iter_bound: Optional[int]
    max_runtime: Optional[int]
    start_time: Optional[float]
    name: Optional[str]

    @abstractmethod
    def __init__(self,
                 knowledge_base: Optional[AbstractKnowledgeBase] = None,
                 learning_problem: Optional[AbstractLearningProblem] = None,
                 refinement_operator: Optional[BaseRefinement] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_child_length: Optional[int] = None,
                 root_concept: Optional[OWLClassExpression] = None):

        self.kb = knowledge_base
        self.lp = learning_problem
        self.operator = refinement_operator
        self.heuristic_func = heuristic_func
        self.quality_func = quality_func
        self.max_num_of_concepts_tested = max_num_of_concepts_tested
        self.terminate_on_goal = terminate_on_goal
        self.max_runtime = max_runtime
        self.iter_bound = iter_bound
        self.start_class = root_concept
        self.max_child_length = max_child_length
        # self.store_onto_flag = False
        self.start_time = None
        self.goal_found = False
        # self.storage_path, _ = create_experiment_folder()
        # self.last_path = None  # path of lastly stored onto.
        self.__default_values()
        self.__sanity_checking()

    def __default_values(self):
        """
        Fill all params with plausible default values.
        """
        if self.kb is None:
            assert isinstance(self.lp, AbstractLearningProblem)
            self.kb = self.lp.kb

        if self.operator is None:
            from ontolearn.refinement_operators import ModifiedCELOERefinement
            from ontolearn import KnowledgeBase
            assert isinstance(self.kb, KnowledgeBase)
            self.operator = ModifiedCELOERefinement(self.kb)

        if self.heuristic_func is None:
            from ontolearn.heuristics import CELOEHeuristic
            self.heuristic_func = CELOEHeuristic()

        if self.quality_func is None:
            from ontolearn.metrics import F1
            from ontolearn.learning_problem import PosNegLPStandard
            assert isinstance(self.lp, PosNegLPStandard)
            self.quality_func = F1(self.lp)

        if self.start_class is None:
            self.start_class = self.kb.thing
        if self.iter_bound is None:
            self.iter_bound = 10_000

        if self.max_num_of_concepts_tested is None:
            self.max_num_of_concepts_tested = 10_000
        if self.terminate_on_goal is None:
            self.terminate_on_goal = True
        if self.max_runtime is None:
            self.max_runtime = 5

        if self.max_child_length is None:
            self.max_child_length = 10

    def __sanity_checking(self):
        assert self.start_class
        assert self.quality_func
        assert self.heuristic_func
        assert self.operator
        assert self.kb

    # def store_ontology(self):
    #     """
    #
    #     @return:
    #     """
    #     # sanity checking.
    #     # (1) get all concepts
    #     # (2) serialize current kb.
    #     # (3) reload (2).
    #     # (4) get all reloaded concepts.
    #     # (5) (1) and (4) must be same.
    #     uri_all_concepts = set([get_full_iri(i) for i in self.kb._ontology.classes()])
    #     self.last_path = self.storage_path + '/' + self.kb.name + str(time.time()) + '.owl'
    #     self.kb.save(path=self.last_path)  # save
    #     d = get_ontology(self.last_path).load()  # load it.
    #     uri_all_concepts_loaded = set([get_full_iri(i) for i in d.classes()])
    #     assert uri_all_concepts == uri_all_concepts_loaded
    #     # check the base iri of ontologeis

    @abstractmethod
    def clean(self):
        """
        Clear all states of the concept learner
        """
        self.quality_func.clean()
        self.heuristic_func.clean()

    def train(self, *args, **kwargs):
        pass

    def terminate(self):
        """

        @return:
        """
        # if self.store_onto_flag:
        #     self.store_ontology()

        if logger.isEnabledFor(logging.INFO):
            logger.info('Elapsed runtime: {0} seconds'.format(round(time.time() - self.start_time, 4)))
            logger.info('Number of concepts tested:{0}'.format(self.number_of_tested_concepts))
            if self.goal_found:
                t = 'A goal concept found:{0}'.format(self.goal_found)
            else:
                t = 'Current best concept:{0}'.format(list(self.best_hypotheses(n=1))[0])
            logger.info(t)

        if logger.isEnabledFor(oplogging.TRACE):
            self.show_search_tree('Final')

        return self

    @abstractmethod
    def next_node_to_expand(self, *args, **kwargs):
        pass

    @abstractmethod
    def downward_refinement(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def fit_from_iterable(self, dataset: List[Tuple[str, Set, Set]], max_runtime: int = None) -> List[Dict]:
        if max_runtime:
            self.max_runtime = max_runtime

        results = []
        assert isinstance(dataset, List)
        for (alc_concept_str, positives, negatives) in dataset:
            # self.logger.info('Concept:{0}\tE^+:[{1}] \t E^-:[{2}]'.format(alc_concept_str, len(positives), len(negatives)))
            start_time = time.time()
            self.fit(pos=positives, neg=negatives)
            h = self.best_hypotheses(1)[0]
            individuals = self.kb.convert_owlready2_individuals_to_uri_from_iterable(h.concept.instances)

            from ontolearn.metrics import F1
            f_measure = F1().score(pos=positives, neg=negatives, instances=individuals)
            from ontolearn.metrics import Accuracy
            accuracy = Accuracy().score(pos=positives, neg=negatives, instances=individuals)
            results.append({'Prediction': h.concept.str,
                            'F-measure': f_measure,
                            'Accuracy': accuracy,
                            'Runtime': time.time() - start_time})
        return results

    @abstractmethod
    def best_hypotheses(self, n=10) -> Iterable[_N]:
        pass

    @abstractmethod
    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        pass

    def assign_labels_to_individuals(self, *, individuals: List[OWLNamedIndividual], hypotheses: List[_N]) -> np.ndarray:
        """
        Use each hypothesis as a binary function and assign 1 or 0 to each individual.

        Args:
            individuals: A list of owlready individuals.
            hypotheses: A

        Returns:
            return matrix of |individuals| x |hypotheses|
        """
        labels = np.zeros((len(individuals), len(hypotheses)))
        for jth_hypo in range(len(hypotheses)):
            node = hypotheses[jth_hypo]

            kb_individuals = self.kb.individuals_set(node.concept)
            for ith_ind in range(len(individuals)):
                ind = individuals[ith_ind]

                kb_test = self.kb.individuals_set(ind)
                if kb_test in kb_individuals:
                    labels[ith_ind][jth_hypo] = 1
        return labels

    def predict(self, individuals: List[OWLNamedIndividual], hypotheses: Optional[List[_N]] = None,
                n: Optional[int] = None) -> pd.DataFrame:
        """

        Args:
            individuals: A list of individuals/instances where each item is a string.
            hypotheses: A list of ALC concepts.
            n: integer denoting number of ALC concepts to extract from search tree if hypotheses=None.
        """
        assert isinstance(individuals, List)  # set would not work.
        if hypotheses is None:
            try:
                assert isinstance(n, int) and n > 0
            except AssertionError:
                raise ValueError('**n** must be positive integer.')
            hypotheses = list(self.best_hypotheses(n))

        dlr = DLSyntaxObjectRenderer()

        return pd.DataFrame(data=self.assign_labels_to_individuals(individuals=individuals, hypotheses=hypotheses),
                            index=[dlr.render(_) for _ in individuals],
                            columns=[dlr.render(c.concept) for c in hypotheses])

    @property
    def number_of_tested_concepts(self):
        return self.quality_func.applied

    def save_best_hypothesis(self, n: int = 10, path='Predictions', rdf_format='rdfxml') -> None:
        best = list(self.best_hypotheses(n))
        try:
            assert len(best) >= n
        except AssertionError:
            logger.warning("There were only %d results", len(best))

        SNS: Final = 'https://dice-research.org/predictions-schema/'
        NS: Final = 'https://dice-research.org/predictions/' + str(time.time()) + '#'
        manager: OWLOntologyManager = OWLOntologyManager_Owlready2()

        ontology: OWLOntology = manager.create_ontology(IRI.create(NS))
        from ontolearn import KnowledgeBase
        assert isinstance(self.kb, KnowledgeBase)
        manager.load_ontology(IRI.create(self.kb.path))
        kb_iri = self.kb.ontology().get_ontology_id().get_ontology_iri()
        manager.apply_change(AddImport(ontology, OWLImportsDeclaration(kb_iri)))
        for ith, h in enumerate(self.best_hypotheses(n=n)):
            cls_a: OWLClass = OWLClass(IRI.create(NS, "Pred_" + str(ith)))
            equivalent_classes_axiom = OWLEquivalentClassesAxiom(cls_a, h.concept)
            manager.add_axiom(ontology, equivalent_classes_axiom)

            try:
                from ontolearn.search import _NodeQuality
                h = cast(_NodeQuality, h)
                quality = h.quality
            except AttributeError:
                quality = None

            from ontolearn.metrics import Accuracy
            from ontolearn.metrics import F1
            if isinstance(self.quality_func, Accuracy):
                accuracy = OWLAnnotationAssertionAxiom(cls_a.get_iri(), OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(SNS, "accuracy")), OWLLiteral(quality)))
                manager.add_axiom(ontology, accuracy)
            elif isinstance(self.quality_func, F1):
                f1_score = OWLAnnotationAssertionAxiom(cls_a.get_iri(), OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(SNS, "f1_score")), OWLLiteral(quality)))
                manager.add_axiom(ontology, f1_score)

        manager.save_ontology(ontology, IRI.create('file:/' + path + '.owl'))

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
