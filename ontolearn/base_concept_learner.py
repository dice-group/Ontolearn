import logging
import time
from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import List, Tuple, Dict, Optional, Iterable, Generic, TypeVar, ClassVar, Final, Union, cast, Callable, Type

import numpy as np
import pandas as pd

from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import F1, Accuracy
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.search import _NodeQuality

from owlapy.model import OWLDeclarationAxiom, OWLNamedIndividual, OWLOntologyManager, OWLOntology, AddImport, \
    OWLImportsDeclaration, OWLClass, OWLEquivalentClassesAxiom, OWLAnnotationAssertionAxiom, OWLAnnotation, \
    OWLAnnotationProperty, OWLLiteral, IRI, OWLClassExpression, OWLReasoner, OWLAxiom, OWLThing
from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLOntology_Owlready2, OWLReasoner_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.render import DLSyntaxObjectRenderer
from .abstracts import BaseRefinement, AbstractScorer, AbstractHeuristic, \
    AbstractConceptNode, AbstractLearningProblem
from .utils import oplogging

_N = TypeVar('_N', bound=AbstractConceptNode)  #:
_X = TypeVar('_X', bound=AbstractLearningProblem)  #:
Factory = Callable

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

    The goal is to to learn a set of concepts $\\hypotheses \\subseteq \\ALCConcepts$ such that
          ∀  H \\in \\hypotheses: { (K \\wedge H \\models E^+) \\wedge  \\neg( K \\wedge H \\models E^-) }.

    """
    __slots__ = 'kb', 'quality_func', 'max_num_of_concepts_tested', 'terminate_on_goal', 'max_runtime', \
                'start_time', '_goal_found', '_number_of_tested_concepts'

    name: ClassVar[str]

    kb: KnowledgeBase
    quality_func: Optional[AbstractScorer]
    max_num_of_concepts_tested: Optional[int]
    terminate_on_goal: Optional[bool]
    _goal_found: bool
    _number_of_tested_concepts: int
    max_runtime: Optional[int]
    start_time: Optional[float]

    @abstractmethod
    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 quality_func: Optional[AbstractScorer] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 terminate_on_goal: Optional[bool] = None):
        """Create a new base concept learner

        Args:
            knowledge_base: knowledge base which is used to learn and test concepts. required, but can be taken
                from the learning problem if not specified
            quality_func: function to evaluate the quality of solution concepts. defaults to `F1`
            max_num_of_concepts_tested: limit to stop the algorithm after n concepts tested. defaults to 10_000
            max_runtime: limit to stop the algorithm after n seconds. defaults to 5
            terminate_on_goal: whether to stop the algorithm if a perfect solution is found. defaults to True
        """
        self.kb = knowledge_base
        self.quality_func = quality_func
        self.max_num_of_concepts_tested = max_num_of_concepts_tested
        self.terminate_on_goal = terminate_on_goal
        self.max_runtime = max_runtime
        # self.store_onto_flag = False
        self.start_time = None
        self._goal_found = False
        self._number_of_tested_concepts = 0
        # self.storage_path, _ = create_experiment_folder()
        # self.last_path = None  # path of lastly stored onto.
        self.__default_values()
        self.__sanity_checking()

    def __default_values(self):
        """
        Fill all params with plausible default values.
        """

        if self.quality_func is None:
            self.quality_func = F1()
        if self.max_num_of_concepts_tested is None:
            self.max_num_of_concepts_tested = 10_000
        if self.terminate_on_goal is None:
            self.terminate_on_goal = True
        if self.max_runtime is None:
            self.max_runtime = 5

    def __sanity_checking(self):
        assert self.quality_func
        assert self.kb

    @abstractmethod
    def clean(self):
        """
        Clear all states of the concept learner
        """
        self._number_of_tested_concepts = 0
        self._goal_found = False
        self.start_time = None

    def train(self, *args, **kwargs):
        pass

    def terminate(self):
        """This method is called when the search algorithm terminates

        If INFO log level is enabled, it prints out some statistics like runtime and concept tests to the logger

        Returns:
            the concept learner object itself
        """
        # if self.store_onto_flag:
        #     self.store_ontology()

        if logger.isEnabledFor(logging.INFO):
            logger.info('Elapsed runtime: {0} seconds'.format(round(time.time() - self.start_time, 4)))
            logger.info('Number of concepts tested: {0}'.format(self.number_of_tested_concepts))
            if self._goal_found:
                t = 'A goal concept found: {0}'
            else:
                t = 'Current best concept: {0}'
            logger.info(t.format(list(self.best_hypotheses(n=1))[0]))

        return self

    def construct_learning_problem(self, type_: Type[_X], xargs: Tuple, xkwargs: Dict) -> _X:
        """Construct learning problem of given type based on args and kwargs. If a learning problem is contained in
        args or the learning_problem kwarg, it is used. otherwise, a new learning problem of type type_ is created
        with args and kwargs as parameters.

        Args:
            type_: type of the learning problem
            xargs: the positional arguments
            xkwargs: the keyword arguments

        Returns:
            the learning problem
        """
        learning_problem = xkwargs.pop("learning_problem", None)
        if learning_problem is None and xargs and isinstance(xargs[0], AbstractLearningProblem):
            learning_problem = xargs[0]
            xargs = xargs[1:]
        if learning_problem is None:
            learning_problem = type_(*xargs, **xkwargs)
        assert isinstance(learning_problem, type_)
        return learning_problem

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Run the concept learning algorithm according to its configuration

        Once finished, the results can be queried with the `best_hypotheses` function"""
        pass

    @abstractmethod
    def best_hypotheses(self, n=10) -> Iterable[_N]:
        """Get the current best found hypotheses according to the quality

        Args:
            n: Maximum number of results

        Returns:
            iterable with hypotheses in form of search tree nodes
        """
        pass

    def _assign_labels_to_individuals(self, individuals: List[OWLNamedIndividual],
                                      hypotheses: List[OWLClassExpression],
                                      reasoner: Optional[OWLReasoner] = None) -> np.ndarray:
        """
        Use each class expression as a hypothesis, and use it as a binary function to assign 1 or 0 to each
        individual.

        Args:
            individuals: A list of OWL individuals.
            hypotheses: A list of class expressions.
            reasoner: Optionally use a different reasoner. If reasoner=None, the knowledge base of the concept
                      learner is used.

        Returns:
            matrix of \\|individuals\\| x \\|hypotheses\\|
        """
        retrieval_func = self.kb.individuals_set if reasoner is None else reasoner.instances

        labels = np.zeros((len(individuals), len(hypotheses)))
        for idx_hyp, hyp in enumerate(hypotheses):
            kb_individuals = set(retrieval_func(hyp))  # type: ignore
            for idx_ind, ind in enumerate(individuals):
                if ind in kb_individuals:
                    labels[idx_ind][idx_hyp] = 1
        return labels

    def predict(self, individuals: List[OWLNamedIndividual],
                hypotheses: Optional[List[Union[_N, OWLClassExpression]]] = None,
                axioms: Optional[List[OWLAxiom]] = None,
                n: int = 10,
                reasoner: Optional[OWLReasoner] = None) -> pd.DataFrame:
        """Creates a binary data frame showing for each individual whether it is entailed in the given hypotheses
        (class expressions). The individuals do not have to be in the ontology/knowledge base yet. In that case,
        axioms describing these individuals must be provided.

        The state of the knowledge base/ontology is not changed, any provided axioms will be removed again.

        Args:
            individuals: A list of individuals/instances.
            hypotheses: (Optional) A list of search tree nodes or class expressions. If not provided, the
                        current :func:`BaseConceptLearner.best_hypothesis` of the concept learner are used.
            axioms: (Optional) A list of axioms that are not in the current knowledge base/ontology.
                    If the individual list contains individuals that are not in the ontology yet, axioms
                    describing these individuals must be provided. The argument can also be used to add
                    arbitrary axioms to the ontology for the prediction.
            n: Integer denoting number of ALC concepts to extract from search tree if hypotheses=None.
            reasoner: (Optional) Use a different reasoner. If reasoner=None, the knowledge base of the concept
                      learner is used.

        Returns:
            Pandas data frame with dimensions |individuals|*|hypotheses| indicating for each individual and each
            hypothesis whether the individual is entailed in the hypothesis
        """
        new_individuals = set(individuals) - self.kb.individuals_set(OWLThing)
        if len(new_individuals) > 0 and (axioms is None or len(axioms) == 0):
            raise RuntimeError('If individuals are provided that are not in the knowledge base yet, a list of axioms '
                               f'has to be provided. New Individuals:\n{new_individuals}.')

        # If axioms are provided they need to be added to the ontology
        if axioms is not None:
            ontology: OWLOntology = cast(OWLOntology_Owlready2, self.kb.ontology())
            manager: OWLOntologyManager = ontology.get_owl_ontology_manager()
            for axiom in axioms:
                manager.add_axiom(ontology, axiom)
            reasoner = OWLReasoner_Owlready2_TempClasses(ontology) if reasoner is None else reasoner

        if hypotheses is None:
            hypotheses = [hyp.concept for hyp in self.best_hypotheses(n)]
        else:
            hypotheses = [(hyp.concept if isinstance(hyp, AbstractConceptNode) else hyp) for hyp in hypotheses]

        renderer = DLSyntaxObjectRenderer()
        predictions = pd.DataFrame(data=self._assign_labels_to_individuals(individuals, hypotheses, reasoner),
                                   index=[renderer.render(i) for i in individuals],
                                   columns=[renderer.render(c) for c in hypotheses])

        # Remove the axioms from the ontology
        if axioms is not None:
            for axiom in axioms:
                manager.remove_axiom(ontology, axiom)
            for ind in individuals:
                manager.remove_axiom(ontology, OWLDeclarationAxiom(ind))

        return predictions

    @property
    def number_of_tested_concepts(self):
        return self._number_of_tested_concepts

    def save_best_hypothesis(self, n: int = 10, path: str = 'Predictions', rdf_format: str = 'rdfxml') -> None:
        """Serialise the best hypotheses to a file

        Args:
            n: maximum number of hypotheses to save
            path: filename base (extension will be added automatically)
            rdf_format: serialisation format. currently supported: "rdfxml"
        """
        SNS: Final = 'https://dice-research.org/predictions-schema/'
        NS: Final = 'https://dice-research.org/predictions/' + str(time.time()) + '#'

        if rdf_format != 'rdfxml':
            raise NotImplementedError(f'Format {rdf_format} not implemented.')

        assert isinstance(self.kb, KnowledgeBase)

        best = list(self.best_hypotheses(n))
        try:
            assert len(best) >= n
        except AssertionError:
            logger.warning("There were only %d results", len(best))

        manager: OWLOntologyManager = OWLOntologyManager_Owlready2()

        ontology: OWLOntology = manager.create_ontology(IRI.create(NS))
        manager.load_ontology(IRI.create(self.kb.path))
        manager.apply_change(AddImport(ontology, OWLImportsDeclaration(IRI.create('file://' + self.kb.path))))
        for ith, h in enumerate(self.best_hypotheses(n=n)):
            cls_a: OWLClass = OWLClass(IRI.create(NS, "Pred_" + str(ith)))
            equivalent_classes_axiom = OWLEquivalentClassesAxiom([cls_a, h.concept])
            manager.add_axiom(ontology, equivalent_classes_axiom)

            try:
                assert isinstance(h, _NodeQuality)
                quality = h.quality
            except AttributeError:
                quality = None

            if isinstance(self.quality_func, Accuracy):
                accuracy = OWLAnnotationAssertionAxiom(cls_a.get_iri(), OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(SNS, "accuracy")), OWLLiteral(quality)))
                manager.add_axiom(ontology, accuracy)
            elif isinstance(self.quality_func, F1):
                f1_score = OWLAnnotationAssertionAxiom(cls_a.get_iri(), OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(SNS, "f1_score")), OWLLiteral(quality)))
                manager.add_axiom(ontology, f1_score)

        manager.save_ontology(ontology, IRI.create('file:/' + path + '.owl'))

    def load_hypotheses(self, path: str) -> Iterable[OWLClassExpression]:
        """Loads hypotheses (class expressions) from a file saved by :func:`BaseConceptLearner.save_best_hypothesis`

        Args:
            path: Path to the file containing hypotheses
        """
        manager: OWLOntologyManager_Owlready2 = OWLOntologyManager_Owlready2()
        ontology: OWLOntology_Owlready2 = manager.load_ontology(IRI.create('file://' + path))
        reasoner = OWLReasoner_Owlready2(ontology)
        yield from chain.from_iterable(reasoner.equivalent_classes(c) for c in ontology.classes_in_signature())


class RefinementBasedConceptLearner(BaseConceptLearner[_N]):
    """
    Base class for refinement based Concept Learning approaches

    """
    __slots__ = 'operator', 'heuristic_func', 'max_child_length', 'start_class', 'iter_bound'

    operator: Optional[BaseRefinement]
    heuristic_func: Optional[AbstractHeuristic]
    max_child_length: Optional[int]
    start_class: Optional[OWLClassExpression]
    iter_bound: Optional[int]

    @abstractmethod
    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 refinement_operator: Optional[BaseRefinement] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_child_length: Optional[int] = None,
                 root_concept: Optional[OWLClassExpression] = None):
        """Create a new base concept learner

        Args:
            knowledge_base: knowledge base which is used to learn and test concepts. required, but can be taken
                from the learning problem if not specified
            refinement_operator: operator used to generate refinements. defaults to `ModifiedCELOERefinement`
            heuristic_func: function to guide the search heuristic. defaults to `CELOEHeuristic`
            quality_func: function to evaluate the quality of solution concepts. defaults to `F1`
            max_num_of_concepts_tested: limit to stop the algorithm after n concepts tested. defaults to 10_000
            max_runtime: limit to stop the algorithm after n seconds. defaults to 5
            terminate_on_goal: whether to stop the algorithm if a perfect solution is found. defaults to True
            iter_bound: limit to stop the algorithm after n refinement steps were done. defaults to 10_000
            max_child_length: limit the length of concepts generated by the refinement operator. defaults to 10.
                only used if refinement_operator is not specified.
            root_concept: the start concept to begin the search from. defaults to OWL Thing
        """
        super().__init__(knowledge_base=knowledge_base,
                         quality_func=quality_func,
                         max_num_of_concepts_tested=max_num_of_concepts_tested,
                         max_runtime=max_runtime,
                         terminate_on_goal=terminate_on_goal)

        self.operator = refinement_operator
        self.heuristic_func = heuristic_func
        self.iter_bound = iter_bound
        self.start_class = root_concept
        self.max_child_length = max_child_length
        self.__default_values()
        self.__sanity_checking()

    def __default_values(self):
        """
        Fill all params with plausible default values.
        """
        if self.max_child_length is None:
            self.max_child_length = 10

        if self.operator is None:
            assert isinstance(self.kb, KnowledgeBase)
            self.operator = ModifiedCELOERefinement(self.kb, max_child_length=self.max_child_length)

        if self.heuristic_func is None:
            self.heuristic_func = CELOEHeuristic()

        if self.start_class is None:
            self.start_class = self.kb.thing
        if self.iter_bound is None:
            self.iter_bound = 10_000

    def __sanity_checking(self):
        assert self.start_class
        assert self.heuristic_func
        assert self.operator

    def terminate(self):
        if logger.isEnabledFor(oplogging.TRACE):
            self.show_search_tree('Final')
        return super().terminate()

    @abstractmethod
    def next_node_to_expand(self, *args, **kwargs):
        """
        Return from the search tree the most promising search tree node to use for the next refinement step

        Returns:
            _N: next search tree node to refine
        """
        pass

    @abstractmethod
    def downward_refinement(self, *args, **kwargs):
        """execute one refinement step of a refinement based learning algorithm

        Args:
            node (_N): the search tree node on which to refine

        Returns:
            Iterable[_N]: refinement results as new search tree nodes (they still need to be added to the tree)
        """
        pass

    @abstractmethod
    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        """A debugging function to print out the current search tree and the current n best found hypotheses to
        standard output

        Args:
            heading_step: a message to display at the beginning of the output
            top_n: the number of currently best hypotheses to print out
        """
        pass
