"""Base classes of concept learners."""

import logging
import time
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict, Optional, Iterable, Generic, TypeVar, ClassVar, Final, Union, cast, Callable, Type
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os

from owlapy.class_expression import OWLClass, OWLClassExpression, OWLThing
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLDeclarationAxiom, OWLEquivalentClassesAxiom, OWLAxiom
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology import OWLOntology
from owlapy.owl_ontology_manager import OWLOntologyManager, AddImport, OWLImportsDeclaration
from owlapy.owl_reasoner import OWLReasoner

from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import F1
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.base import OWLOntologyManager_Owlready2, OWLOntology_Owlready2
from ontolearn.base import OWLReasoner_Owlready2_ComplexCEInstances
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
    @TODO: CD: Why should this class inherit from AbstractConceptNode ?
    @TODO: CD: This class should be redefined. An owl class expression learner does not need to be a search based model.

    Base class for Concept Learning approaches.

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

    The goal is to learn a set of concepts $\\hypotheses \\subseteq \\ALCConcepts$ such that
          ∀  H \\in \\hypotheses: { (K \\wedge H \\models E^+) \\wedge  \\neg( K \\wedge H \\models E^-) }.

    Attributes:
        kb (KnowledgeBase): The knowledge base that the concept learner is using.
        quality_func (AbstractScorer) The quality function to be used.
        max_num_of_concepts_tested (int) Limit to stop the algorithm after n concepts tested.
        terminate_on_goal (bool): Whether to stop the algorithm if a perfect solution is found.
        max_runtime (int): Limit to stop the algorithm after n seconds.
        _number_of_tested_concepts (int): Yes, you got it. This stores the number of tested concepts.
        reasoner (OWLReasoner): The reasoner that this model is using.
        start_time (float): The time when :meth:`fit` starts the execution. Used to calculate the total time :meth:`fit`
                            takes to execute.
    """
    __slots__ = 'kb', 'reasoner', 'quality_func', 'max_num_of_concepts_tested', 'terminate_on_goal', 'max_runtime', \
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
                 reasoner: Optional[OWLReasoner] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 terminate_on_goal: Optional[bool] = None):
        """Create a new base concept learner

        Args:
            knowledge_base: Knowledge base which is used to learn and test concepts. required, but can be taken
                from the learning problem if not specified.
            quality_func: Function to evaluate the quality of solution concepts. Defaults to `F1`.
            max_num_of_concepts_tested: Limit to stop the algorithm after n concepts tested. Defaults to 10'000.
            max_runtime: Limit to stop the algorithm after n seconds. Defaults to 5.
            terminate_on_goal: Whether to stop the algorithm if a perfect solution is found. Defaults to True.
            reasoner: Optionally use a different reasoner. If reasoner=None, the reasoner of the :attr:`knowledge_base`
                is used.
        """
        self.kb = knowledge_base
        self.reasoner = reasoner
        self.quality_func = quality_func
        self.max_num_of_concepts_tested = max_num_of_concepts_tested
        self.terminate_on_goal = terminate_on_goal
        self.max_runtime = max_runtime

        self.start_time = None
        self._goal_found = False
        self._number_of_tested_concepts = 0

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
        Clear all states of the concept learner.
        """
        self._number_of_tested_concepts = 0
        self._goal_found = False
        self.start_time = None

    def train(self, *args, **kwargs):
        """Train RL agent on learning problems.

        Returns:
            self.
        """
        pass

    def terminate(self):
        """This method is called when the search algorithm terminates.

        If INFO log level is enabled, it prints out some statistics like runtime and concept tests to the logger.

        Returns:
            The concept learner object itself.
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
            logger.info(t.format(self.best_hypotheses(n=1)))

        return self

    def construct_learning_problem(self, type_: Type[_X], xargs: Tuple, xkwargs: Dict) -> _X:
        """Construct learning problem of given type based on args and kwargs. If a learning problem is contained in
        args or the learning_problem kwarg, it is used. otherwise, a new learning problem of type type_ is created
        with args and kwargs as parameters.

        Args:
            type_: Type of the learning problem.
            xargs: The positional arguments.
            xkwargs: The keyword arguments.

        Returns:
            The learning problem.
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
        """Run the concept learning algorithm according to its configuration.

        Once finished, the results can be queried with the `best_hypotheses` function."""
        pass

    @abstractmethod
    def best_hypotheses(self, n=10) -> Iterable[OWLClassExpression]:
        """Get the current best found hypotheses according to the quality.

        Args:
            n: Maximum number of results.

        Returns:
            Iterable with hypotheses in form of search tree nodes.

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

        Returns:
            Matrix of \\|individuals\\| x \\|hypotheses\\|.
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
                hypotheses: Optional[Union[OWLClassExpression, List[Union[_N, OWLClassExpression]]]] = None,
                axioms: Optional[List[OWLAxiom]] = None,
                n: int = 10) -> pd.DataFrame:
        """
        @TODO: CD: Predicting an individual can be done by a retrieval function not a concept learner
        @TODO: A concept learner learns an owl class expression.
        @TODO: This learned expression can be used as a binary predictor.


        Creates a binary data frame showing for each individual whether it is entailed in the given hypotheses
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

        Returns:
            Pandas data frame with dimensions |individuals|*|hypotheses| indicating for each individual and each
            hypothesis whether the individual is entailed in the hypothesis.
        """
        reasoner = self.reasoner
        new_individuals = set(individuals) - self.kb.individuals_set(OWLThing)
        if len(new_individuals) > 0 and (axioms is None or len(axioms) == 0):
            raise RuntimeError('If individuals are provided that are not in the knowledge base yet, a list of axioms '
                               f'has to be provided. New Individuals:\n{new_individuals}.')

        # If axioms are provided they need to be added to the ontology
        if axioms is not None:
            ontology: OWLOntology = cast(OWLOntology_Owlready2, self.kb.ontology)
            manager: OWLOntologyManager = ontology.get_owl_ontology_manager()
            for axiom in axioms:
                manager.add_axiom(ontology, axiom)
            if reasoner is None:
                reasoner = OWLReasoner_Owlready2_ComplexCEInstances(ontology)

        if hypotheses is None:
            hypotheses = [hyp.concept for hyp in self.best_hypotheses(n)]
        elif isinstance(hypotheses, list):
            hypotheses = [(hyp.concept if isinstance(hyp, AbstractConceptNode) else hyp) for hyp in hypotheses]
        else:
            hypotheses = [hypotheses]

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
        """Serialise the best hypotheses to a file.
        @TODO: CD: This function should be deprecated.
        @TODO: CD: Saving owl class expressions into disk should be disentangled from a concept earner

        Args:
            n: Maximum number of hypotheses to save.
            path: Filename base (extension will be added automatically).
            rdf_format: Serialisation format. currently supported: "rdfxml".
        """
        SNS: Final = 'https://dice-research.org/predictions-schema/'
        NS: Final = 'https://dice-research.org/predictions/' + str(time.time()) + '#'

        if rdf_format != 'rdfxml':
            raise NotImplementedError(f'Format {rdf_format} not implemented.')

        assert isinstance(self.kb, KnowledgeBase)

        best = self.best_hypotheses(n)
        if len(best) >= n:
            logger.warning("There was/were only %d unique result/-s found", len(best))

        manager: OWLOntologyManager = OWLOntologyManager_Owlready2()

        ontology: OWLOntology = manager.create_ontology(IRI.create(NS))
        manager.load_ontology(IRI.create(self.kb.path))
        manager.apply_change(AddImport(ontology, OWLImportsDeclaration(IRI.create('file://' + self.kb.path))))
        for ith, h in enumerate(self.best_hypotheses(n=n)):
            cls_a: OWLClass = OWLClass(IRI.create(NS, "Pred_" + str(ith)))
            equivalent_classes_axiom = OWLEquivalentClassesAxiom([cls_a, h])
            manager.add_axiom(ontology, equivalent_classes_axiom)
            # @TODO:CD: We should find a way to include information (F1score etc) outside of OWL class expression instances
            """
            try:
                assert isinstance(h, _NodeQuality)
                quality = h.quality
            except AttributeError:
                quality = None
            if isinstance(self.quality_func, Accuracy):
                accuracy = OWLAnnotationAssertionAxiom(cls_a.iri, OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(SNS, "accuracy")), OWLLiteral(quality)))
                manager.add_axiom(ontology, accuracy)
            elif isinstance(self.quality_func, F1):
                f1_score = OWLAnnotationAssertionAxiom(cls_a.iri, OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(SNS, "f1_score")), OWLLiteral(quality)))
                manager.add_axiom(ontology, f1_score)
            """

        manager.save_ontology(ontology, IRI.create('file:/' + path + '.owl'))

    def load_hypotheses(self, path: str) -> Iterable[OWLClassExpression]:
        """
        @TODO: CD: This function should be deprecated.
        @TODO: CD: Loading owl class expressions from disk should be disentangled from a concept earner


        Loads hypotheses (class expressions) from a file saved by :func:`BaseConceptLearner.save_best_hypothesis`.

        Args:
            path: Path to the file containing hypotheses.
        """
        manager: OWLOntologyManager_Owlready2 = OWLOntologyManager_Owlready2()
        ontology: OWLOntology_Owlready2 = manager.load_ontology(IRI.create('file://' + path))
        for c in ontology.classes_in_signature():
            for equivalent_classes in ontology.equivalent_classes_axioms(c):
                for equivalent_c in equivalent_classes.class_expressions():
                    if equivalent_c != c:
                        yield equivalent_c

    @staticmethod
    def verbalize(predictions_file_path: str):
        """
        @TODO:CD: this function should be removed from this class. This should be defined at best as a static func.

        """

        tree = ET.parse(predictions_file_path)
        root = tree.getroot()
        tmp_file = 'tmp_file_' + predictions_file_path
        owl = 'http://www.w3.org/2002/07/owl#'
        ontology_elem = root.find(f'{{{owl}}}Ontology')
        ontology_elem.remove(ontology_elem.find(f'{{{owl}}}imports'))

        # The commented lines below are needed if you want to use `verbaliser.verbalise_class_expression`
        # They assign labels to classes and properties.

        # rdf = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        # rdfs = 'http://www.w3.org/2000/01/rdf-schema#'
        # for element in root.iter():
        #     resource = None
        #     if f'{{{rdf}}}about' in element.attrib:
        #         resource = element.attrib[f'{{{rdf}}}about']
        #     elif f'{{{rdf}}}resource' in element.attrib:
        #         resource = element.attrib[f'{{{rdf}}}resource']
        #     if resource is not None:
        #         label = resource.split('#')
        #         if len(label) > 1:
        #             element.set(f'{{{rdfs}}}label', label[1])
        #         else:
        #             element.set(f'{{{rdfs}}}label', resource)

        tree.write(tmp_file)

        try:
            from deeponto.onto import Ontology, OntologyVerbaliser
            from anytree.dotexport import RenderTreeGraph
            from IPython.display import Image
        except Exception as e:
            print("You need to install deeponto to use this feature (pip install deeponto). If you have already, check "
                  "whether it's installed properly. \n   ----> Error: " + f'{e}')
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            return

        onto = Ontology(tmp_file)
        verbalizer = OntologyVerbaliser(onto)
        complex_concepts = onto.get_asserted_complex_classes()
        try:
            for i, ce in enumerate(complex_concepts):
                tree = verbalizer.parser.parse(str(ce))
                tree.render_image()
                os.rename("range_node.png", f"Prediction_{i}.png")
        except Exception as e:
            print("If you have not installed graphviz, please do so at https://graphviz.org/download/ to make the "
                  "verbalization possible. Otherwise check the error message: \n" + f'{e}')
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        if len(complex_concepts) == 0:
            print("No complex classes found!")
        elif len(complex_concepts) == 1:
            print("Image generated successfully!")
        else:
            print("Images generated successfully!")


class RefinementBasedConceptLearner(BaseConceptLearner[_N]):
    """
    Base class for refinement based Concept Learning approaches.

    Attributes:
        kb (KnowledgeBase): The knowledge base that the concept learner is using.
        quality_func (AbstractScorer) The quality function to be used.
        max_num_of_concepts_tested (int) Limit to stop the algorithm after n concepts tested.
        terminate_on_goal (bool): Whether to stop the algorithm if a perfect solution is found.
        max_runtime (int): Limit to stop the algorithm after n seconds.
        _number_of_tested_concepts (int): Yes, you got it. This stores the number of tested concepts.
        reasoner (OWLReasoner): The reasoner that this model is using.
        start_time (float): The time when :meth:`fit` starts the execution. Used to calculate the total time :meth:`fit`
                            takes to execute.
        iter_bound (int): Limit to stop the algorithm after n refinement steps are done.
        heuristic_func (AbstractHeuristic): Function to guide the search heuristic.
        operator (BaseRefinement): Operator used to generate refinements.
        start_class (OWLClassExpression): The starting class expression for the refinement operation.
        max_child_length (int): Limit the length of concepts generated by the refinement operator.


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
                 reasoner: Optional[OWLReasoner] = None,
                 refinement_operator: Optional[BaseRefinement] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_child_length: Optional[int] = None,
                 root_concept: Optional[OWLClassExpression] = None):
        """Create a new base concept learner.

        Args:
            knowledge_base: Knowledge base which is used to learn and test concepts. required, but can be taken
                from the learning problem if not specified.
            refinement_operator: Operator used to generate refinements. Defaults to `ModifiedCELOERefinement`.
            heuristic_func: Function to guide the search heuristic. Defaults to `CELOEHeuristic`.
            quality_func: Function to evaluate the quality of solution concepts. Defaults to `F1`.
            max_num_of_concepts_tested: Limit to stop the algorithm after n concepts tested. Defaults to 10_000.
            max_runtime: Limit to stop the algorithm after n seconds. Defaults to 5.
            terminate_on_goal: Whether to stop the algorithm if a perfect solution is found. Defaults to True.
            iter_bound: Limit to stop the algorithm after n refinement steps are done. Defaults to 10_000.
            max_child_length: Limit the length of concepts generated by the refinement operator. defaults to 10.
                Only used if refinement_operator is not specified.
            root_concept: The start concept to begin the search from. Defaults to OWL Thing.
        """
        super().__init__(knowledge_base=knowledge_base,
                         reasoner=reasoner,
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
            self.start_class = OWLThing
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
        Return from the search tree the most promising search tree node to use for the next refinement step.

        Returns:
            _N: Next search tree node to refine.
        """
        pass

    @abstractmethod
    def downward_refinement(self, *args, **kwargs):
        """Execute one refinement step of a refinement based learning algorithm.

        Args:
            node (_N): the search tree node on which to refine.

        Returns:
            Iterable[_N]: Refinement results as new search tree nodes (they still need to be added to the tree).
        """
        pass

    @abstractmethod
    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        """A debugging function to print out the current search tree and the current n best found hypotheses to
        standard output.

        Args:
            heading_step: A message to display at the beginning of the output.
            top_n: The number of current best hypotheses to print out.
        """
        pass
