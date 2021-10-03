from ontolearn.ea_algorithms import AbstractEvolutionaryAlgorithm, EASimple
from ontolearn.ea_initialization import AbstractEAInitialization, EARandomInitialization, EARandomWalkInitialization
from ontolearn.ea_utils import OperatorVocabulary, PrimitiveFactory
from ontolearn.search import EvoLearnerNode
from ontolearn.fitness_functions import LinearPressureFitness
from ontolearn.metrics import Accuracy
from ontolearn.learning_problem import EncodedPosNegLPStandard, PosNegLPStandard
from ontolearn.base_concept_learner import BaseConceptLearner
from ontolearn.abstracts import AbstractFitness, AbstractScorer
from typing import Callable, List, Optional, Set
from ontolearn.knowledge_base import KnowledgeBase
import operator

from deap import base, creator, tools, gp

from owlapy.model import OWLClass, OWLClassExpression, OWLNamedIndividual
import time


class EvoLearner(BaseConceptLearner[EvoLearnerNode]):

    __slots__ = 'fitness_func', 'init_method', 'algorithm', 'expressivity', 'tournament_size',  \
                'population_size', 'num_generations', 'height_limit', 'pset', 'toolbox', \
                '_learning_problem', '_result_population', 'mut_uniform_gen'

    name = 'evolearner'

    fitness_func: AbstractFitness
    init_method: AbstractEAInitialization
    algorithm: AbstractEvolutionaryAlgorithm
    mut_uniform_gen: AbstractEAInitialization
    expressivity: str
    tournament_size: int
    population_size: int
    num_generations: int
    height_limit: int

    pset: gp.PrimitiveSetTyped
    toolbox: base.Toolbox
    _learning_problem: EncodedPosNegLPStandard
    _result_population: Optional[List['creator.Individual']]

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 quality_func: Optional[AbstractScorer] = None,
                 fitness_func: Optional[AbstractFitness] = None,
                 init_method: Optional[AbstractEAInitialization] = None,
                 algorithm: Optional[AbstractEvolutionaryAlgorithm] = None,
                 mut_uniform_gen: Optional[AbstractEAInitialization] = None,
                 terminate_on_goal: Optional[bool] = None,
                 max_runtime: Optional[int] = None,
                 expressivity: str = 'ALC',
                 tournament_size: int = 7,
                 population_size: int = 800,
                 num_generations: int = 200,
                 height_limit: int = 17):

        if quality_func is None:
            quality_func = Accuracy()

        super().__init__(knowledge_base=knowledge_base,
                         quality_func=quality_func,
                         terminate_on_goal=terminate_on_goal,
                         max_runtime=max_runtime)

        self.fitness_func = fitness_func
        self.init_method = init_method
        self.algorithm = algorithm
        self.mut_uniform_gen = mut_uniform_gen
        self.expressivity = expressivity
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.num_generations = num_generations
        self.height_limit = height_limit

        self._result_population = None

        self.__setup()

    def __setup(self):
        if self.fitness_func is None:
            self.fitness_func = LinearPressureFitness()

        if self.init_method is None:
            self.init_method = EARandomWalkInitialization()

        if self.algorithm is None:
            self.algorithm = EASimple()

        if self.mut_uniform_gen is None:
            self.mut_uniform_gen = EARandomInitialization(min_height=1, max_height=3)

        self.pset = self.__build_primitive_set()
        self.toolbox = self.__build_toolbox()

    def __build_primitive_set(self) -> gp.PrimitiveSetTyped:
        ontology = self.kb.ontology()
        factory = PrimitiveFactory(self.kb)
        union = factory.create_union()
        intersection = factory.create_intersection()

        pset = gp.PrimitiveSetTyped("concept_tree", [], OWLClassExpression)
        pset.addPrimitive(self.kb.negation, [OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.NEGATION)
        pset.addPrimitive(union, [OWLClassExpression, OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.UNION)
        pset.addPrimitive(intersection, [OWLClassExpression, OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.INTERSECTION)

        for property_ in ontology.object_properties_in_signature():
            name = property_.get_iri().get_remainder()
            existential, universal = factory.create_existential_universal(property_)
            pset.addPrimitive(existential, [OWLClassExpression], OWLClassExpression,
                              name=OperatorVocabulary.EXISTENTIAL + name)
            pset.addPrimitive(universal, [OWLClassExpression], OWLClassExpression,
                              name=OperatorVocabulary.UNIVERSAL + name)

        for class_ in ontology.classes_in_signature():
            pset.addTerminal(class_, OWLClass, name=class_.get_iri().get_remainder())

        return pset

    def __build_toolbox(self) -> base.Toolbox:
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Quality", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, quality=creator.Quality)

        toolbox = base.Toolbox()
        toolbox.register("population", self.init_method.get_population, creator.Individual, self.pset)
        toolbox.register("compile", gp.compile, pset=self.pset)

        toolbox.register("apply_fitness", self._fitness_func)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("create_tree_mut", self.mut_uniform_gen.get_expression)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.create_tree_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                                max_value=self.height_limit))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                                  max_value=self.height_limit))

        toolbox.register("get_top_hypotheses", self._get_top_hypotheses)
        toolbox.register("terminate_on_goal", lambda: self.terminate_on_goal)
        toolbox.register("max_runtime", lambda: self.max_runtime)
        toolbox.register("pset", lambda: self.pset)

        return toolbox

    def register_op(self, alias: str, function: Callable, *args, **kargs):
        self.toolbox.register(alias, function, *args, **kargs)
        if alias == 'mate' or alias == 'mutate':
            self.toolbox.decorate(alias, gp.staticLimit(key=operator.attrgetter("height"),
                                                        max_value=self.height_limit))

    def fit(self, *args, **kwargs):
        """
        Find hypotheses that explain pos and neg.
        """
        self.clean()
        learning_problem = self.construct_learning_problem(PosNegLPStandard, args, kwargs)
        self._learning_problem = learning_problem.encode_kb(self.kb)

        verbose = kwargs.pop("verbose", 0)

        self.start_time = time.time()
        population = self._initialize(learning_problem.pos)
        self._goal_found, self._result_population = self.algorithm.evolve(self.toolbox,
                                                                          population,
                                                                          self.num_generations,
                                                                          self.start_time,
                                                                          verbose=verbose)

        return self.terminate()

    def _initialize(self, pos: Set[OWLNamedIndividual]):
        population = None
        if isinstance(self.init_method, EARandomWalkInitialization):
            population = self.toolbox.population(population_size=self.population_size, pos=list(pos), kb=self.kb)
        else:
            population = self.toolbox.population(population_size=self.population_size)
        return population

    def best_hypotheses(self, n=5, key='fitness'):
        assert self._result_population is not None
        assert len(self._result_population) > 0

        yield from self._get_top_hypotheses(self._result_population, n, key)

    def _get_top_hypotheses(self, population, n: int = 5, key: str = 'fitness'):
        best_inds = tools.selBest(population, k=n, fit_attr=key)
        best_concepts = [gp.compile(ind, self.pset) for ind in best_inds]

        for con, ind in zip(best_concepts, best_inds):
            individuals_count = len(self.kb.individuals_set(con))
            yield EvoLearnerNode(con, self.kb.cl(con), individuals_count, ind.quality.values[0],
                                 len(ind), ind.height)

    def _fitness_func(self, individual):
        concept = gp.compile(individual, self.pset)
        instances = self.kb.individuals_set(concept)
        quality = self.quality_func.score(instances, self._learning_problem)
        individual.quality.values = (quality[1],)
        self.fitness_func.apply(individual)

    def clean(self):
        self._result_population = None
        super().clean()
