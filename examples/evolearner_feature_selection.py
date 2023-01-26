import operator
import time
from itertools import chain
from typing import Any, Callable, Dict, FrozenSet, List, Tuple, Iterable, Optional
from deap import gp, tools, base, creator

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.abstracts import AbstractFitness, AbstractScorer
from ontolearn.base_concept_learner import BaseConceptLearner
from ontolearn.ea_algorithms import AbstractEvolutionaryAlgorithm, EASimple
from ontolearn.ea_initialization import AbstractEAInitialization, EARandomInitialization, EARandomWalkInitialization
from ontolearn.ea_utils import PrimitiveFactory, OperatorVocabulary, ToolboxVocabulary, Tree, escape, ind_to_string, \
    owlliteral_to_primitive_string
from ontolearn.fitness_functions import LinearPressureFitness
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.metrics import Accuracy
from ontolearn.search import EvoLearnerNode
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter, EntropyValueSplitter
from owlapy.model import OWLClassExpression, OWLDataProperty, OWLLiteral, OWLNamedIndividual
from ontolearn.concept_learner import EvoLearner


class EvoLearnerFeatureSelection(BaseConceptLearner[EvoLearnerNode]):

    __slots__ = 'fitness_func', 'init_method', 'algorithm', 'value_splitter', 'tournament_size',  \
                'population_size', 'num_generations', 'height_limit', 'use_data_properties', 'pset', 'toolbox', \
                '_learning_problem', '_result_population', 'mut_uniform_gen', '_dp_to_prim_type', '_dp_splits', \
                '_split_properties', '_cache', 'use_card_restrictions', 'card_limit', 'use_inverse', \
                'feature_obj_prop_name', 'feature_data_categorical_prop', 'feature_data_numeric_prop'

    name = 'evolearner'

    kb: KnowledgeBase
    fitness_func: AbstractFitness
    init_method: AbstractEAInitialization
    algorithm: AbstractEvolutionaryAlgorithm
    mut_uniform_gen: AbstractEAInitialization
    value_splitter: AbstractValueSplitter
    use_data_properties: bool
    use_card_restrictions: bool
    use_inverse: bool
    tournament_size: int
    card_limit: int
    population_size: int
    num_generations: int
    height_limit: int
    feature_obj_prop_name: list
    feature_data_categorical_prop: list
    feature_data_numeric_prop: list

    pset: gp.PrimitiveSetTyped
    toolbox: base.Toolbox
    _learning_problem: EncodedPosNegLPStandard
    _result_population: Optional[List[Tree]]
    _dp_to_prim_type: Dict[OWLDataProperty, Any]
    _dp_splits: Dict[OWLDataProperty, List[OWLLiteral]]
    _split_properties: List[OWLDataProperty]
    _cache: Dict[str, Tuple[float, float]]

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 quality_func: Optional[AbstractScorer] = None,
                 fitness_func: Optional[AbstractFitness] = None,
                 init_method: Optional[AbstractEAInitialization] = None,
                 algorithm: Optional[AbstractEvolutionaryAlgorithm] = None,
                 mut_uniform_gen: Optional[AbstractEAInitialization] = None,
                 value_splitter: Optional[AbstractValueSplitter] = None,
                 terminate_on_goal: Optional[bool] = None,
                 max_runtime: Optional[int] = None,
                 feature_obj_prop_name: Optional[list] = None,
                 feature_data_categorical_prop: Optional[list] = None,
                 feature_data_numeric_prop: Optional[list] = None,
                 use_data_properties: bool = True,
                 use_card_restrictions: bool = True,
                 use_inverse: bool = False,
                 tournament_size: int = 7,
                 card_limit: int = 10,
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
        self.value_splitter = value_splitter
        self.use_data_properties = use_data_properties
        self.use_card_restrictions = use_card_restrictions
        self.use_inverse = use_inverse
        self.tournament_size = tournament_size
        self.card_limit = card_limit
        self.population_size = population_size
        self.num_generations = num_generations
        self.height_limit = height_limit
        self.feature_obj_prop_name = feature_obj_prop_name
        self.feature_data_numeric_prop = feature_data_numeric_prop
        self.feature_data_categorical_prop = feature_data_categorical_prop

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

        if self.value_splitter is None:
            self.value_splitter = EntropyValueSplitter()

        self._result_population = None
        self._dp_to_prim_type = dict()
        self._dp_splits = dict()
        self._cache = dict()
        self._split_properties = []

        self.pset = self.__build_primitive_set()
        self.toolbox = self.__build_toolbox()

    def __build_primitive_set(self) -> gp.PrimitiveSetTyped:
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

        for op in self.kb.get_object_properties():
            name = escape(op.get_iri().get_remainder())
            if self.feature_obj_prop_name:
                if name in self.feature_obj_prop_name:
                    existential, universal = factory.create_existential_universal(op)
                    pset.addPrimitive(existential, [OWLClassExpression], OWLClassExpression,
                                    name=OperatorVocabulary.EXISTENTIAL + name)
                    pset.addPrimitive(universal, [OWLClassExpression], OWLClassExpression,
                                    name=OperatorVocabulary.UNIVERSAL + name)

                    if self.use_inverse:
                        existential, universal = factory.create_existential_universal(op.get_inverse_property())
                        pset.addPrimitive(existential, [OWLClassExpression], OWLClassExpression,
                                        name=OperatorVocabulary.INVERSE + OperatorVocabulary.EXISTENTIAL + name)
                        pset.addPrimitive(universal, [OWLClassExpression], OWLClassExpression,
                                        name=OperatorVocabulary.INVERSE + OperatorVocabulary.UNIVERSAL + name)
            else:
                existential, universal = factory.create_existential_universal(op)
                pset.addPrimitive(existential, [OWLClassExpression], OWLClassExpression,
                                name=OperatorVocabulary.EXISTENTIAL + name)
                pset.addPrimitive(universal, [OWLClassExpression], OWLClassExpression,
                                name=OperatorVocabulary.UNIVERSAL + name)

                if self.use_inverse:
                    existential, universal = factory.create_existential_universal(op.get_inverse_property())
                    pset.addPrimitive(existential, [OWLClassExpression], OWLClassExpression,
                                    name=OperatorVocabulary.INVERSE + OperatorVocabulary.EXISTENTIAL + name)
                    pset.addPrimitive(universal, [OWLClassExpression], OWLClassExpression,
                                    name=OperatorVocabulary.INVERSE + OperatorVocabulary.UNIVERSAL + name)

        if self.use_data_properties:
            class Bool(object):
                pass
            false_ = OWLLiteral(False)
            true_ = OWLLiteral(True)
            pset.addTerminal(false_, Bool, name=owlliteral_to_primitive_string(false_))
            pset.addTerminal(true_, Bool, name=owlliteral_to_primitive_string(true_))

            for bool_dp in self.kb.get_boolean_data_properties():
                name = escape(bool_dp.get_iri().get_remainder())
                if self.feature_data_categorical_prop:
                    if name in self.feature_data_categorical_prop:
                        self._dp_to_prim_type[bool_dp] = Bool

                        data_has_value = factory.create_data_has_value(bool_dp)
                        pset.addPrimitive(data_has_value, [Bool], OWLClassExpression,
                                        name=OperatorVocabulary.DATA_HAS_VALUE + name)
                else:
                    self._dp_to_prim_type[bool_dp] = Bool
                    data_has_value = factory.create_data_has_value(bool_dp)
                    pset.addPrimitive(data_has_value, [Bool], OWLClassExpression,
                                    name=OperatorVocabulary.DATA_HAS_VALUE + name)

            for split_dp in chain(self.kb.get_time_data_properties(), self.kb.get_numeric_data_properties()):
                name = escape(split_dp.get_iri().get_remainder())
                if self.feature_data_numeric_prop:
                    if name in self.feature_data_numeric_prop:
                        type_ = type(name, (object,), {})
                        self._dp_to_prim_type[split_dp] = type_
                        self._split_properties.append(split_dp)
                        min_inc, max_inc, _, _ = factory.create_data_some_values(split_dp)
                        pset.addPrimitive(min_inc, [type_], OWLClassExpression,
                                        name=OperatorVocabulary.DATA_MIN_INCLUSIVE + name)
                        pset.addPrimitive(max_inc, [type_], OWLClassExpression,
                                        name=OperatorVocabulary.DATA_MAX_INCLUSIVE + name)
                        # pset.addPrimitive(min_exc, [type_], OWLClassExpression,
                        #                  name=OperatorVocabulary.DATA_MIN_EXCLUSIVE + name)
                        # pset.addPrimitive(max_exc, [type_], OWLClassExpression,
                        #                  name=OperatorVocabulary.DATA_MAX_EXCLUSIVE + name)
                else:
                    type_ = type(name, (object,), {})
                    self._dp_to_prim_type[split_dp] = type_
                    self._split_properties.append(split_dp)
                    min_inc, max_inc, _, _ = factory.create_data_some_values(split_dp)
                    pset.addPrimitive(min_inc, [type_], OWLClassExpression,
                                    name=OperatorVocabulary.DATA_MIN_INCLUSIVE + name)
                    pset.addPrimitive(max_inc, [type_], OWLClassExpression,
                                    name=OperatorVocabulary.DATA_MAX_INCLUSIVE + name)

        if self.use_card_restrictions:
            for i in range(1, self.card_limit+1):
                pset.addTerminal(i, int)
            for op in self.kb.get_object_properties():
                name = escape(op.get_iri().get_remainder())
                if self.feature_obj_prop_name:
                    if name in self.feature_obj_prop_name:
                        card_min, card_max, _ = factory.create_card_restrictions(op)
                        pset.addPrimitive(card_min, [int, OWLClassExpression], OWLClassExpression,
                                        name=OperatorVocabulary.CARD_MIN + name)
                        pset.addPrimitive(card_max, [int, OWLClassExpression], OWLClassExpression,
                                        name=OperatorVocabulary.CARD_MAX + name)
                        # pset.addPrimitive(card_exact, [int, OWLClassExpression], OWLClassExpression,
                        #                  name=OperatorVocabulary.CARD_EXACT + name)
                else:
                        card_min, card_max, _ = factory.create_card_restrictions(op)
                        pset.addPrimitive(card_min, [int, OWLClassExpression], OWLClassExpression,
                                        name=OperatorVocabulary.CARD_MIN + name)
                        pset.addPrimitive(card_max, [int, OWLClassExpression], OWLClassExpression,
                                        name=OperatorVocabulary.CARD_MAX + name)


        for class_ in self.kb.get_concepts():
            pset.addTerminal(class_, OWLClassExpression, name=escape(class_.get_iri().get_remainder()))

        pset.addTerminal(self.kb.thing, OWLClassExpression, name=escape(self.kb.thing.get_iri().get_remainder()))
        pset.addTerminal(self.kb.nothing, OWLClassExpression, name=escape(self.kb.nothing.get_iri().get_remainder()))
        return pset

    def __build_toolbox(self) -> base.Toolbox:
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Quality", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, quality=creator.Quality)

        toolbox = base.Toolbox()
        toolbox.register(ToolboxVocabulary.INIT_POPULATION, self.init_method.get_population,
                         creator.Individual, self.pset)
        toolbox.register(ToolboxVocabulary.COMPILE, gp.compile, pset=self.pset)

        toolbox.register(ToolboxVocabulary.FITNESS_FUNCTION, self._fitness_func)
        toolbox.register(ToolboxVocabulary.SELECTION, tools.selTournament, tournsize=self.tournament_size)
        toolbox.register(ToolboxVocabulary.CROSSOVER, gp.cxOnePoint)
        toolbox.register("create_tree_mut", self.mut_uniform_gen.get_expression)
        toolbox.register(ToolboxVocabulary.MUTATION, gp.mutUniform, expr=toolbox.create_tree_mut, pset=self.pset)

        toolbox.decorate(ToolboxVocabulary.CROSSOVER,
                         gp.staticLimit(key=operator.attrgetter(ToolboxVocabulary.HEIGHT_KEY),
                                        max_value=self.height_limit))
        toolbox.decorate(ToolboxVocabulary.MUTATION,
                         gp.staticLimit(key=operator.attrgetter(ToolboxVocabulary.HEIGHT_KEY),
                                        max_value=self.height_limit))

        toolbox.register("get_top_hypotheses", self._get_top_hypotheses)
        toolbox.register("terminate_on_goal", lambda: self.terminate_on_goal)
        toolbox.register("max_runtime", lambda: self.max_runtime)
        toolbox.register("pset", lambda: self.pset)

        return toolbox

    def __set_splitting_values(self):
        for p in self._dp_splits:
            if len(self._dp_splits[p]) == 0:
                if p in self.kb.get_numeric_data_properties():
                    if self.feature_data_numeric_prop:
                        self._dp_splits[p].append(OWLLiteral(0))
                    else:
                        self._dp_splits[p].append(OWLLiteral(0))
                else:
                    pass  # TODO:

            # Remove terminal for multiple fits, unfortunately there exists no better way in DEAP
            # This removal is probably not needed, the important one is removal from the context below
            self.pset.terminals.pop(self._dp_to_prim_type[p], None)
            for split in self._dp_splits[p]:
                terminal_name = owlliteral_to_primitive_string(split, p)
                # Remove terminal for multiple fits, unfortunately there exists no better way in DEAP
                self.pset.context.pop(terminal_name, None)
                self.pset.addTerminal(split, self._dp_to_prim_type[p], name=terminal_name)

    def register_op(self, alias: str, function: Callable, *args, **kargs):
        self.toolbox.register(alias, function, *args, **kargs)
        if alias == ToolboxVocabulary.CROSSOVER or alias == ToolboxVocabulary.MUTATION:
            self.toolbox.decorate(alias, gp.staticLimit(key=operator.attrgetter(ToolboxVocabulary.HEIGHT_KEY),
                                                        max_value=self.height_limit))

    def fit(self, *args, **kwargs) -> 'EvoLearner':
        """
        Find hypotheses that explain pos and neg.
        """
        self.clean()
        learning_problem = self.construct_learning_problem(PosNegLPStandard, args, kwargs)
        self._learning_problem = learning_problem.encode_kb(self.kb)

        verbose = kwargs.pop("verbose", 0)

        population = self._initialize(learning_problem.pos, learning_problem.neg)
        self.start_time = time.time()
        self._goal_found, self._result_population = self.algorithm.evolve(self.toolbox,
                                                                          population,
                                                                          self.num_generations,
                                                                          self.start_time,
                                                                          verbose=verbose)
        return self.terminate()

    def _initialize(self, pos: FrozenSet[OWLNamedIndividual], neg: FrozenSet[OWLNamedIndividual]) -> List[Tree]:
        if self.use_data_properties:
            if isinstance(self.value_splitter, BinningValueSplitter):
                self._dp_splits = self.value_splitter.compute_splits_properties(self.kb.reasoner(),
                                                                                self._split_properties)
            elif isinstance(self.value_splitter, EntropyValueSplitter):
                entropy_splits = self.value_splitter.compute_splits_properties(self.kb.reasoner(),
                                                                               self._split_properties,
                                                                               pos, neg)
                no_splits = [prop for prop in entropy_splits if len(entropy_splits[prop]) == 0]
                temp_splitter = BinningValueSplitter(max_nr_splits=10)
                binning_splits = temp_splitter.compute_splits_properties(self.kb.reasoner(), no_splits)
                self._dp_splits = {**entropy_splits, **binning_splits}
            else:
                raise ValueError(self.value_splitter)
            self.__set_splitting_values()

        population = None
        if isinstance(self.init_method, EARandomWalkInitialization):
            population = self.toolbox.population(population_size=self.population_size, pos=list(pos),
                                                 kb=self.kb, dp_to_prim_type=self._dp_to_prim_type,
                                                 dp_splits=self._dp_splits)
        else:
            population = self.toolbox.population(population_size=self.population_size)
        return population

    def best_hypotheses(self, n: int = 5, key: str = 'fitness') -> Iterable[EvoLearnerNode]:
        assert self._result_population is not None
        assert len(self._result_population) > 0
        yield from self._get_top_hypotheses(self._result_population, n, key)

    def _get_top_hypotheses(self, population: List[Tree], n: int = 5, key: str = 'fitness') \
            -> Iterable[EvoLearnerNode]:
        best_inds = tools.selBest(population, k=n, fit_attr=key)
        best_concepts = [gp.compile(ind, self.pset) for ind in best_inds]

        for con, ind in zip(best_concepts, best_inds):
            individuals_count = len(self.kb.individuals_set(con))
            yield EvoLearnerNode(con, self.kb.concept_len(con), individuals_count, ind.quality.values[0],
                                 len(ind), ind.height)

    def _fitness_func(self, individual: Tree):
        ind_str = ind_to_string(individual)
        # experimental
        if ind_str in self._cache:
            individual.quality.values = (self._cache[ind_str][0],)
            individual.fitness.values = (self._cache[ind_str][1],)
        else:
            concept = gp.compile(individual, self.pset)
            e = self.kb.evaluate_concept(concept, self.quality_func, self._learning_problem)
            individual.quality.values = (e.q,)
            self.fitness_func.apply(individual)
            self._cache[ind_str] = (e.q, individual.fitness.values[0])
            self._number_of_tested_concepts += 1

    def clean(self):
        self._result_population = None

        # Resets classes if they already exist, names must match the ones that were created in the toolbox
        try:
            del creator.Fitness
            del creator.Individual
            del creator.Quality
        except AttributeError:
            pass

        super().clean()
