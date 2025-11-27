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

"""EvoLearner: An evolutionary approach to learn concepts in ALCQ(D)."""

import operator
import time
from itertools import chain
from typing import Any, Callable, Dict, FrozenSet, List, Tuple, Iterable, Optional, Union

from deap import gp, tools, base, creator
from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLDataProperty
from owlapy.abstracts import AbstractOWLReasoner

from ontolearn.concept_generator import ConceptGenerator
from ontolearn.abstracts import AbstractKnowledgeBase
from ontolearn.abstracts import AbstractFitness, AbstractScorer
from ontolearn.learners.base import BaseConceptLearner
from ontolearn.ea_algorithms import AbstractEvolutionaryAlgorithm, EASimple
from ontolearn.ea_initialization import AbstractEAInitialization, EARandomInitialization, EARandomWalkInitialization
from ontolearn.ea_utils import PrimitiveFactory, OperatorVocabulary, ToolboxVocabulary, Tree, escape, ind_to_string, \
    owlliteral_to_primitive_string
from ontolearn.fitness_functions import LinearPressureFitness
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.metrics import Accuracy
from ontolearn.utils.static_funcs import concept_len
from ontolearn.quality_funcs import evaluate_concept
from ontolearn.search import EvoLearnerNode
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter, EntropyValueSplitter


class EvoLearner(BaseConceptLearner):
    """An evolutionary approach to learn concepts in ALCQ(D).

    Attributes:
        algorithm (AbstractEvolutionaryAlgorithm): The evolutionary algorithm.
        card_limit (int): The upper cardinality limit if using cardinality restriction on object properties.
        fitness_func (AbstractFitness): Fitness function.
        height_limit (int): The maximum value allowed for the height of the Crossover and Mutation operations.
        init_method (AbstractEAInitialization): The evolutionary algorithm initialization method.
        kb (AbstractKnowledgeBase): The knowledge base that the concept learner is using.
        max_num_of_concepts_tested (int): Limit to stop the algorithm after n concepts tested.
        max_runtime (int): max_runtime: Limit to stop the algorithm after n seconds.
        mut_uniform_gen (AbstractEAInitialization): The initialization method to create the tree for mutation operation.
        name (str): Name of the model = 'evolearner'.
        num_generations (int): Number of generation for the evolutionary algorithm.
        _number_of_tested_concepts (int): Yes, you got it. This stores the number of tested concepts.
        population_size (int): Population size for the evolutionary algorithm.
        pset (gp.PrimitiveSetTyped): Contains the primitives that can be used to solve a Strongly Typed GP problem.
        quality_func: Function to evaluate the quality of solution concepts.
        reasoner (AbstractOWLReasoner): The reasoner that this model is using.
        start_time (float): The time when :meth:`fit` starts the execution. Used to calculate the total time :meth:`fit`
                            takes to execute.
        terminate_on_goal (bool): Whether to stop the algorithm if a perfect solution is found.
        toolbox (base.Toolbox): A toolbox for evolution that contains the evolutionary operators.
        tournament_size (int): The number of evolutionary individuals participating in each tournament.
        use_card_restrictions (bool): Use cardinality restriction for object properties?
        use_data_properties (bool): Consider data properties?
        use_inverse (bool): Consider inversed concepts?
        value_splitter (AbstractValueSplitter): Used to calculate the splits for data properties values.
    """

    __slots__ = 'fitness_func', 'init_method', 'algorithm', 'value_splitter', 'tournament_size', \
        'population_size', 'num_generations', 'height_limit', 'use_data_properties', 'pset', 'toolbox', \
        '_learning_problem', '_result_population', 'mut_uniform_gen', '_dp_to_prim_type', '_dp_splits', \
        '_split_properties', '_cache', 'use_card_restrictions', 'card_limit', 'use_inverse', 'total_fits', 'generator'

    name = 'evolearner'

    kb: AbstractKnowledgeBase
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
    generator: ConceptGenerator

    pset: gp.PrimitiveSetTyped
    toolbox: base.Toolbox
    _learning_problem: EncodedPosNegLPStandard
    _result_population: Optional[List[Tree]]
    _dp_to_prim_type: Dict[OWLDataProperty, Any]
    _dp_splits: Dict[OWLDataProperty, List[OWLLiteral]]
    _split_properties: List[OWLDataProperty]
    _cache: Dict[str, Tuple[float, float]]

    def __init__(self,
                 knowledge_base: AbstractKnowledgeBase,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 fitness_func: Optional[AbstractFitness] = None,
                 init_method: Optional[AbstractEAInitialization] = None,
                 algorithm: Optional[AbstractEvolutionaryAlgorithm] = None,
                 mut_uniform_gen: Optional[AbstractEAInitialization] = None,
                 value_splitter: Optional[AbstractValueSplitter] = None,
                 terminate_on_goal: Optional[bool] = None,
                 max_runtime: Optional[int] = None,
                 use_data_properties: bool = True,
                 use_card_restrictions: bool = True,
                 use_inverse: bool = False,
                 tournament_size: int = 7,
                 card_limit: int = 10,
                 population_size: int = 800,
                 num_generations: int = 200,
                 height_limit: int = 17):
        """ Create a new instance of EvoLearner

        Args:
            algorithm (AbstractEvolutionaryAlgorithm): The evolutionary algorithm. Defaults to `EASimple`.
            card_limit (int): The upper cardinality limit if using cardinality restriction for object properties.
                                Defaults to 10.
            fitness_func (AbstractFitness): Fitness function. Defaults to `LinearPressureFitness`.
            height_limit (int): The maximum value allowed for the height of the Crossover and Mutation operations.
                                Defaults to 17.
            init_method (AbstractEAInitialization): The evolutionary algorithm initialization method. Defaults
                                                    to EARandomWalkInitialization.
            knowledge_base (AbstractKnowledgeBase): The knowledge base that the concept learner is using.
            max_runtime (int): max_runtime: Limit to stop the algorithm after n seconds. Defaults to 5.
            mut_uniform_gen (AbstractEAInitialization): The initialization method to create the tree for mutation
                                                        operation. Defaults to
                                                        EARandomInitialization(min_height=1, max_height=3).
            num_generations (int): Number of generation for the evolutionary algorithm. Defaults to 200.
            population_size (int): Population size for the evolutionary algorithm. Defaults to 800.
            quality_func: Function to evaluate the quality of solution concepts. Defaults to `Accuracy`.
            reasoner (AbstractOWLReasoner): Optionally use a different reasoner. If reasoner=None, the reasoner of
                                    the :attr:`knowledge_base` is used.
            terminate_on_goal (bool): Whether to stop the algorithm if a perfect solution is found. Defaults to True.
            tournament_size (int): The number of evolutionary individuals participating in each tournament.
                                    Defaults to 7.
            use_card_restrictions (bool): Use cardinality restriction for object properties? Default to True.
            use_data_properties (bool): Consider data properties? Defaults to True.
            use_inverse (bool): Consider inversed concepts? Defaults to False.
            value_splitter (AbstractValueSplitter): Used to calculate the splits for data properties values. Defaults to
                                                `EntropyValueSplitter`.
        """

        if quality_func is None:
            quality_func = Accuracy()

        super().__init__(knowledge_base=knowledge_base,
                         reasoner=reasoner,
                         quality_func=quality_func,
                         terminate_on_goal=terminate_on_goal,
                         max_runtime=max_runtime)
        self.reasoner = reasoner
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
        self.total_fits = 0
        self.generator = ConceptGenerator()
        self.__setup()

    def __setup(self):
        self.clean(partial=True)
        self._cache = dict()
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
        self._split_properties = []

        self.pset = self.__build_primitive_set()
        self.toolbox = self.__build_toolbox()

    def __build_primitive_set(self) -> gp.PrimitiveSetTyped:
        factory = PrimitiveFactory()
        union = factory.create_union()
        intersection = factory.create_intersection()

        pset = gp.PrimitiveSetTyped("concept_tree", [], OWLClassExpression)
        pset.addPrimitive(self.generator.negation, [OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.NEGATION.value)
        pset.addPrimitive(union, [OWLClassExpression, OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.UNION.value)
        pset.addPrimitive(intersection, [OWLClassExpression, OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.INTERSECTION.value)

        for op in self.kb.get_object_properties():
            name = escape(op.iri.get_remainder())
            existential, universal = factory.create_existential_universal(op)
            pset.addPrimitive(existential, [OWLClassExpression], OWLClassExpression,
                              name=OperatorVocabulary.EXISTENTIAL + name)
            pset.addPrimitive(universal, [OWLClassExpression], OWLClassExpression,
                              name=OperatorVocabulary.UNIVERSAL + name)

            if self.use_inverse:  # pragma: no cover
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
                name = escape(bool_dp.iri.get_remainder())
                self._dp_to_prim_type[bool_dp] = Bool

                data_has_value = factory.create_data_has_value(bool_dp)
                pset.addPrimitive(data_has_value, [Bool], OWLClassExpression,
                                  name=OperatorVocabulary.DATA_HAS_VALUE + name)

            for split_dp in chain(self.kb.get_time_data_properties(), self.kb.get_numeric_data_properties()):
                name = escape(split_dp.iri.get_remainder())
                type_ = type(name, (object,), {})

                self._dp_to_prim_type[split_dp] = type_
                self._split_properties.append(split_dp)

                min_inc, max_inc, _, _ = factory.create_data_some_values(split_dp)
                pset.addPrimitive(min_inc, [type_], OWLClassExpression,
                                  name=OperatorVocabulary.DATA_MIN_INCLUSIVE + name)
                pset.addPrimitive(max_inc, [type_], OWLClassExpression,
                                  name=OperatorVocabulary.DATA_MAX_INCLUSIVE + name)

        if self.use_card_restrictions:
            for i in range(1, self.card_limit + 1):
                pset.addTerminal(i, int)
            for op in self.kb.get_object_properties():
                name = escape(op.iri.get_remainder())
                card_min, card_max, _ = factory.create_card_restrictions(op)
                pset.addPrimitive(card_min, [int, OWLClassExpression], OWLClassExpression,
                                  name=OperatorVocabulary.CARD_MIN + name)
                pset.addPrimitive(card_max, [int, OWLClassExpression], OWLClassExpression,
                                  name=OperatorVocabulary.CARD_MAX + name)

        for class_ in self.kb.get_concepts():
            if class_ == OWLThing:
                continue
            pset.addTerminal(class_, OWLClassExpression, name=escape(class_.iri.get_remainder()))

        pset.addTerminal(self.generator.thing, OWLClassExpression,
                         name=escape(self.generator.thing.iri.get_remainder()))
        pset.addTerminal(self.generator.nothing, OWLClassExpression,
                         name=escape(self.generator.nothing.iri.get_remainder()))
        return pset

    def __build_toolbox(self) -> base.Toolbox:
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Quality", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, quality=creator.Quality)

        toolbox = base.Toolbox()
        toolbox.register(ToolboxVocabulary.INIT_POPULATION.value, self.init_method.get_population,
                         creator.Individual, self.pset)
        toolbox.register(ToolboxVocabulary.COMPILE.value, gp.compile, pset=self.pset)

        toolbox.register(ToolboxVocabulary.FITNESS_FUNCTION.value, self._fitness_func)
        toolbox.register(ToolboxVocabulary.SELECTION.value, tools.selTournament, tournsize=self.tournament_size)
        toolbox.register(ToolboxVocabulary.CROSSOVER.value, gp.cxOnePoint)
        toolbox.register("create_tree_mut", self.mut_uniform_gen.get_expression)
        toolbox.register(ToolboxVocabulary.MUTATION.value, gp.mutUniform, expr=toolbox.create_tree_mut, pset=self.pset)

        toolbox.decorate(ToolboxVocabulary.CROSSOVER.value,
                         gp.staticLimit(key=operator.attrgetter(ToolboxVocabulary.HEIGHT_KEY),
                                        max_value=self.height_limit))
        toolbox.decorate(ToolboxVocabulary.MUTATION.value,
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

    def register_op(self, alias: str, function: Callable, *args, **kargs):  # pragma: no cover
        """Register a *function* in the toolbox under the name *alias*.
        You may provide default arguments that will be passed automatically when
        calling the registered function. Fixed arguments can then be overriden
        at function call time.

        Args:
            alias: The name the operator will take in the toolbox. If the
                    alias already exist it will overwrite the operator
                    already present.
            function: The function to which refer the alias.
            args: One or more argument (and keyword argument) to pass
                     automatically to the registered function when called,
                     optional.
        """
        self.toolbox.register(alias, function, *args, **kargs)
        if alias == ToolboxVocabulary.CROSSOVER or alias == ToolboxVocabulary.MUTATION:
            self.toolbox.decorate(alias, gp.staticLimit(key=operator.attrgetter(ToolboxVocabulary.HEIGHT_KEY),
                                                        max_value=self.height_limit))

    def fit(self, *args, **kwargs) -> 'EvoLearner':
        """
        Find hypotheses that explain pos and neg.
        """
        # Don't reset everything if the user is just using this model for 1 learning problem, since he may use the
        # register_op method, else-wise we need to `clean` before fitting to get a fresh fit.
        if self.total_fits > 0:
            self.clean()
        self.total_fits += 1
        learning_problem = self.construct_learning_problem(PosNegLPStandard, args, kwargs)
        self._learning_problem = learning_problem.encode_kb(self.kb)

        verbose = kwargs.pop("verbose", False)

        population = self._initialize(learning_problem.pos, learning_problem.neg)
        self.start_time = time.time()
        self._goal_found, self._result_population = self.algorithm.evolve(self.toolbox,
                                                                          population,
                                                                          self.num_generations,
                                                                          self.start_time,
                                                                          verbose=verbose)
        return self.terminate()

    def _initialize(self, pos: FrozenSet[OWLNamedIndividual], neg: FrozenSet[OWLNamedIndividual]) -> List[Tree]:
        reasoner = self.kb.reasoner if self.reasoner is None else self.reasoner
        if self.use_data_properties:
            if isinstance(self.value_splitter, BinningValueSplitter):
                self._dp_splits = self.value_splitter.compute_splits_properties(reasoner,
                                                                                self._split_properties)
            elif isinstance(self.value_splitter, EntropyValueSplitter):
                entropy_splits = self.value_splitter.compute_splits_properties(reasoner,
                                                                               self._split_properties,
                                                                               pos, neg)
                no_splits = [prop for prop in entropy_splits if len(entropy_splits[prop]) == 0]
                temp_splitter = BinningValueSplitter(max_nr_splits=10)
                binning_splits = temp_splitter.compute_splits_properties(reasoner, no_splits)
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

    def best_hypotheses(self, n: int = 1, key: str = 'fitness', return_node: bool = False) -> Union[OWLClassExpression,
    Iterable[OWLClassExpression]]:
        assert self._result_population is not None
        assert len(self._result_population) > 0
        if n > 1:
            if return_node:
                return [i for i in self._get_top_hypotheses(self._result_population, n, key)]

            else:
                return [i.concept for i in self._get_top_hypotheses(self._result_population, n, key)]
        else:
            if return_node:
                return next(self._get_top_hypotheses(self._result_population, n, key))
            else:
                return next(self._get_top_hypotheses(self._result_population, n, key)).concept

    def _get_top_hypotheses(self, population: List[Tree], n: int = 5, key: str = 'fitness') \
            -> Iterable[EvoLearnerNode]:
        best_inds = tools.selBest(population, k=n * 10, fit_attr=key)
        best_inds_distinct = []
        for ind in best_inds:
            if ind not in best_inds_distinct:
                best_inds_distinct.append(ind)
        best_concepts = [gp.compile(ind, self.pset) for ind in best_inds_distinct[:n]]

        for con, ind in zip(best_concepts, best_inds):
            individuals_count = len(self.kb.individuals_set(con))
            yield EvoLearnerNode(con, concept_len(con), individuals_count, ind.quality.values[0],
                                 len(ind), ind.height)

    def _fitness_func(self, individual: Tree):
        ind_str = ind_to_string(individual)
        # experimental
        if ind_str in self._cache:
            individual.quality.values = (self._cache[ind_str][0],)
            individual.fitness.values = (self._cache[ind_str][1],)
        else:
            concept = gp.compile(individual, self.pset)
            e = evaluate_concept(self.kb, concept, self.quality_func, self._learning_problem)
            individual.quality.values = (e.q,)
            self.fitness_func.apply(individual)
            self._cache[ind_str] = (e.q, individual.fitness.values[0])
            self._number_of_tested_concepts += 1

    def clean(self, partial: bool = False):
        # Resets classes if they already exist, names must match the ones that were created in the toolbox
        try:
            del creator.Fitness
            del creator.Individual
            del creator.Quality
        except AttributeError:
            pass
        super().clean()
        if not partial:
            # Reset everything if fitting more than one lp. Tests have shown that this is necessary to get the
            # best performance of EvoLearner.
            self._result_population = None
            self._cache.clear()
            self.fitness_func = LinearPressureFitness()
            self.init_method = EARandomWalkInitialization()
            self.algorithm = EASimple()
            self.mut_uniform_gen = EARandomInitialization(min_height=1, max_height=3)
            self.value_splitter = EntropyValueSplitter()
            self._dp_to_prim_type = dict()
            self._dp_splits = dict()
            self._split_properties = []
            self.pset = self.__build_primitive_set()
            self.toolbox = self.__build_toolbox()
