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

"""Concept learning algorithms of Ontolearn."""

import logging
import operator
import time
from datetime import datetime
from contextlib import contextmanager
from itertools import islice, chain
from typing import Any, Callable, Dict, FrozenSet, Set, List, Tuple, Iterable, Optional, Union

import pandas as pd
import numpy as np
import torch
from owlapy.class_expression import OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLDataProperty
from owlapy.abstracts import AbstractOWLReasoner
from torch.utils.data import DataLoader
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from deap import gp, tools, base, creator

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.abstracts import AbstractFitness, AbstractScorer, BaseRefinement, \
    AbstractHeuristic, EncodedPosNegLPStandardKind
from ontolearn.base_concept_learner import BaseConceptLearner, RefinementBasedConceptLearner
from owlapy.utils import EvaluatedDescriptionSet, ConceptOperandSorter, OperandSetTransform
from ontolearn.data_struct import NCESDataLoader, NCESDataLoaderInference, CLIPDataLoader, CLIPDataLoaderInference
from ontolearn.ea_algorithms import AbstractEvolutionaryAlgorithm, EASimple
from ontolearn.ea_initialization import AbstractEAInitialization, EARandomInitialization, EARandomWalkInitialization
from ontolearn.ea_utils import PrimitiveFactory, OperatorVocabulary, ToolboxVocabulary, Tree, escape, ind_to_string, \
    owlliteral_to_primitive_string
from ontolearn.fitness_functions import LinearPressureFitness
from ontolearn.heuristics import OCELHeuristic
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.metrics import Accuracy
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.search import EvoLearnerNode, NCESNode, HeuristicOrderedNode, LBLNode, OENode, TreeNode, \
    LengthOrderedNode, \
    QualityOrderedNode, EvaluatedConcept
from ontolearn.utils import oplogging
from ontolearn.utils.static_funcs import init_length_metric, compute_tp_fn_fp_tn
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter, EntropyValueSplitter
from ontolearn.base_nces import BaseNCES
from ontolearn.nces_architectures import LSTM, GRU, SetTransformer
from ontolearn.clip_architectures import LengthLearner_LSTM, LengthLearner_GRU, LengthLearner_CNN, \
    LengthLearner_SetTransformer
from .utils import read_csv
from ontolearn.nces_trainer import NCESTrainer, before_pad
from ontolearn.clip_trainer import CLIPTrainer
from ontolearn.nces_utils import SimpleSolution, Data
from ontolearn.nces_modules import ConEx
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.parser import DLSyntaxParser
from owlapy.utils import OrderedOWLObject
from sortedcontainers import SortedSet
import os
import json
import glob
from ontolearn.lp_generator import LPGen
from .learners import CELOE

_concept_operand_sorter = ConceptOperandSorter()

class EvoLearner(BaseConceptLearner):
    """An evolutionary approach to learn concepts in ALCQ(D).

    Attributes:
        algorithm (AbstractEvolutionaryAlgorithm): The evolutionary algorithm.
        card_limit (int): The upper cardinality limit if using cardinality restriction on object properties.
        fitness_func (AbstractFitness): Fitness function.
        height_limit (int): The maximum value allowed for the height of the Crossover and Mutation operations.
        init_method (AbstractEAInitialization): The evolutionary algorithm initialization method.
        kb (KnowledgeBase): The knowledge base that the concept learner is using.
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
        '_split_properties', '_cache', 'use_card_restrictions', 'card_limit', 'use_inverse', 'total_fits'

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
            card_limit (int): The upper cardinality limit if using cardinality restriction for object properties. Defaults to 10.
            fitness_func (AbstractFitness): Fitness function. Defaults to `LinearPressureFitness`.
            height_limit (int): The maximum value allowed for the height of the Crossover and Mutation operations.
                                Defaults to 17.
            init_method (AbstractEAInitialization): The evolutionary algorithm initialization method. Defaults
                                                    to EARandomWalkInitialization.
            knowledge_base (KnowledgeBase): The knowledge base that the concept learner is using.
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
        pset.addPrimitive(self.kb.generator.negation, [OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.NEGATION)
        pset.addPrimitive(union, [OWLClassExpression, OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.UNION)
        pset.addPrimitive(intersection, [OWLClassExpression, OWLClassExpression], OWLClassExpression,
                          name=OperatorVocabulary.INTERSECTION)

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
                # pset.addPrimitive(min_exc, [type_], OWLClassExpression,
                #                  name=OperatorVocabulary.DATA_MIN_EXCLUSIVE + name)
                # pset.addPrimitive(max_exc, [type_], OWLClassExpression,
                #                  name=OperatorVocabulary.DATA_MAX_EXCLUSIVE + name)

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
                # pset.addPrimitive(card_exact, [int, OWLClassExpression], OWLClassExpression,
                #                  name=OperatorVocabulary.CARD_EXACT + name)

        for class_ in self.kb.get_concepts():
            pset.addTerminal(class_, OWLClassExpression, name=escape(class_.iri.get_remainder()))

        pset.addTerminal(self.kb.generator.thing, OWLClassExpression,
                         name=escape(self.kb.generator.thing.iri.get_remainder()))
        pset.addTerminal(self.kb.generator.nothing, OWLClassExpression,
                         name=escape(self.kb.generator.nothing.iri.get_remainder()))
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


class CLIP(CELOE):
    """Concept Learner with Integrated Length Prediction.
    This algorithm extends the CELOE algorithm by using concept length predictors and a different refinement operator, i.e., ExpressRefinement

    Attributes:
        best_descriptions (EvaluatedDescriptionSet[OENode, QualityOrderedNode]): Best hypotheses ordered.
        best_only (bool): If False pick only nodes with quality < 1.0, else pick without quality restrictions.
        calculate_min_max (bool): Calculate minimum and maximum horizontal expansion? Statistical purpose only.
        heuristic_func (AbstractHeuristic): Function to guide the search heuristic.
        heuristic_queue (SortedSet[OENode]): A sorted set that compares the nodes based on Heuristic.
        iter_bound (int): Limit to stop the algorithm after n refinement steps are done.
        kb (KnowledgeBase): The knowledge base that the concept learner is using.
        max_child_length (int): Limit the length of concepts generated by the refinement operator.
        max_he (int): Maximal value of horizontal expansion.
        max_num_of_concepts_tested (int) Limit to stop the algorithm after n concepts tested.
        max_runtime (int): Limit to stop the algorithm after n seconds.
        min_he (int): Minimal value of horizontal expansion.
        name (str): Name of the model = 'celoe_python'.
        _number_of_tested_concepts (int): Yes, you got it. This stores the number of tested concepts.
        operator (BaseRefinement): Operator used to generate refinements.
        quality_func (AbstractScorer) The quality function to be used.
        reasoner (AbstractOWLReasoner): The reasoner that this model is using.
        search_tree (Dict[OWLClassExpression, TreeNode[OENode]]): Dict to store the TreeNode for a class expression.
        start_class (OWLClassExpression): The starting class expression for the refinement operation.
        start_time (float): The time when :meth:`fit` starts the execution. Used to calculate the total time :meth:`fit`
                            takes to execute.
        terminate_on_goal (bool): Whether to stop the algorithm if a perfect solution is found.

    """
    __slots__ = 'best_descriptions', 'max_he', 'min_he', 'best_only', 'calculate_min_max', 'heuristic_queue', \
        'search_tree', '_learning_problem', '_max_runtime', '_seen_norm_concepts', 'predictor_name', 'pretrained_predictor_name', \
        'load_pretrained', 'output_size', 'num_examples', 'path_of_embeddings', 'instance_embeddings', 'input_size', 'device', 'length_predictor', \
        'num_workers', 'knowledge_base_path'

    name = 'clip'

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 knowledge_base_path='',
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 refinement_operator: Optional[BaseRefinement[OENode]] = ExpressRefinement,
                 quality_func: Optional[AbstractScorer] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 max_results: int = 10,
                 best_only: bool = False,
                 calculate_min_max: bool = True,
                 path_of_embeddings="",
                 predictor_name=None,
                 pretrained_predictor_name=["SetTransformer", "LSTM", "GRU", "CNN"],
                 load_pretrained=False,
                 num_workers=4,
                 num_examples=1000,
                 output_size=15
                 ):
        super().__init__(knowledge_base,
                         reasoner,
                         refinement_operator,
                         quality_func,
                         heuristic_func,
                         terminate_on_goal,
                         iter_bound,
                         max_num_of_concepts_tested,
                         max_runtime,
                         max_results,
                         best_only,
                         calculate_min_max)
        self.predictor_name = predictor_name
        self.pretrained_predictor_name = pretrained_predictor_name
        self.knowledge_base_path = knowledge_base_path
        self.load_pretrained = load_pretrained
        self.num_workers = num_workers
        self.output_size = output_size
        self.num_examples = num_examples
        self.path_of_embeddings = path_of_embeddings
        assert os.path.isfile(self.path_of_embeddings), '!!! Wrong path for CLIP embeddings'
        self.instance_embeddings = pd.read_csv(path_of_embeddings, index_col=0)
        self.input_size = self.instance_embeddings.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.length_predictor = self.get_length_predictor()

    def get_length_predictor(self):
        def load_model(predictor_name, load_pretrained):
            if predictor_name is None:
                return []
            if predictor_name == 'SetTransformer':
                model = LengthLearner_SetTransformer(self.input_size, self.output_size, proj_dim=256, num_heads=4,
                                                     num_seeds=1, m=32)
            elif predictor_name == 'GRU':
                model = LengthLearner_GRU(self.input_size, self.output_size, proj_dim=256, rnn_n_layers=2,
                                          drop_prob=0.2)
            elif predictor_name == 'LSTM':
                model = LengthLearner_LSTM(self.input_size, self.output_size, proj_dim=256, rnn_n_layers=2,
                                           drop_prob=0.2)
            elif predictor_name == 'CNN':
                model = LengthLearner_CNN(self.input_size, self.output_size, self.num_examples, proj_dim=256,
                                          kernel_size=[[5, 7], [5, 7]], stride=[[3, 3], [3, 3]])
            pretrained_model_path = self.path_of_embeddings.split("embeddings")[
                                        0] + "trained_models/trained_" + predictor_name + ".pt"
            if load_pretrained and os.path.isfile(pretrained_model_path):
                model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device, weights_only=True))
                model.eval()
                print("\n Loaded length predictor!")
            return model

        if not self.load_pretrained:
            return [load_model(self.predictor_name, self.load_pretrained)]
        elif self.load_pretrained and isinstance(self.pretrained_predictor_name, str):
            return [load_model(self.pretrained_predictor_name, self.load_pretrained)]
        elif self.load_pretrained and isinstance(self.pretrained_predictor_name, list):
            return [load_model(name, self.load_pretrained) for name in self.pretrained_predictor_name]

    def refresh(self):
        self.length_predictor = self.get_length_predictor()

    def collate_batch(self, batch):  # pragma: no cover
        pos_emb_list = []
        neg_emb_list = []
        target_labels = []
        for pos_emb, neg_emb, label in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
            target_labels.append(label)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        return pos_emb_list, neg_emb_list, torch.LongTensor(target_labels)

    def collate_batch_inference(self, batch):  # pragma: no cover
        pos_emb_list = []
        neg_emb_list = []
        for pos_emb, neg_emb in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        return pos_emb_list, neg_emb_list

    def pos_neg_to_tensor(self, pos: Union[Set[OWLNamedIndividual]], neg: Union[Set[OWLNamedIndividual], Set[str]]):
        if isinstance(pos[0], OWLNamedIndividual):
            pos_str = [ind.str.split("/")[-1] for ind in pos][:self.num_examples]
            neg_str = [ind.str.split("/")[-1] for ind in neg][:self.num_examples]
        elif isinstance(pos[0], str):
            pos_str = pos[:self.num_examples]
            neg_str = neg[:self.num_examples]
        else:
            raise ValueError(f"Invalid input type, was expecting OWLNamedIndividual or str but found {type(pos[0])}")

        assert self.load_pretrained and self.pretrained_predictor_name, \
            "No pretrained model found. Please first train length predictors, see the <<train>> method below"

        dataset = CLIPDataLoaderInference([("", pos_str, neg_str)], self.instance_embeddings, False, False)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers,
                                collate_fn=self.collate_batch_inference, shuffle=False)
        x_pos, x_neg = next(iter(dataloader))
        return x_pos, x_neg

    def predict_length(self, models, x1, x2):
        for i, model in enumerate(models):
            model.eval()
            model.to(self.device)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            if i == 0:
                scores = model(x1, x2)
            else:
                sc = model(x1, x2)
                scores = scores + sc
        scores = scores / len(models)
        prediction = int(scores.argmax(1).cpu())
        print(f"\n***** Predicted length: {prediction} *****\n")
        return prediction

    def fit(self, *args, **kwargs):
        """
        Find hypotheses that explain pos and neg.
        """
        self.clean()
        max_runtime = kwargs.pop("max_runtime", None)
        learning_problem = self.construct_learning_problem(PosNegLPStandard, args, kwargs)

        assert not self.search_tree
        self._learning_problem = learning_problem.encode_kb(self.kb)

        if max_runtime is not None:
            self._max_runtime = max_runtime
        else:
            self._max_runtime = self.max_runtime

        if (self.pretrained_predictor_name is not None) and (self.length_predictor is not None):
            x_pos, x_neg = self.pos_neg_to_tensor(list(self._learning_problem.kb_pos)[:self.num_examples],
                                                  list(self._learning_problem.kb_neg)[:self.num_examples])
            max_length = self.predict_length(self.length_predictor, x_pos, x_neg)
            self.operator.max_child_length = max_length
            print(f'***** Predicted length: {max_length} *****')
        else:
            print('\n!!! No length predictor provided, running CLIP without length predictor !!!')

        root = self.make_node(_concept_operand_sorter.sort(self.start_class), is_root=True)
        self._add_node(root, None)
        assert len(self.heuristic_queue) == 1
        # TODO:CD:suggest to add another assert,e.g. assert #. of instance in root > 1

        self.start_time = time.time()
        for j in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(j)
            tree_parent = self.tree_node(most_promising)
            minimum_length = most_promising.h_exp
            # if logger.isEnabledFor(oplogging.TRACE):
            # logger.debug("now refining %s", most_promising)
            for ref in self.downward_refinement(most_promising):
                # we ignore all refinements with lower length
                # (this also avoids duplicate node children)
                # TODO: ignore too high depth
                if ref.len < minimum_length:
                    # ignoring refinement, it does not satisfy minimum_length condition
                    continue

                # note: tree_parent has to be equal to node_tree_parent(ref.parent_node)!
                added = self._add_node(ref, tree_parent)

                goal_found = added and ref.quality == 1.0

                if goal_found and self.terminate_on_goal:
                    return self.terminate()

            if self.calculate_min_max:
                # This is purely a statistical function, it does not influence CELOE
                self.update_min_max_horiz_exp(most_promising)

            if time.time() - self.start_time > self._max_runtime:
                return self.terminate()

            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()

            # if logger.isEnabledFor(oplogging.TRACE) and j % 100 == 0:
            #    self._log_current_best(j)

        return self.terminate()

    def train(self, data: Iterable[List[Tuple]], epochs=300, batch_size=256, learning_rate=1e-3, decay_rate=0.0,
              clip_value=5.0, save_model=True, storage_path=None, optimizer='Adam', record_runtime=True,
              example_sizes=None, shuffle_examples=False):
        train_dataset = CLIPDataLoader(data, self.instance_embeddings, shuffle_examples=shuffle_examples,
                                       example_sizes=example_sizes)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=self.num_workers,
                                      collate_fn=self.collate_batch, shuffle=True)
        if storage_path is None:
            storage_path = self.knowledge_base_path[:self.knowledge_base_path.rfind("/")]
        elif not os.path.exists(storage_path) and (record_runtime or save_model):
            os.mkdir(storage_path)
        trainer = CLIPTrainer(self, epochs=epochs, learning_rate=learning_rate, decay_rate=decay_rate,
                              clip_value=clip_value, storage_path=storage_path)
        trainer.train(train_dataloader, save_model, optimizer, record_runtime)


class NCES(BaseNCES):
    """Neural Class Expression Synthesis."""

    def __init__(self, knowledge_base_path,
                 quality_func: Optional[AbstractScorer] = None, num_predictions=5,
                 learner_names=["SetTransformer"], path_of_embeddings="", proj_dim=128, rnn_n_layers=2, drop_prob=0.1,
                 num_heads=4, num_seeds=1, m=32, ln=False, learning_rate=1e-4, decay_rate=0.0, clip_value=5.0,
                 batch_size=256, num_workers=4, max_length=48, load_pretrained=True, sorted_examples=False, verbose: int = 0):
        super().__init__(knowledge_base_path, quality_func, num_predictions, proj_dim, drop_prob,
                 num_heads, num_seeds, m, ln, learning_rate, decay_rate, clip_value,
                 batch_size, num_workers, max_length, load_pretrained, sorted_examples, verbose)
        
        self.learner_names = learner_names
        self.path_of_embeddings = path_of_embeddings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnn_n_layers = rnn_n_layers
        self.instance_embeddings = read_csv(path_of_embeddings)
        self.input_size = self.instance_embeddings.shape[1]
        self.model = self.get_synthesizer()
        

    def get_synthesizer(self, path=None):
        m1 = SetTransformer(self.knowledge_base_path, self.vocab, self.inv_vocab, self.max_length,
                                   self.input_size, self.proj_dim, self.num_heads, self.num_seeds, self.m,
                                   self.ln)
        m2 = GRU(self.knowledge_base_path, self.vocab, self.inv_vocab, self.max_length, self.input_size,
                        self.proj_dim, self.rnn_n_layers, self.drop_prob)

        m3 = LSTM(self.knowledge_base_path, self.vocab, self.inv_vocab, self.max_length, self.input_size,
                         self.proj_dim, self.rnn_n_layers, self.drop_prob)
        Untrained = []
        for name in self.learner_names:
            for m in [m1,m2,m3]:
                if m.name == name:
                    Untrained.append(m)

        Models = []

        if self.load_pretrained:
            if path is None:
                try:
                    if len(glob.glob(self.path_of_embeddings.split("embeddings")[0] + "trained_models/*.pt")) == 0:
                        raise FileNotFoundError
                    else:
                        for file_name in glob.glob(self.path_of_embeddings.split("embeddings")[0] + "trained_models/*.pt"):
                            for m in Untrained:
                                if m.name in file_name:
                                    try:
                                        m.load_state_dict(torch.load(file_name, map_location=self.device, weights_only=True))
                                        Models.append(m.eval())
                                    except Exception as e:
                                        print(e)
                                        pass
                except Exception as e:
                    print(e)
                    raise RuntimeError

                if Models:
                    print("\n Loaded NCES weights!\n")
                    return Models
                else:
                    print("!!!Returning untrained models, could not load pretrained")
                    return Untrained

            elif len(glob.glob(path+"/*.pt")) == 0:
                 print("No pretrained model found! If directory is empty or does not exist, set the NCES `load_pretrained` parameter to `False` or make sure `save_model` was set to `True` in the .train() method.")
                raise FileNotFoundError
            else:
                for file_name in glob.glob(path+"/*.pt"):
                    for m in Untrained:
                        if m.name in file_name:
                            try:
                                m.load_state_dict(torch.load(file_name, map_location=self.device, weights_only=True))
                                Models.append(m.eval())
                            except Exception as e:
                                print(e)
                                pass
                if Models:
                    print("\n Loaded NCES weights!\n")
                    return Models
                else:
                    print("!!!Returning untrained models, could not load pretrained")
                    return Untrained
        else:
            print("!!!Returning untrained models, could not load pretrained. Check the `load_pretrained parameter` or train the models using NCES.train(data).")
            return Untrained


    def refresh(self, path=None):
        if path is not None:
            self.load_pretrained = True
        self.model = self.get_synthesizer(path)

    def sample_examples(self, pos, neg):  # pragma: no cover
        assert type(pos[0]) == type(neg[0]), "The two iterables pos and neg must be of same type"
        num_ex = self.num_examples
        if min(len(pos), len(neg)) >= num_ex // 2:
            if len(pos) > len(neg):
                num_neg_ex = num_ex // 2
                num_pos_ex = num_ex - num_neg_ex
            else:
                num_pos_ex = num_ex // 2
                num_neg_ex = num_ex - num_pos_ex
        elif len(pos) + len(neg) >= num_ex and len(pos) > len(neg):
            num_neg_ex = len(neg)
            num_pos_ex = num_ex - num_neg_ex
        elif len(pos) + len(neg) >= num_ex and len(pos) < len(neg):
            num_pos_ex = len(pos)
            num_neg_ex = num_ex - num_pos_ex
        else:
            num_pos_ex = len(pos)
            num_neg_ex = len(neg)
        positive = np.random.choice(pos, size=min(num_pos_ex, len(pos)), replace=False)
        negative = np.random.choice(neg, size=min(num_neg_ex, len(neg)), replace=False)
        return positive, negative

    def get_prediction(self, models, x1, x2):
        for i, model in enumerate(models):
            model.eval()
            model.to(self.device)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            if i == 0:
                _, scores = model(x1, x2)
            else:
                _, sc = model(x1, x2)
                scores = scores + sc
        scores = scores / len(models)
        prediction = model.inv_vocab[scores.argmax(1).cpu()]
        return prediction

    def fit_one(self, pos: Union[Set[OWLNamedIndividual], Set[str]], neg: Union[Set[OWLNamedIndividual], Set[str]]):
        #print("\n\n#### In fit one\n\n")
        if isinstance(pos[0], OWLNamedIndividual):
            pos_str = [ind.str.split("/")[-1] for ind in pos]
            neg_str = [ind.str.split("/")[-1] for ind in neg]
        elif isinstance(pos[0], str):
            pos_str = pos
            neg_str = neg
        else:
            raise ValueError(f"Invalid input type, was expecting OWLNamedIndividual or str but found {type(pos[0])}")
        Pos = np.random.choice(pos_str, size=(self.num_predictions, len(pos_str)), replace=True)
        Neg = np.random.choice(neg_str, size=(self.num_predictions, len(neg_str)), replace=True)

        assert self.load_pretrained and self.learner_names, \
            "No pretrained model found. Please first train NCES, see the <<train>> method below"

        dataset = NCESDataLoaderInference([("", Pos_str, Neg_str) for (Pos_str, Neg_str) in zip(Pos, Neg)],
                                          self.instance_embeddings,
                                          self.vocab, self.inv_vocab, False, self.sorted_examples)

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                collate_fn=self.collate_batch_inference, shuffle=False)
        x_pos, x_neg = next(iter(dataloader))
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions_raw = self.get_prediction(self.model, x_pos, x_neg)

        predictions = []
        for prediction in predictions_raw:
            try:
                prediction_str = "".join(before_pad(prediction.squeeze()))
                concept = self.dl_parser.parse(prediction_str)
            except:
                prediction_str = simpleSolution.predict("".join(before_pad(prediction.squeeze())))
                concept = self.dl_parser.parse(prediction_str)
                if self.verbose>0:
                    print("Prediction: ", prediction_str)
            predictions.append(concept)
        return predictions

    def fit(self, learning_problem: PosNegLPStandard, **kwargs):
        pos = learning_problem.pos
        neg = learning_problem.neg
        if isinstance(pos, set) or isinstance(pos, frozenset):
            pos_list = list(pos)
            neg_list = list(neg)
            if self.sorted_examples:
                pos_list = sorted(pos_list)
                neg_list = sorted(neg_list)
        else:
            raise ValueError(f"Expected pos and neg to be sets, got {type(pos)} and {type(neg)}")
        predictions = self.fit_one(pos_list, neg_list)

        predictions_as_nodes = []
        for concept in predictions:
            try:
                concept_individuals_count = self.kb.individuals_count(concept)
            except AttributeError:
                concept = self.dl_parser.parse('⊤')
                concept_individuals_count = self.kb.individuals_count(concept)
            concept_length = init_length_metric().length(concept)
            concept_instances = set(self.kb.individuals(concept)) if isinstance(pos_list[0],
                                                                                OWLNamedIndividual) else set(
                [ind.str.split("/")[-1] for ind in self.kb.individuals(concept)])
            tp, fn, fp, tn = compute_tp_fn_fp_tn(concept_instances, pos, neg)
            quality = self.quality_func.score2(tp, fn, fp, tn)[1]
            node = NCESNode(concept, length=concept_length, individuals_count=concept_individuals_count,
                            quality=quality)
            predictions_as_nodes.append(node)
        predictions_as_nodes = sorted(predictions_as_nodes, key=lambda x: -x.quality)
        self.best_predictions = predictions_as_nodes
        return self

    def best_hypotheses(self, n=1) -> Union[OWLClassExpression, Iterable[OWLClassExpression]]:  # pragma: no cover
        if self.best_predictions is None:
            print("NCES needs to be fitted to a problem first")
            return None
        elif len(self.best_predictions) == 1 or n == 1:
            return self.best_predictions[0].concept
        else:
            return [best.concept for best in self.best_predictions[:n]]

    def convert_to_list_str_from_iterable(self, data):  # pragma: no cover
        target_concept_str, examples = data[0], data[1:]
        pos = list(examples[0])
        neg = list(examples[1])
        if isinstance(pos[0], OWLNamedIndividual):
            pos_str = [ind.str.split("/")[-1] for ind in pos]
            neg_str = [ind.str.split("/")[-1] for ind in neg]
        elif isinstance(pos[0], str):
            pos_str, neg_str = list(pos), list(neg)
        else:
            raise ValueError(f"Invalid input type, was expecting OWLNamedIndividual or str but found {type(pos[0])}")
        if self.sorted_examples:
            pos_str, neg_str = sorted(pos_str), sorted(neg_str)
        return (target_concept_str, pos_str, neg_str)

    def fit_from_iterable(self, dataset: Union[List[Tuple[str, Set[OWLNamedIndividual], Set[OWLNamedIndividual]]],
    List[Tuple[str, Set[str], Set[str]]]], shuffle_examples=False,
                          verbose=False, **kwargs) -> List:  # pragma: no cover
        """
        - Dataset is a list of tuples where the first items are strings corresponding to target concepts.
        
        - This function returns predictions as owl class expressions, not nodes as in fit
        """
        assert self.load_pretrained and self.learner_names, \
            "No pretrained model found. Please first train NCES, refer to the <<train>> method"
        dataset = [self.convert_to_list_str_from_iterable(datapoint) for datapoint in dataset]
        dataset = NCESDataLoaderInference(dataset, self.instance_embeddings, self.vocab, self.inv_vocab,
                                          shuffle_examples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                collate_fn=self.collate_batch_inference, shuffle=False)
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions_as_owl_class_expressions = []
        predictions_str = []
        for x_pos, x_neg in dataloader:
            predictions = self.get_prediction(self.model, x_pos, x_neg)
            for prediction in predictions:
                try:
                    prediction_str = "".join(before_pad(prediction))
                    ce = self.dl_parser.parse(prediction_str)
                    predictions_str.append(prediction_str)
                except:
                    prediction_str = simpleSolution.predict("".join(before_pad(prediction)))
                    predictions_str.append(prediction_str)
                    ce = self.dl_parser.parse(prediction_str)
                predictions_as_owl_class_expressions.append(ce)
        if verbose:
            print("Predictions: ", predictions_str)
        return predictions_as_owl_class_expressions

    @staticmethod
    def generate_training_data(kb_path, num_lps=1000, storage_dir="./NCES_Training_Data"):
        lp_gen = LPGen(kb_path=kb_path, max_num_lps=num_lps, storage_dir=storage_dir)
        lp_gen.generate()
        print("Loading generated data...")
        with open(f"{storage_dir}/LPs.json") as file:
            lps = list(json.load(file).items())
            print("Number of learning problems:", len(lps))
        return lps



    def train(self, data: Iterable[List[Tuple]]=None, epochs=50, batch_size=64, num_lps=1000, learning_rate=1e-4, decay_rate=0.0,
              clip_value=5.0, num_workers=8, save_model=True, storage_path=None, optimizer='Adam', record_runtime=True,
              example_sizes=None, shuffle_examples=False):
        if os.cpu_count() <= num_workers:
            num_workers = max(0,os.cpu_count()-1)
        if storage_path is None:
            currentDateAndTime = datetime.now()
            storage_path = f'NCES-Experiment-{currentDateAndTime.strftime("%H-%M-%S")}'
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        self.trained_models_path = storage_path+"/trained_models"
        if batch_size is None:
            batch_size = self.batch_size
        if data is None:
            data = self.generate_training_data(self.knowledge_base_path, num_lps=num_lps, storage_dir=storage_path)
        train_dataset = NCESDataLoader(data, self.instance_embeddings, self.vocab, self.inv_vocab,
                                       shuffle_examples=shuffle_examples, max_length=self.max_length,
                                       example_sizes=example_sizes)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      collate_fn=self.collate_batch, shuffle=True)
        
        trainer = NCESTrainer(self, epochs=epochs, learning_rate=learning_rate, decay_rate=decay_rate,
                              clip_value=clip_value, num_workers=num_workers, storage_path=storage_path)
        trainer.train(train_dataloader, save_model, optimizer, record_runtime)
        
        
        
class NCES2(BaseNCES):
    """Neural Class Expression Synthesis in ALCHIQ(D)."""

    def __init__(self, knowledge_base_path,
                 quality_func: Optional[AbstractScorer] = None, num_predictions=5,
                 path_of_trained_models="", proj_dim=128, rnn_n_layers=2, drop_prob=0.1,
                 num_heads=4, num_seeds=1, m=[32, 64, 128], ln=False, embedding_dim=256, 
                 input_dropout=0.0, feature_map_dropout=0.1, kernel_size=4, num_of_output_channels=32, 
                 learning_rate=1e-4, decay_rate=0.0, clip_value=5.0, batch_size=256, num_workers=4, 
                 max_length=48, load_pretrained=True, sorted_examples=False, verbose: int = 0):
        super().__init__(knowledge_base_path, quality_func, num_predictions, proj_dim, drop_prob,
                 num_heads, num_seeds, m, ln, learning_rate, decay_rate, clip_value,
                 batch_size, num_workers, max_length, load_pretrained, sorted_examples, verbose)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        self.input_dropout = input_dropout
        self.feature_map_dropout = feature_map_dropout
        self.kernel_size = kernel_size
        self.num_of_output_channels = num_of_output_channels
        self._maybe_load_pretrained_config()
        self.model = self.get_synthesizer()
        
        
    def _maybe_load_pretrained_config(self):
        if isinstance(self.m, int):
            self.m = [self.m]
        if self.load_pretrained and len(glob.glob(self.path_of_trained_models + "/*.pt")):
            stop_outer_loop = False
            try:
                for file_name in glob.glob(self.path_of_trained_models + "/*.pt"):
                    for m in self.m:
                        if m in file_name:
                            try:
                                weights = torch.load(file_name, map_location=self.device, weights_only=True)
                                num_ents, half_emb_dim = weights["emb_ent_real"].weight.data.shape
                                num_rels, _ = weights["emb_rel_real"].weight.data.shape
                                self.embedding_dim = 2*half_emb_dim
                                self.num_entities = num_ents
                                self.num_relations = num_rels
                                stop_outer_loop = True
                                break
                            except:
                                pass
                    if stop_outer_loop:
                        break
                
    
    def get_synthesizer(self, path=None):
        
        Models = {str(m): {"emb_model": ConEx(self.embedding_dim, self.num_entities, self.num_relations, self.input_dropout,
                                              self.feature_map_dropout, self.kernel_size, self.num_of_output_channels), 
                           "model": SetTransformer(self.knowledge_base_path, self.vocab, self.inv_vocab, self.max_length,
                                   self.input_size, self.proj_dim, self.num_heads, self.num_seeds, m, self.ln)} for m in self.m}
        if self.load_pretrained:
            num_loaded = 0
            if path is None:
                try:
                    if len(glob.glob(self.path_of_trained_models + "/*.pt")) == 0:
                        raise FileNotFoundError(f"`path` is None and no models found in {self.path_of_trained_models}")
                    else:
                        for file_name in glob.glob(self.path_of_trained_models + "/*.pt"):
                            for m in self.m:
                                if m in file_name:
                                    try:
                                        weights = torch.load(file_name, map_location=self.device, weights_only=True)
                                        model = Models[str(m)]["model"]
                                        model.load_state_dict(weights)
                                        Models[str(m)]["model"] = model.eval()
                                        num_loaded += 1
                                    except Exception:
                                        weights = torch.load(file_name, map_location=self.device, weights_only=True)
                                        emb_model = Models[str(m)]["emb_model"]
                                        emb_model.load_state_dict(weights)
                                        Models[str(m)]["emb_model"] = emb_model.eval()
                except Exception as e:
                    print(e)
                    raise FileNotFoundError
                
                if num_loaded == len(self.m):
                    print("\nLoaded NCES2 weights!\n")
                    return Models
                else:
                    print("!!!Returning untrained models, could not load pretrained")
                    return Models

            elif len(glob.glob(path+"/*.pt")) == 0:
                 print("No pretrained model found! If directory is empty or does not exist, set the NCES2 `load_pretrained` parameter to `False` or make sure `save_model` was set to `True` in the .train() method.")
                raise FileNotFoundError
            else:
                num_loaded = 0
                for file_name in glob.glob(path+"/*.pt"):
                    for m, model, emb_model in zip(self.m, Untrained_Synt, Untrained_Emb):
                        if m in file_name:
                            try:
                                weights = torch.load(file_name, map_location=self.device, weights_only=True)
                                model = Models[str(m)]["model"]
                                model.load_state_dict(weights)
                                Models[str(m)]["model"] = model.eval()
                                num_loaded += 1
                            except Exception:
                                weights = torch.load(file_name, map_location=self.device, weights_only=True)
                                emb_model = Models[str(m)]["emb_model"]
                                emb_model.load_state_dict(weights)
                                Models[str(m)]["emb_model"] = emb_model.eval()
                if num_loaded == len(self.m):
                    print("\nLoaded NCES2 weights!\n")
                    return Models
                else:
                    print("!!!Returning untrained models, could not load pretrained")
                    return Models
        else:
            print("!!!Returning untrained models, could not load pretrained. Check the `load_pretrained parameter` or train the models using NCES2.train(data).")
            return Models


    def refresh(self, path=None):
        if path is not None:
            self.load_pretrained = True
        self.model = self.get_synthesizer(path)

    def sample_examples(self, pos, neg):  # pragma: no cover
        assert type(pos[0]) == type(neg[0]), "The two iterables pos and neg must be of same type"
        num_ex = self.num_examples
        if min(len(pos), len(neg)) >= num_ex // 2:
            if len(pos) > len(neg):
                num_neg_ex = num_ex // 2
                num_pos_ex = num_ex - num_neg_ex
            else:
                num_pos_ex = num_ex // 2
                num_neg_ex = num_ex - num_pos_ex
        elif len(pos) + len(neg) >= num_ex and len(pos) > len(neg):
            num_neg_ex = len(neg)
            num_pos_ex = num_ex - num_neg_ex
        elif len(pos) + len(neg) >= num_ex and len(pos) < len(neg):
            num_pos_ex = len(pos)
            num_neg_ex = num_ex - num_pos_ex
        else:
            num_pos_ex = len(pos)
            num_neg_ex = len(neg)
        positive = np.random.choice(pos, size=min(num_pos_ex, len(pos)), replace=False)
        negative = np.random.choice(neg, size=min(num_neg_ex, len(neg)), replace=False)
        return positive, negative

    def get_prediction(self, models, x1, x2):
        for i, model in enumerate(models):
            model.eval()
            model.to(self.device)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            if i == 0:
                _, scores = model(x1, x2)
            else:
                _, sc = model(x1, x2)
                scores = scores + sc
        scores = scores / len(models)
        prediction = model.inv_vocab[scores.argmax(1).cpu()]
        return prediction

    def fit_one(self, pos: Union[Set[OWLNamedIndividual], Set[str]], neg: Union[Set[OWLNamedIndividual], Set[str]]):
        #print("\n\n#### In fit one\n\n")
        if isinstance(pos[0], OWLNamedIndividual):
            pos_str = [ind.str.split("/")[-1] for ind in pos]
            neg_str = [ind.str.split("/")[-1] for ind in neg]
        elif isinstance(pos[0], str):
            pos_str = pos
            neg_str = neg
        else:
            raise ValueError(f"Invalid input type, was expecting OWLNamedIndividual or str but found {type(pos[0])}")
        Pos = np.random.choice(pos_str, size=(self.num_predictions, len(pos_str)), replace=True)
        Neg = np.random.choice(neg_str, size=(self.num_predictions, len(neg_str)), replace=True)

        assert self.load_pretrained and self.learner_names, \
            "No pretrained model found. Please first train NCES, see the <<train>> method below"

        dataset = NCESDataLoaderInference([("", Pos_str, Neg_str) for (Pos_str, Neg_str) in zip(Pos, Neg)],
                                          self.instance_embeddings,
                                          self.vocab, self.inv_vocab, False, self.sorted_examples)

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                collate_fn=self.collate_batch_inference, shuffle=False)
        x_pos, x_neg = next(iter(dataloader))
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions_raw = self.get_prediction(self.model, x_pos, x_neg)

        predictions = []
        for prediction in predictions_raw:
            try:
                prediction_str = "".join(before_pad(prediction.squeeze()))
                concept = self.dl_parser.parse(prediction_str)
            except:
                prediction_str = simpleSolution.predict("".join(before_pad(prediction.squeeze())))
                concept = self.dl_parser.parse(prediction_str)
                if self.verbose>0:
                    print("Prediction: ", prediction_str)
            predictions.append(concept)
        return predictions

    def fit(self, learning_problem: PosNegLPStandard, **kwargs):
        pos = learning_problem.pos
        neg = learning_problem.neg
        if isinstance(pos, set) or isinstance(pos, frozenset):
            pos_list = list(pos)
            neg_list = list(neg)
            if self.sorted_examples:
                pos_list = sorted(pos_list)
                neg_list = sorted(neg_list)
        else:
            raise ValueError(f"Expected pos and neg to be sets, got {type(pos)} and {type(neg)}")
        predictions = self.fit_one(pos_list, neg_list)

        predictions_as_nodes = []
        for concept in predictions:
            try:
                concept_individuals_count = self.kb.individuals_count(concept)
            except AttributeError:
                concept = self.dl_parser.parse('⊤')
                concept_individuals_count = self.kb.individuals_count(concept)
            concept_length = init_length_metric().length(concept)
            concept_instances = set(self.kb.individuals(concept)) if isinstance(pos_list[0],
                                                                                OWLNamedIndividual) else set(
                [ind.str.split("/")[-1] for ind in self.kb.individuals(concept)])
            tp, fn, fp, tn = compute_tp_fn_fp_tn(concept_instances, pos, neg)
            quality = self.quality_func.score2(tp, fn, fp, tn)[1]
            node = NCESNode(concept, length=concept_length, individuals_count=concept_individuals_count,
                            quality=quality)
            predictions_as_nodes.append(node)
        predictions_as_nodes = sorted(predictions_as_nodes, key=lambda x: -x.quality)
        self.best_predictions = predictions_as_nodes
        return self

    def best_hypotheses(self, n=1) -> Union[OWLClassExpression, Iterable[OWLClassExpression]]:  # pragma: no cover
        if self.best_predictions is None:
            print("NCES needs to be fitted to a problem first")
            return None
        elif len(self.best_predictions) == 1 or n == 1:
            return self.best_predictions[0].concept
        else:
            return [best.concept for best in self.best_predictions[:n]]

    def convert_to_list_str_from_iterable(self, data):  # pragma: no cover
        target_concept_str, examples = data[0], data[1:]
        pos = list(examples[0])
        neg = list(examples[1])
        if isinstance(pos[0], OWLNamedIndividual):
            pos_str = [ind.str.split("/")[-1] for ind in pos]
            neg_str = [ind.str.split("/")[-1] for ind in neg]
        elif isinstance(pos[0], str):
            pos_str, neg_str = list(pos), list(neg)
        else:
            raise ValueError(f"Invalid input type, was expecting OWLNamedIndividual or str but found {type(pos[0])}")
        if self.sorted_examples:
            pos_str, neg_str = sorted(pos_str), sorted(neg_str)
        return (target_concept_str, pos_str, neg_str)

    def fit_from_iterable(self, dataset: Union[List[Tuple[str, Set[OWLNamedIndividual], Set[OWLNamedIndividual]]],
    List[Tuple[str, Set[str], Set[str]]]], shuffle_examples=False,
                          verbose=False, **kwargs) -> List:  # pragma: no cover
        """
        - Dataset is a list of tuples where the first items are strings corresponding to target concepts.
        
        - This function returns predictions as owl class expressions, not nodes as in fit
        """
        assert self.load_pretrained and self.learner_names, \
            "No pretrained model found. Please first train NCES, refer to the <<train>> method"
        dataset = [self.convert_to_list_str_from_iterable(datapoint) for datapoint in dataset]
        dataset = NCESDataLoaderInference(dataset, self.instance_embeddings, self.vocab, self.inv_vocab,
                                          shuffle_examples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                collate_fn=self.collate_batch_inference, shuffle=False)
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions_as_owl_class_expressions = []
        predictions_str = []
        for x_pos, x_neg in dataloader:
            predictions = self.get_prediction(self.model, x_pos, x_neg)
            for prediction in predictions:
                try:
                    prediction_str = "".join(before_pad(prediction))
                    ce = self.dl_parser.parse(prediction_str)
                    predictions_str.append(prediction_str)
                except:
                    prediction_str = simpleSolution.predict("".join(before_pad(prediction)))
                    predictions_str.append(prediction_str)
                    ce = self.dl_parser.parse(prediction_str)
                predictions_as_owl_class_expressions.append(ce)
        if verbose:
            print("Predictions: ", predictions_str)
        return predictions_as_owl_class_expressions

    @staticmethod
    def generate_training_data(kb_path, num_lps=1000, storage_dir="./NCES_Training_Data"):
        lp_gen = LPGen(kb_path=kb_path, max_num_lps=num_lps, storage_dir=storage_dir)
        lp_gen.generate()
        print("Loading generated data...")
        with open(f"{storage_dir}/LPs.json") as file:
            lps = list(json.load(file).items())
            print("Number of learning problems:", len(lps))
        return lps



    def train(self, data: Iterable[List[Tuple]]=None, epochs=50, batch_size=64, num_lps=1000, learning_rate=1e-4, decay_rate=0.0,
              clip_value=5.0, num_workers=8, save_model=True, storage_path=None, optimizer='Adam', record_runtime=True,
              example_sizes=None, shuffle_examples=False):
        if os.cpu_count() <= num_workers:
            num_workers = max(0,os.cpu_count()-1)
        if storage_path is None:
            currentDateAndTime = datetime.now()
            storage_path = f'NCES-Experiment-{currentDateAndTime.strftime("%H-%M-%S")}'
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        self.trained_models_path = storage_path+"/trained_models"
        if batch_size is None:
            batch_size = self.batch_size
        if data is None:
            data = self.generate_training_data(self.knowledge_base_path, num_lps=num_lps, storage_dir=storage_path)
        train_dataset = NCESDataLoader(data, self.instance_embeddings, self.vocab, self.inv_vocab,
                                       shuffle_examples=shuffle_examples, max_length=self.max_length,
                                       example_sizes=example_sizes)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      collate_fn=self.collate_batch, shuffle=True)
        
        trainer = NCESTrainer(self, epochs=epochs, learning_rate=learning_rate, decay_rate=decay_rate,
                              clip_value=clip_value, num_workers=num_workers, storage_path=storage_path)
        trainer.train(train_dataloader, save_model, optimizer, record_runtime)


        
class ROCES(NCES2):
    """Robust Class Expression Synthesis in Description Logics via Iterative Sampling."""

    def __init__(self, knowledge_base_path,
                 quality_func: Optional[AbstractScorer] = None, num_predictions=5,
                 path_of_trained_models="", proj_dim=128, rnn_n_layers=2, drop_prob=0.1,
                 num_heads=4, num_seeds=1, m=[32, 64, 128], ln=False, embedding_dim=256, 
                 input_dropout=0.0, feature_map_dropout=0.1, kernel_size=4, num_of_output_channels=32, 
                 learning_rate=1e-4, decay_rate=0.0, clip_value=5.0, batch_size=256, num_workers=4, 
                 max_length=48, load_pretrained=True, sorted_examples=False, verbose: int = 0):
        super().__init__(knowledge_base_path,
                 quality_func, num_predictions, path_of_trained_models, proj_dim, rnn_n_layers, drop_prob,
                 num_heads, num_seeds, m, ln, embedding_dim, input_dropout, feature_map_dropout, 
                 kernel_size, num_of_output_channels, learning_rate, decay_rate, clip_value, batch_size,
                 num_workers, max_length, load_pretrained=True, sorted_examples, verbose)


    def train(self, data: Iterable[List[Tuple]]=None, epochs=50, batch_size=64, num_lps=1000, learning_rate=1e-4, decay_rate=0.0,
              clip_value=5.0, num_workers=8, save_model=True, storage_path=None, optimizer='Adam', record_runtime=True,
              example_sizes=None, shuffle_examples=False):
        if os.cpu_count() <= num_workers:
            num_workers = max(0,os.cpu_count()-1)
        if storage_path is None:
            currentDateAndTime = datetime.now()
            storage_path = f'NCES-Experiment-{currentDateAndTime.strftime("%H-%M-%S")}'
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        self.trained_models_path = storage_path+"/trained_models"
        if batch_size is None:
            batch_size = self.batch_size
        if data is None:
            data = self.generate_training_data(self.knowledge_base_path, num_lps=num_lps, storage_dir=storage_path)
        train_dataset = NCESDataLoader(data, self.instance_embeddings, self.vocab, self.inv_vocab,
                                       shuffle_examples=shuffle_examples, max_length=self.max_length,
                                       example_sizes=example_sizes)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      collate_fn=self.collate_batch, shuffle=True)
        
        trainer = NCESTrainer(self, epochs=epochs, learning_rate=learning_rate, decay_rate=decay_rate,
                              clip_value=clip_value, num_workers=num_workers, storage_path=storage_path)
        trainer.train(train_dataloader, save_model, optimizer, record_runtime)

