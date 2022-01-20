from abc import ABCMeta, abstractmethod
from typing import ClassVar, Final, List, Optional, Tuple
from deap.algorithms import varAnd
from deap.base import Toolbox
from heapq import nlargest
import time
import logging
import itertools

from ontolearn.ea_utils import Tree

logger = logging.getLogger(__name__)


class AbstractEvolutionaryAlgorithm(metaclass=ABCMeta):
    """
    An abstract class for evolutionary algorithms.
    """
    __slots__ = ()

    name: ClassVar[str]

    @abstractmethod
    def __init__(self):
        """Create a new evolutionary algorithm"""
        pass

    @abstractmethod
    def evolve(self,
               toolbox: Toolbox,
               population: List[Tree],
               num_generations: int,
               start_time: float,
               verbose: bool = False) -> Tuple[bool, List[Tree]]:
        pass

    def _log_generation_info(self, toolbox: Toolbox, gen: int, population: List[Tree]):
        logger.info(f'Generation: {gen}')
        for node in toolbox.get_top_hypotheses(population):
            logger.info(node)
        print('#'*100)


class BaseEvolutionaryAlgorithm(AbstractEvolutionaryAlgorithm):
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        pass

    def evolve(self,
               toolbox: Toolbox,
               population: List[Tree],
               num_generations: int,
               start_time: float,
               verbose: int = 0) -> Tuple[bool, List[Tree]]:

        for ind in population:
            toolbox.apply_fitness(ind)

        gen = 1
        goal_found = False
        while gen <= num_generations and not (goal_found and toolbox.terminate_on_goal()):
            goal_found, population = self.generation(toolbox, population)

            if verbose > 0:
                self._log_generation_info(toolbox, gen, population)

            if (time.time() - start_time) > toolbox.max_runtime():
                return goal_found, population

            gen += 1

        return goal_found, population

    @abstractmethod
    def generation(self, toolbox: Toolbox, population: List[Tree], num_selections: int = 0) -> Tuple[bool, List[Tree]]:
        pass


class EASimple(BaseEvolutionaryAlgorithm):
    __slots__ = 'crossover_pr', 'mutation_pr', 'elitism', 'elite_size'

    name: Final = 'EASimple'

    crossover_pr: float
    mutation_pr: float
    elitism: bool
    elite_size: float

    def __init__(self,
                 crossover_pr: float = 0.9,
                 mutation_pr: float = 0.1,
                 elitism: bool = False,
                 elite_size: float = 0.1):

        self.crossover_pr = crossover_pr
        self.mutation_pr = mutation_pr
        self.elitism = elitism
        self.elite_size = elite_size

    def generation(self, toolbox: Toolbox, population: List[Tree], num_selections: int = 0) -> Tuple[bool, List[Tree]]:
        elite = []
        goal_found = False

        num_selections = num_selections if num_selections > 0 else len(population)

        if self.elitism:
            num_elite = int(self.elite_size*num_selections)
            num_selections = num_selections - num_elite
            elite = nlargest(num_elite, population, key=lambda ind: ind.fitness.values[0])

        offspring = toolbox.select(population, k=num_selections)
        offspring = varAnd(offspring, toolbox, self.crossover_pr, self.mutation_pr)

        for off in offspring:
            if not off.fitness.valid:
                toolbox.apply_fitness(off)
                if off.quality.values[0] == 1.0:
                    goal_found = True

        population[:] = offspring + elite
        return goal_found, population


class RegularizedEvolution(BaseEvolutionaryAlgorithm):
    __slots__ = ()

    name: Final = 'RegularizedEvolution'

    def __init__(self):
        pass

    def generation(self, toolbox: Toolbox, population: List[Tree], num_selections: int = 0) -> Tuple[bool, List[Tree]]:
        # TODO: use queue, since normal list has O(n) for pop

        parent = toolbox.select(population, 1)[0]
        parent_copy = toolbox.clone(parent)
        offspring, = toolbox.mutate(parent_copy)
        toolbox.apply_fitness(offspring)
        population.append(offspring)
        population.pop(0)

        return offspring.quality.values[0] == 1.0, population


class MultiPopulation(AbstractEvolutionaryAlgorithm):
    __slots__ = 'base_algorithm', 'migration_size', 'num_populations', 'iso_generations', 'boost'

    name: Final = 'MultiPopulation'

    base_algorithm: BaseEvolutionaryAlgorithm
    migration_size: float
    num_populations: int
    iso_generations: float
    boost: float

    def __init__(self,
                 base_algorithm: Optional[BaseEvolutionaryAlgorithm] = None,
                 migration_size: float = 0.1,
                 num_populations: int = 4,
                 iso_generations: float = 0.1,
                 boost: float = 0.0):
        self.migration_size = migration_size
        self.num_populations = num_populations
        self.iso_generations = iso_generations
        self.base_algorithm = base_algorithm
        self.boost = boost

        if self.base_algorithm is None:
            self.base_algorithm = EASimple()

    def evolve(self,
               toolbox: Toolbox,
               population: List[Tree],
               num_generations: int,
               start_time: float,
               verbose: int = 0) -> Tuple[bool, List[Tree]]:

        assert len(population) % self.num_populations == 0
        population_size = len(population) // self.num_populations
        populations = [population[i::self.num_populations] for i in range(self.num_populations)]

        iso_ngen = int(num_generations*self.iso_generations)
        num_migration = int(population_size*self.migration_size)

        for p in populations:
            for ind in p:
                toolbox.apply_fitness(ind)

        gen = 1
        goal_found = [False] * self.num_populations
        while gen <= iso_ngen and not (any(goal_found) and toolbox.terminate_on_goal()):
            for idx, p in enumerate(populations):
                goal_found[idx], population = self.base_algorithm.generation(toolbox, p, population_size)

                if verbose > 0:
                    self._log_generation_info(toolbox, gen, population, idx)

                populations[idx] = population

            if (time.time() - start_time) > toolbox.max_runtime():
                return self._finalize(goal_found, populations, toolbox)

            gen += 1

        while gen <= num_generations and not (any(goal_found) and toolbox.terminate_on_goal()):

            migrate_inds = []
            for idx, p in enumerate(populations):
                mig = nlargest(num_migration, p, key=lambda ind: ind.fitness.values[0])
                migrate_inds.append([toolbox.clone(ind) for ind in mig])
                if self.boost > 0.0:
                    for ind in migrate_inds[idx]:
                        ind.fitness.values = (ind.fitness.values[0] + self.boost, )

            for idx, p in enumerate(populations):
                goal_found[idx], population = self.base_algorithm.generation(toolbox,
                                                                             p + migrate_inds[idx],
                                                                             population_size)
                if verbose > 0:
                    self._log_generation_info(toolbox, gen, population, idx)

                populations[idx] = population

            if (time.time() - start_time) > toolbox.max_runtime():
                return self._finalize(goal_found, populations, toolbox)

            gen += 1

        return self._finalize(goal_found, populations, toolbox)

    def _log_generation_info(self, toolbox: Toolbox, gen: int, population: List[Tree], idx: int = 0):
        logger.info(f'Population {idx}:')
        logger.info(f'Generation: {gen}')
        for node in toolbox.get_top_hypotheses(population):
            logger.info(node)
        print('#'*100)

    def _finalize(self, goal_found, populations, toolbox):
        population = list(itertools.chain.from_iterable(populations))
        # If boost was used we need to compute the actual fitness values once more, so there is
        # no individual left which has the boost applied
        if self.boost > 0:
            for ind in population:
                toolbox.apply_fitness(ind)
        return any(goal_found), population
