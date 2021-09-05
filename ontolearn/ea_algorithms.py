from abc import ABCMeta, abstractmethod
from typing import ClassVar, Final, List, Optional
from deap.algorithms import varAnd
from deap.base import Toolbox
from deap import creator
from heapq import nlargest
import time, logging

logger = logging.getLogger(__name__)


class AbstractEvolutionaryAlgorithm(metaclass=ABCMeta):
    """
    An abstract class for evolutionary algorithms.
    """
    __slots__ = 'crossover_pr', 'mutation_pr', 'elitism', 'elite_size'

    name: ClassVar[str]

    crossover_pr: float
    mutation_pr: float
    elitism: bool
    elite_size: float

    def __init__(self,
                 crossover_pr: float = 0.9,
                 mutation_pr: float= 0.1,
                 elitism: bool = False,
                 elite_size: float = 0.1):
        """Create a new evolutionary algorithm

        Args:
            crossover_pb: crossover probability
            mutation_pb: mutation probability
            elitism: whether to use elitism. Defaults to False
            elite_size: ratio of individuals that are kept between generations if elitism is used
        """
        self.crossover_pr = crossover_pr
        self.mutation_pr = mutation_pr
        self.elitism = elitism
        self.elite_size = elite_size

    @abstractmethod
    def evolve(self,
               toolbox: Toolbox,
               population: 'List[creator.Individual]',
               num_generations: int,
               start_time: float,
               verbose: bool = False) -> bool:
        pass


class EASimple(AbstractEvolutionaryAlgorithm):
    __slots__ = ()

    name: Final = 'EASimple'

    def __init__(self,
                 crossover_pr: float = 0.9,
                 mutation_pr: float = 0.1,
                 elitism: bool = False,
                 elite_size: float = 0.1):
        super().__init__(crossover_pr=crossover_pr,
                         mutation_pr=mutation_pr,
                         elitism=elitism,
                         elite_size=elite_size)

    def evolve(self,
               toolbox: Toolbox,
               population: 'List[creator.Individual]',
               num_generations: int,
               start_time: float,
               verbose: int = 0) -> bool:

        num_elite = int(self.elite_size*len(population)) if self.elitism else 0
        num_tournaments = len(population) - num_elite

        for p in population:
            toolbox.apply_fitness(p)

        elite = []
        gen = 1
        goal_found = False
        while gen <= num_generations and not (goal_found and toolbox.terminate_on_goal()):
            if self.elitism:
                elite = nlargest(num_elite, population, key=lambda p: p.fitness.values[0])

            offspring = toolbox.select(population, k=num_tournaments)
            offspring = varAnd(offspring, toolbox, self.crossover_pr, self.mutation_pr)

            for off in offspring:
                if not off.fitness.valid:
                    toolbox.apply_fitness(off)
                    if off.quality.values[0] == 1.0:
                        goal_found = True

            population[:] = offspring + elite

            if verbose > 0:
                logger.info(f"Generation: {gen}")
                toolbox.print()

            if (time.time() - start_time) > toolbox.max_runtime():
                return goal_found

            gen += 1

        return goal_found
