from abc import ABCMeta, abstractmethod
from typing import ClassVar, Final, Optional
from deap.algorithms import varAnd
from heapq import nlargest


class AbstractEvolutionaryAlgorithm(metaclass=ABCMeta):
    """
    An abstract class for evolutionary algorithms.
    """
    __slots__ = 'crossover_pr', 'mutation_pr', 'elitism', 'elite_size'

    name: ClassVar[str]

    def __init__(self,
                 crossover_pr: Optional[float] = None,
                 mutation_pr: Optional[float] = None,
                 elitism: Optional[bool] = None,
                 elite_size: Optional[float] = None):
        """Create a new evolutionary algorithm

        Args:
            crossover_pb: crossover probability
            mutation_pb: mutation probability
            elitism: whether to use elitism. default to False
            elite_size: ratio of individuals that are kept between generations if elitism is used
        """
        self.crossover_pr = crossover_pr
        self.mutation_pr = mutation_pr
        self.elitism = elitism
        self.elite_size = elite_size

        self.__set_default_values()

    def __set_default_values(self):
        if self.crossover_pr is None:
            self.crossover_pr = 0.9

        if self.mutation_pr is None:
            self.mutation_pr = 0.1

        if self.elitism is None:
            self.elitism = False

        if self.elite_size is None:
            self.elite_size = 0.1

    @abstractmethod
    def evolve(self, toolbox, population, num_generations, verbose=False):
        pass


class EASimple(AbstractEvolutionaryAlgorithm):
    __slots__ = ()

    name: Final = 'EASimple'

    def __init__(self,
                 crossover_pr: Optional[float] = None,
                 mutation_pr: Optional[float] = None,
                 elitism: Optional[bool] = None,
                 elite_size: Optional[float] = None):
        super().__init__(crossover_pr=crossover_pr,
                         mutation_pr=mutation_pr,
                         elitism=elitism,
                         elite_size=elite_size)

    def evolve(self, toolbox, population, num_generations, verbose=False):
        num_elite = self.elite_size*len(population) if self.elitism else 0
        num_tournaments = len(population) - num_elite

        for p in population:
            toolbox.evaluate(p)

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
                    toolbox.evaluate(off)
                    if off.quality.values[0] == 1.0:
                        goal_found = True

            population[:] = offspring + elite

            if verbose:
                print("\nGeneration: ", gen, "-----------------------------")
                toolbox.print()
            
            gen += 1
