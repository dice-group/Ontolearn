from ontolearn.concept_learner import EvoLearner
from typing import Optional
from ontolearn.abstracts  import AbstractDrill, AbstractFitness, AbstractScorer, AbstractNode, BaseRefinement, \
    AbstractHeuristic, EncodedPosNegLPStandardKind
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.search import EvoLearnerNode
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter, EntropyValueSplitter
from ontolearn.ea_initialization import AbstractEAInitialization, EARandomInitialization, EARandomWalkInitialization
from ontolearn.ea_algorithms import AbstractEvolutionaryAlgorithm, EASimple

class EvoLearnerWrapper:
    def __init__(self, knowledge_base: KnowledgeBase, quality_func: Optional[AbstractScorer] = None, fitness_func: Optional[AbstractFitness] = None, init_method: Optional[AbstractEAInitialization] = None, algorithm: Optional[AbstractEvolutionaryAlgorithm] = None, mut_uniform_gen: Optional[AbstractEAInitialization] = None, value_splitter: Optional[str] = None, terminate_on_goal: Optional[bool] = None, max_runtime: Optional[int] = None, use_data_properties: bool = True, use_card_restrictions: bool = True, use_inverse: bool = False, tournament_size: int = 7, card_limit: int = 10, population_size: int = 800, num_generations: int = 200, height_limit: int = 17):
        self.EvoLearner = EvoLearner
        self.binning_value_splitter = BinningValueSplitter()
        self.entropy_value_splitter = EntropyValueSplitter()
        self.knowledge_base = knowledge_base
        self.quality_func = quality_func
        self.fitness_func = fitness_func
        self.init_method = init_method
        self.terminate_on_goal = terminate_on_goal
        self.algorithm = algorithm
        self.mut_uniform_gen = mut_uniform_gen
        self.value_splitter = self.transform_value_splitter(value_splitter)
        self.use_data_properties = use_data_properties
        self.use_card_restrictions = use_card_restrictions
        self.max_runtime = max_runtime
        self.use_inverse = use_inverse
        self.tournament_size = tournament_size
        self.card_limit = card_limit
        self.population_size = population_size
        self.num_generations = num_generations
        self.height_limit = height_limit

    def transform_value_splitter(self, value_splitter):
        if value_splitter == 'EntropyValueSplitter':
            value_splitter = self.entropy_value_splitter        
        elif value_splitter == 'BinningValueSplitter':
            value_splitter = self.binning_value_splitter
        else:
            value_splitter = None
        return value_splitter
        
    def get_evolearner_model(self):
        model = EvoLearner(self.knowledge_base,
            self.quality_func,
            self.fitness_func, 
            self.init_method, 
            self.algorithm, 
            self.mut_uniform_gen, 
            self.value_splitter, 
            self.terminate_on_goal,
            self. max_runtime, 
            self.use_data_properties, 
            self.use_card_restrictions, 
            self.use_inverse, 
            self.tournament_size, 
            self.card_limit, 
            self.population_size, 
            self.num_generations, 
            self.height_limit)
        return model
        

        
            
