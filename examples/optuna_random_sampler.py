import optuna
import json
import os
from random import shuffle
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from ontolearn.metrics import F1
from sklearn.model_selection import ParameterGrid
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.value_splitter import BinningValueSplitter, EntropyValueSplitter

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('carcinogenesis_lp.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
#TO DO: Create a Wrapper in order to try other hyper parameters such as use_card_restrictions, different value_splitter etc

class OptunaSamplers():
    def __init__(self, lp):
        self.lp = lp
        self.study_random_sampler = optuna.create_study(sampler=RandomSampler())
        self.study_tpe_sampler = optuna.create_study(sampler=TPESampler())
        self.sampler = optuna.samplers.CmaEsSampler()
        self.nsgii = optuna.samplers.NSGAIISampler(population_size=100)
        self.qmc_sampler = optuna.samplers.QMCSampler()
        self.study_cmaes_sampler = optuna.create_study(sampler=self.sampler)
        self.study_nsgaii_sampler = optuna.create_study(sampler=self.nsgii)
        self.study_qmc_sampler = optuna.create_study(sampler=self.qmc_sampler)

    def objective(self,trial):
        binning_value_splitter = BinningValueSplitter()
        entropy_value_splitter = EntropyValueSplitter()
        max_runtime = trial.suggest_int("max_runtime", 10, 20)
        tournament_size = trial.suggest_int("tournament_size", 2, 10)
        height_limit = trial.suggest_int('height_limit',3, 25)
        card_limit = trial.suggest_int('card_limit', 5, 10)
        #value_splitter = trial.suggest_categorical('value_splitter',[binning_value_splitter,entropy_value_splitter])
        model = EvoLearner(knowledge_base=kb, max_runtime=max_runtime, tournament_size=tournament_size, height_limit=height_limit,card_limit=card_limit)
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        quality = hypotheses[0].quality
        return quality

    def get_best_optimisation_result_for_random_sampler(self, n_trials):
        self.study_random_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_random_sampler.best_trial)
        #best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_random_sampler.best_params)
        #score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_random_sampler.best_value)

    def get_best_optimization_result_for_tpe_sampler(self, n_trials):
        self.study_tpe_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_tpe_sampler.best_trial)
        #best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_tpe_sampler.best_params)
        #score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_tpe_sampler.best_value)

    def get_best_optimization_result_for_cmes_sampler(self, n_trials):
        self.study_cmaes_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_cmaes_sampler.best_trial)
        #best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_cmaes_sampler.best_params)
        #score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_cmaes_sampler.best_value)

    def get_best_optimization_result_for_nsgii_sampler(self, population_size):
        self.study_nsgaii_sampler.optimize(self.objective, n_trials=population_size)
        print("-------BEST TRIAL-----------")
        print(self.study_nsgaii_sampler.best_trial)
        #best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_nsgaii_sampler.best_params)
        #score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_nsgaii_sampler.best_value)
    
    def get_best_optimization_result_for_qmc_sampler(self, n_trials):
        self.study_qmc_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_qmc_sampler.best_trial)
        #best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_qmc_sampler.best_params)
        #score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_qmc_sampler.best_value)


if __name__ == "__main__":
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))
        
        #shuffle the Positive and Negative Sample
        shuffle(typed_pos)   
        shuffle(typed_neg)

        #Split the data into Training Set and Test Set
        train_pos = set(typed_pos[:int(len(typed_pos)*0.8)])
        train_neg = set(typed_neg[:int(len(typed_neg)*0.8)])
        test_pos = set(typed_pos[-int(len(typed_pos)*0.2):])
        test_neg = set(typed_neg[-int(len(typed_neg)*0.2):])
        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        #create class object and get the optimised result
        optuna1 = OptunaSamplers(lp)
        optuna1.get_best_optimization_result_for_qmc_sampler(10)