import optuna
import json
import os
from random import shuffle
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
import pandas as pd
import time

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from wrapper_evolearner import EvoLearnerWrapper

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('carcinogenesis_lp.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
df = pd.DataFrame(columns=['LP', 'max_runtime', 'tournament_size', 'height_limit', 'use_data_properties',
                           'quality_func', 'card_limit', 'value_splitter', 'quality_score'])


class OptunaSamplers():
    def __init__(self, lp, concept):
        self.lp = lp
        self.concept = concept
        self.study_random_sampler = optuna.create_study(sampler=RandomSampler(), direction='maximize')
        self.study_tpe_sampler = optuna.create_study(sampler=TPESampler(), direction='maximize')
        self.sampler = optuna.samplers.CmaEsSampler()
        self.nsgii = optuna.samplers.NSGAIISampler(population_size=100)
        self.qmc_sampler = optuna.samplers.QMCSampler()
        self.study_cmaes_sampler = optuna.create_study(sampler=self.sampler)
        self.study_nsgaii_sampler = optuna.create_study(sampler=self.nsgii)
        self.study_qmc_sampler = optuna.create_study(sampler=self.qmc_sampler)
    
    def write_to_df(self, **space):
        df.loc[len(df.index)] = [self.concept, space['max_runtime'], space['tournament_size'], 
                                 space['height_limit'], space['card_limit'], space['use_data_properties'], 
                                 space['quality_func'], space['value_splitter'], space['quality_score']]
    
    def convert_to_csv(self, df):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "Output"+str(timestr)  
        df.to_csv(filename+".csv", index=False)

    def objective(self, trial):             
        max_runtime = trial.suggest_int("max_runtime", 10, 20)
        tournament_size = trial.suggest_int("tournament_size", 2, 10)
        height_limit = trial.suggest_int('height_limit', 3, 25)
        card_limit = trial.suggest_int('card_limit', 5, 10)
        use_data_properties = trial.suggest_categorical('use_data_properties', ['True', 'False'])
        quality_func = trial.suggest_categorical('quality_func', ['F1', 'Accuracy'])
        value_splitter = trial.suggest_categorical('value_splitter', 
                                                   ['binning_value_splitter', 'entropy_value_splitter'])
        # create a dictionary
        space = dict()
        space['max_runtime'] = max_runtime
        space['tournament_size'] = tournament_size
        space['height_limit'] = height_limit
        space['card_limit'] = card_limit
        space['use_data_properties'] = use_data_properties
        space['value_splitter'] = value_splitter
        space['quality_func'] = quality_func
        # call the wrapper class
        wrap_obj = EvoLearnerWrapper(knowledge_base=kb, max_runtime=max_runtime, tournament_size=tournament_size,
                                     height_limit=height_limit, card_limit=card_limit, 
                                     use_data_properties=use_data_properties,
                                     quality_func=quality_func,
                                     value_splitter=value_splitter)
        model = wrap_obj.get_evolearner_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        quality = hypotheses[0].quality
        space['quality_score'] = quality
        self.write_to_df(**space)
        return quality    

    def get_best_optimisation_result_for_random_sampler(self, n_trials):
        self.study_random_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_random_sampler.best_trial)
        # best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_random_sampler.best_params)
        # score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_random_sampler.best_value)

    def get_best_optimization_result_for_tpe_sampler(self, n_trials):
        self.study_tpe_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_tpe_sampler.best_trial)
        # best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_tpe_sampler.best_params)
        # score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_tpe_sampler.best_value)

    def get_best_optimization_result_for_cmes_sampler(self, n_trials):
        self.study_cmaes_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_cmaes_sampler.best_trial)
        # best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_cmaes_sampler.best_params)
        # score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_cmaes_sampler.best_value)

    def get_best_optimization_result_for_nsgii_sampler(self, population_size):
        self.study_nsgaii_sampler.optimize(self.objective, n_trials=population_size)
        print("-------BEST TRIAL-----------")
        print(self.study_nsgaii_sampler.best_trial)
        # best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_nsgaii_sampler.best_params)
        # score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_nsgaii_sampler.best_value)

    def get_best_optimization_result_for_qmc_sampler(self, n_trials):
        self.study_qmc_sampler.optimize(self.objective, n_trials=n_trials)
        print("-------BEST TRIAL-----------")
        print(self.study_qmc_sampler.best_trial)
        # best parameter combination
        print("-------BEST PARAMS----------")
        print(self.study_qmc_sampler.best_params)
        # score achieved with best parameter combination
        print("-------BEST Value----------")
        print(self.study_qmc_sampler.best_value)


if __name__ == "__main__":
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))
        
        # shuffle the Positive and Negative Sample
        shuffle(typed_pos)
        shuffle(typed_neg)

        # Split the data into Training Set and Test Set
        train_pos = set(typed_pos[:int(len(typed_pos)*0.8)])
        train_neg = set(typed_neg[:int(len(typed_neg)*0.8)])
        test_pos = set(typed_pos[-int(len(typed_pos)*0.2):])
        test_neg = set(typed_neg[-int(len(typed_neg)*0.2):])
        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        # create class object and get the optimised result
        optuna1 = OptunaSamplers(lp, str_target_concept)
        optuna1.get_best_optimization_result_for_tpe_sampler(10)
        optuna1.convert_to_csv(df)