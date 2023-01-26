import optuna
import json
import os
from random import shuffle
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
import pandas as pd
import time
import logging
import numpy as np

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI
from wrapper_evolearner import EvoLearnerWrapper
from search import calc_prediction


LOG_FILE = "hpo.log"
DATASET = "lymphography"

logging.basicConfig(filename=LOG_FILE,
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    pass

path_dataset = f'dataset/{DATASET}.json'
with open(path_dataset) as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
df = pd.DataFrame(columns=['LP', 'max_runtime', 'tournament_size',
                           'height_limit', 'card_limit',
                           'use_data_properties', 'quality_func',
                           'use_inverse_prop', 'value_splitter',
                           'quality_score', 'Validation_f1_Score',
                           'Validation_accuracy'])


class OptunaSamplers():
    def __init__(self, lp, concept, val_pos, val_neg):
        # For grid sampler specify the search space
        search_space = {'max_runtime': [2, 10, 50, 100],
                        'tournament_size': [2, 5, 15],
                        'height_limit': [3, 10, 25],
                        'card_limit': [2, 5, 10]
                        }
        self.lp = lp
        self.concept = concept
        self.val_pos = val_pos
        self.val_neg = val_neg
        self.study_random_sampler = optuna.create_study(sampler=RandomSampler(),
                                                        direction='maximize')
        self.study_grid_sampler = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space=search_space))
        self.study_tpe_sampler = optuna.create_study(sampler=TPESampler(),
                                                     direction='maximize')
        self.sampler = optuna.samplers.CmaEsSampler()
        self.nsgii = optuna.samplers.NSGAIISampler(population_size=100)
        self.qmc_sampler = optuna.samplers.QMCSampler()
        self.study_cmaes_sampler = optuna.create_study(sampler=self.sampler)
        self.study_nsgaii_sampler = optuna.create_study(sampler=self.nsgii)
        self.study_qmc_sampler = optuna.create_study(sampler=self.qmc_sampler)

    def write_to_df(self, **space):
        df.loc[len(df.index)] = [self.concept, space['max_runtime'],
                                 space['tournament_size'], space['height_limit'],
                                 space['card_limit'], space['use_data_properties'],
                                 space['quality_func'], space['use_inverse'],
                                 space['value_splitter'], space['quality_score'],
                                 space['Validation_f1_Score'], space['Validation_accuracy']]

    def convert_to_csv(self, df):
        timestr = str(time.strftime("%Y%m%d-%H%M%S"))
        filename = f'{DATASET}_Output {timestr}'
        df.to_csv(filename+".csv", index=False)

    def objective(self, trial):
        max_runtime = trial.suggest_int("max_runtime", 2, 100)
        tournament_size = trial.suggest_int("tournament_size", 2, 16)
        height_limit = trial.suggest_int('height_limit', 3, 26)
        card_limit = trial.suggest_int('card_limit', 1, 10)

        # call the wrapper class
        wrap_obj = EvoLearnerWrapper(knowledge_base=kb,
                                     max_runtime=max_runtime,
                                     tournament_size=tournament_size,
                                     height_limit=height_limit,
                                     card_limit=card_limit)
        model = wrap_obj.get_evolearner_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality
        return quality

    def objective_with_categorical_distribution(self, trial):
        # categorical distribution efficiently works for the following samplers
        # Random Sampler, Grid Sampler, TPE Sampler and NSGAII Sampler
        max_runtime = trial.suggest_int("max_runtime", 2, 20)
        tournament_size = trial.suggest_int("tournament_size", 2, 10)
        height_limit = trial.suggest_int('height_limit', 3, 25)
        card_limit = trial.suggest_int('card_limit', 5, 10)
        use_data_properties = trial.suggest_categorical('use_data_properties',
                                                        ['True', 'False'])
        use_inverse = trial.suggest_categorical('use_inverse', ['True', 'False'])
        quality_func = trial.suggest_categorical('quality_func', ['F1', 'Accuracy'])
        value_splitter = trial.suggest_categorical('value_splitter',
                                                   ['binning_value_splitter',
                                                    'entropy_value_splitter'])
        # call the wrapper class
        wrap_obj = EvoLearnerWrapper(knowledge_base=kb,
                                     max_runtime=max_runtime,
                                     tournament_size=tournament_size,
                                     height_limit=height_limit,
                                     card_limit=card_limit,
                                     use_data_properties=use_data_properties,
                                     use_inverse=use_inverse,
                                     quality_func=quality_func,
                                     value_splitter=value_splitter)
        model = wrap_obj.get_evolearner_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality

        # create a dictionary
        space = dict()
        space['max_runtime'] = max_runtime
        space['tournament_size'] = tournament_size
        space['height_limit'] = height_limit
        space['card_limit'] = card_limit
        space['use_data_properties'] = use_data_properties
        space['use_inverse'] = use_inverse
        space['value_splitter'] = value_splitter
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.write_to_df(**space)
        return quality

    def objective_without_categorical_distribution(self, trial):
        # Categorical distribution does not work for QMC and CMEs sampler
        # The categorical distribution values are sampled in such cases
        # using Random Sampler
        # Using a threshold value to categorise the hyperparameters.
        max_runtime = trial.suggest_int("max_runtime", 2, 20)
        tournament_size = trial.suggest_int("tournament_size", 2, 10)
        height_limit = trial.suggest_int('height_limit', 3, 25)
        card_limit = trial.suggest_int('card_limit', 5, 10)
        use_data_properties = trial.suggest_int('use_data_properties', 1, 2)
        use_inverse = trial.suggest_int('use_inverse', 1, 2)
        quality_func = trial.suggest_int('quality_func', 1, 2)
        value_splitter = trial.suggest_int('value_splitter', 1, 2)

        use_data_properties = 'True' if use_data_properties >= 2 else 'False'
        use_inverse = 'True' if use_inverse >= 2 else 'False'
        quality_func = 'F1' if quality_func >= 2 else 'Accuracy'
        value_splitter = 'binning_value_splitter' if value_splitter >= 2 else 'entropy_value_splitter'

        # call the wrapper class
        wrap_obj = EvoLearnerWrapper(knowledge_base=kb,
                                     max_runtime=max_runtime,
                                     tournament_size=tournament_size,
                                     height_limit=height_limit,
                                     card_limit=card_limit,
                                     use_data_properties=use_data_properties,
                                     use_inverse=use_inverse,
                                     quality_func=quality_func,
                                     value_splitter=value_splitter)
        model = wrap_obj.get_evolearner_model()
        model.fit(self.lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(self.val_pos | self.val_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, self.val_pos, self.val_neg)
        quality = hypotheses[0].quality

        # create a dictionary
        space = dict()
        space['max_runtime'] = max_runtime
        space['tournament_size'] = tournament_size
        space['height_limit'] = height_limit
        space['card_limit'] = card_limit
        space['use_data_properties'] = use_data_properties
        space['use_inverse'] = use_inverse
        space['value_splitter'] = value_splitter
        space['quality_func'] = quality_func
        space['quality_score'] = quality
        space['Validation_f1_Score'] = f1_score[1]
        space['Validation_accuracy'] = accuracy[1]
        self.write_to_df(**space)
        return quality

    def get_best_optimisation_result_for_random_sampler(self, n_trials):
        self.study_random_sampler.optimize(self.objective_with_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL RANDOM SAMPLER : {self.study_random_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS RANDOM SAMPLER : {self.study_random_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE RANDOM SAMPLER : {self.study_random_sampler.best_value}")

    def get_best_optimization_result_for_grid_sampler(self, n_trials):
        self.study_grid_sampler.optimize(self.objective, n_trials=n_trials)
        logging.info(f"BEST TRIAL GRID SAMPLER : {self.study_grid_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS GRID SAMPLER : {self.study_grid_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE GRID SAMPLER : {self.study_grid_sampler.best_value}")

    def get_best_optimization_result_for_tpe_sampler(self, n_trials):
        self.study_tpe_sampler.optimize(self.objective_with_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL TPE SAMPLER : {self.study_tpe_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS TPE SAMPLER : {self.study_tpe_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE TPE SAMPLER : {self.study_tpe_sampler.best_value}")

    def get_best_optimization_result_for_cmes_sampler(self, n_trials):
        self.study_cmaes_sampler.optimize(self.objective_with_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL CMAES SAMPLER : {self.study_cmaes_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS CMAES SAMPLER : {self.study_cmaes_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE CMAES SAMPLER : {self.study_cmaes_sampler.best_value}")

    def get_best_optimization_result_for_nsgii_sampler(self, population_size):
        self.study_nsgaii_sampler.optimize(self.objective_with_categorical_distribution, n_trials=population_size)
        logging.info(f"BEST TRIAL NSGAII SAMPLER : {self.study_nsgaii_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS NSGAII SAMPLER : {self.study_nsgaii_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE NSGAII SAMPLER : {self.study_nsgaii_sampler.best_value}")

    def get_best_optimization_result_for_qmc_sampler(self, n_trials):
        self.study_qmc_sampler.optimize(self.objective_without_categorical_distribution, n_trials=n_trials)
        logging.info(f"BEST TRIAL QMC SAMPLER : {self.study_qmc_sampler.best_trial}")
        # best parameter combination
        logging.info(f"BEST PARAMS QMC SAMPLER : {self.study_qmc_sampler.best_params}")
        # score achieved with best parameter combination
        logging.info(f"BEST VALUE QMC SAMPLER : {self.study_qmc_sampler.best_value}")


if __name__ == "__main__":
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))

        # shuffle the Positive and Negative Sample
        shuffle(typed_pos)
        shuffle(typed_neg)

        # Split the data into Training Set, Validation Set and Test Set
        train_pos, val_pos, test_pos = np.split(typed_pos,
                                                [int(len(typed_pos)*0.6),
                                                 int(len(typed_pos)*0.8)])
        train_neg, val_neg, test_neg = np.split(typed_neg,
                                                [int(len(typed_neg)*0.6),
                                                 int(len(typed_neg)*0.8)])
        train_pos, train_neg = set(train_pos), set(train_neg)
        val_pos, val_neg = set(val_pos), set(val_neg)
        test_pos, test_neg = set(test_pos), set(test_neg)

        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        # create class object and get the optimised result
        optuna1 = OptunaSamplers(lp, str_target_concept, val_pos, val_neg)
        optuna1.get_best_optimization_result_for_tpe_sampler(2)
        optuna1.convert_to_csv(df)

        # get the best hpo
        best_hpo = df.loc[df['Validation_f1_Score'] == df['Validation_f1_Score'].values.max()]
        if len(best_hpo.index) > 1:
            best_hpo = best_hpo.loc[(best_hpo['Validation_accuracy'] == best_hpo['Validation_accuracy'].values.max()) &
                                    (best_hpo['max_runtime'] == best_hpo['max_runtime'].values.min())]
        logging.info(f"BEST HPO : {best_hpo}")

        wrap_obj = EvoLearnerWrapper(knowledge_base=kb,
                                     max_runtime=int(best_hpo['max_runtime'].values[0]),
                                     tournament_size=int(best_hpo['tournament_size'].values[0]),
                                     height_limit=int(best_hpo['height_limit'].values[0]),
                                     card_limit=int(best_hpo['card_limit'].values[0]),
                                     use_data_properties=str(best_hpo['use_data_properties'].values[0]),
                                     use_inverse=str(best_hpo['use_inverse_prop'].values[0]),
                                     quality_func=str(best_hpo['quality_func'].values[0]),
                                     value_splitter=str(best_hpo['value_splitter'].values[0]))
        model = wrap_obj.get_evolearner_model()
        model.fit(lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.
                                   format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(test_pos | test_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality

        with open(f'{DATASET}.txt', 'a') as f:
            print('F1 Score', f1_score[1], file=f)
            print('Accuracy', accuracy[1], file=f)
