import optuna
import json
import os
from random import shuffle
from optuna.samplers import RandomSampler

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

def get_optimised_scores(lp):
    def objective(trial):
        max_runtime = trial.suggest_int("max_runtime", 10, 20)
        tournament_size = trial.suggest_int("tournament_size", 2, 10)
        model = EvoLearner(knowledge_base=kb, max_runtime=max_runtime, tournament_size=tournament_size)
        model.fit(lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))      
        hypotheses = list(model.best_hypotheses(n=1))   
        quality = hypotheses[0].quality
        return quality

    study = optuna.create_study(sampler=RandomSampler())
    print("Quality:",study)
    study.optimize(objective, n_trials=10)
    print(study.best_trial)

    #best parameter combination
    study.best_params

    #score achieved with best parameter combination
    study.best_value    

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
        get_optimised_scores(lp)