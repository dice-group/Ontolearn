import json
import os
from random import shuffle

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from ontolearn.metrics import F1
from sklearn.model_selection import ParameterGrid

setup_logging()

def grid_search(target_kb, space, lp):
    best_quality = None
    best_parameter = None

    for parameter_grid in space_grid:        
        model = EvoLearner(knowledge_base=target_kb, **parameter_grid)
        model.fit(lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))      
        hypotheses = list(model.best_hypotheses(n=1))      
        quality = hypotheses[0].quality
        
        if best_quality is None:
            best_quality = quality
            best_hyperparameter = parameter_grid
        elif best_quality <= quality:
            best_quality = quality
            best_parameter = parameter_grid

    return best_parameter            
    
try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])


# noinspection DuplicatedCode
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)

    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        NS = 'http://www.benchmark.org/family#'
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Sister')),
            OWLClass(IRI(NS, 'Daughter')),
            OWLClass(IRI(NS, 'Mother')),
            OWLClass(IRI(NS, 'Grandmother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'PersonWithASibling')),
            OWLClass(IRI(NS, 'Granddaughter')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    else:
        target_kb = kb
    
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

    #Create the grid space for hyper parameter tuning
    space = dict()
    space['max_runtime'] = [10, 500, 900, 1300]
    space['tournament_size'] = [2, 5, 7, 10]
    space_grid = list(ParameterGrid(space))
        
    best_hyperparameter = grid_search(target_kb, space_grid, lp)
    print("Best Hyper Parameter", best_hyperparameter)
   