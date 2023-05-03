import json
import os
import numpy as np
import time
import logging
from random import shuffle

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import OCEL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils import setup_logging
from owlapy.model import OWLClass, IRI, OWLNamedIndividual
from search import calc_prediction

setup_logging()
DIRECTORY = './OCELLOG/'
LOG_FILE = "ocel_without_hpo_fs.log"

DATASET = 'hepatitis'
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

LOG_FILE_PATH = os.path.join(DIRECTORY, LOG_FILE)
logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
try:
    os.chdir("examples")
except FileNotFoundError:
    pass


path_dataset = f'dataset/{DATASET}.json'
print('Path_To_Dataset', path_dataset)
with open(path_dataset) as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

# noinspection DuplicatedCode
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
                                            [int(len(typed_pos) * 0.6),
                                             int(len(typed_pos) * 0.8)])
    train_neg, val_neg, test_neg = np.split(typed_neg,
                                            [int(len(typed_neg) * 0.6),
                                             int(len(typed_neg) * 0.8)])
    train_pos, train_neg = set(train_pos), set(train_neg)
    val_pos, val_neg = set(val_pos), set(val_neg)
    test_pos, test_neg = set(test_pos), set(test_neg)

    lp = PosNegLPStandard(pos=train_pos, neg=train_neg)
    st = time.time()
    model = OCEL(knowledge_base=kb,
                 max_runtime=600,
                 max_num_of_concepts_tested=10_000_000_000,
                 iter_bound=10_000_000_000)
    model.fit(lp)
    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    hypotheses = model.best_hypotheses(n=1)
    hypotheses = [hypo for hypo in hypotheses]
    print(hypotheses)
    predictions = model.predict(individuals=list(test_pos | test_neg),
                                hypotheses=hypotheses)
    f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
    quality = hypotheses[0].quality
    et = time.time()
    elapsed_time = et - st
    with open(f'{DIRECTORY}{DATASET}.txt', 'a') as f:
        print('DATASET:', DATASET, file=f)
        print('F1 Score:', f1_score[1], file=f)
        print('Accuracy:', accuracy[1], file=f)
        print('Time Taken:', elapsed_time, file=f)