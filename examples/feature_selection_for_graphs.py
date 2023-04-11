import re
import json
import os
import time
from random import shuffle
from owlready2 import get_ontology, default_world, destroy_entity
import logging

from examples.search import calc_prediction
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.concept_learner import EvoLearner
from owlapy.render import DLSyntaxObjectRenderer

LOG_FILE = 'featureSelectionWithGraph.log'
DATASET = 'pyrimidine'

logging.basicConfig(filename=LOG_FILE,
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    logging.error(FileNotFoundError)
    pass

with open(f'dataset/{DATASET}.json') as json_file:
    settings = json.load(json_file)


def get_data_properties(onto):
    try:
        data_properties = onto.data_properties()
    except Exception as e:
        data_properties = None
        logging.error(e)
    return data_properties


def get_object_properties(onto):
    try:
        object_properties = onto.object_properties()
    except Exception as e:
        object_properties = None
        logging.error(e)
    return object_properties

def create_new_kb(feature):
    for prop in feature:
        destroy_entity(prop)
    onto.save("newkb.owl")
    return KnowledgeBase(path="./newkb.owl")


def get_top_k_concepts_from_evolearner(lp, k=3):
    model = EvoLearner(knowledge_base=kb)
    model.fit(lp)
    model.save_best_hypothesis(n=5, path=f'Predictions_{str_target_concept}')
    hypotheses = list(model.best_hypotheses(n=5))

    with open(f'{DATASET}_featureSelectionWithGraph.txt', 'a') as f:
        print('K Top Feature:', k, file=f)

    dlr = DLSyntaxObjectRenderer()
    concept_sorted = [dlr.render(c.concept) for c in hypotheses]
    properties_relation = {}

    for concepts in concept_sorted:
        concept_sorted = re.sub('[⊓∃⊔!?(.)]', " ", concepts)
        concept_sorted_prop = concept_sorted.split(" ")
        concept_sorted_prop = [i for i in concept_sorted_prop if i != ""]
        relationships = [x for x in concept_sorted_prop if x[0].islower()]
        for relations in relationships:
            if relations not in properties_relation:
                properties_relation[relations] = 1
            else:
                value = properties_relation[relations]
                properties_relation[relations] = value + 1

    object_properties = []
    if len(properties_relation) <= k:
        object_properties = list(properties_relation.keys())
    else:
        # Get a list of key-value pair sorted by value
        sorted_obj_prop = sorted(properties_relation.items(), key=lambda x: x[1], reverse=True)
        object_properties = [x[0] for x in sorted_obj_prop[:k]]

    # object_prop = [escape(obj_prop.get_iri().get_remainder()) for obj_prop in kb.get_object_properties()]
    prop_object = list(get_object_properties(onto)) + list(get_data_properties(onto))
    k_object_properties = [x for x in prop_object if x.name in object_properties]

    return k_object_properties


if __name__ == "__main__":
    onto = get_ontology(settings['data_path']).load()
    kb = KnowledgeBase(path=settings['data_path'])
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))

        # Shuffle the Positive and Negative Sample
        shuffle(typed_pos)
        shuffle(typed_neg)

        # Split the data into Training Set and Test Set
        train_pos = set(typed_pos[:int(len(typed_pos) * 0.8)])
        train_neg = set(typed_neg[:int(len(typed_neg) * 0.8)])
        test_pos = set(typed_pos[-int(len(typed_pos) * 0.2):])
        test_neg = set(typed_neg[-int(len(typed_neg) * 0.2):])
        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        st = time.time()
        relevant_prop = get_top_k_concepts_from_evolearner(lp, k=5)
        new_kb = create_new_kb(relevant_prop)
        model = EvoLearner(knowledge_base=new_kb)
        model.fit(lp)
        model.save_best_hypothesis(n=3, path=f'Predictions_{str_target_concept}')
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(test_pos | test_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality
        et = time.time()
        elapsed_time = et - st
        with open(f'{DATASET}_featureSelectionWithGraph.txt', 'a') as f:
            print('Concept', str_target_concept, file=f)
            print('F1 Score', f1_score[1], file=f)
            print('Accuracy', accuracy[1], file=f)
            print('Time Taken', elapsed_time, file=f)
