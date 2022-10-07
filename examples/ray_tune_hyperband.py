from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
import argparse
import time
import json
import os
from random import shuffle
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.tuner import Tuner

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)
kb = KnowledgeBase(path=settings['data_path'])

def evaluation_fn(str_target_concept, target_kb, lp, max_runtime, tournament_size):
    space = dict()
    space['max_runtime'] = max_runtime
    space['tournament_size'] = tournament_size           
    model = EvoLearner(knowledge_base=target_kb, quality_func=F1(), **space)
    model.fit(lp, verbose=False)
    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))      
    hypotheses = list(model.best_hypotheses(n=1))   
    quality = hypotheses[0].quality
    return quality


def easy_objective(config):
    # Hyperparameters
    max_runtime, tournamenet_size = config["max_runtime"], config["tournamenet_size"]        
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
        
        intermediate_score = evaluation_fn(str_target_concept,target_kb, lp, max_runtime, tournamenet_size)
        session.report({"iterations": str(str_target_concept), "F1_Score": intermediate_score})
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.",
        required=False,
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using Ray Client.",
    )
    args, _ = parser.parse_known_args()
    if args.server_address is not None:
        ray.init(f"ray://{args.server_address}")
    else:
        ray.init(address=args.ray_address)

    # AsyncHyperBand enables aggressive early stopping of bad trials.
    scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100)

    # 'training_iteration' is incremented every time `trainable.step` is called
    stopping_criteria = {"training_iteration": 1 if args.smoke_test else 9999}
    run_config=air.RunConfig(
        name="asynchyperband_test",
        stop=stopping_criteria,
        verbose=1,
    )
    
    tune_config=tune.TuneConfig(
            metric="mean_loss", mode="min", scheduler=scheduler, num_samples=20
    )
    param_space={  # Hyperparameter space
            "max_runtime": tune.randint(2, 500),
            "tournament_size": tune.randint(2, 20),
    }
    tune.with_resources(easy_objective, {"cpu": 1, "gpu": 0})
    tuner = tune.run(
        tune.with_resources(easy_objective, {"cpu": 1, "gpu": 0}),
        run_config=air.RunConfig(
            name="asynchyperband_test",
            stop=stopping_criteria,
            verbose=1,
        ),
        tune_config=tune.TuneConfig(
            metric="F1_Score", mode="max", scheduler=scheduler, num_samples=20
        ),
        param_space={  # Hyperparameter space
            "max_runtime": tune.randint(2, 500),
            "tournament_size": tune.randint(2, 20),
        },
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
