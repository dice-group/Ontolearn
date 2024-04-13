"""
====================================================================
Drill -- Neuro-Symbolic Class Expression Learning

# Learn Embeddings
dicee --path_single_kg KGs/Family/family-benchmark_rich_background.owl --path_to_store_single_run embeddings --backend rdflib --save_embeddings_as_csv --model Keci --num_epoch 10


====================================================================
"""
import json
from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import StratifiedKFold
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import Drill
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer


def start(args):
    kb = KnowledgeBase(path=args.path_knowledge_base)
    drill = Drill(knowledge_base=kb,
                  path_embeddings=args.path_embeddings,
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                  quality_func=F1(),
                  reward_func=CeloeBasedReward(),
                  epsilon_decay=args.epsilon_decay,
                  learning_rate=args.learning_rate,
                  num_of_sequential_actions=args.num_of_sequential_actions,
                  num_episode=args.num_episode,
                  iter_bound=args.iter_bound,
                  max_runtime=args.max_runtime)

    if args.path_pretrained_dir:
        drill.load(directory=args.path_pretrained_dir)
    else:
        drill.train(num_of_target_concepts=args.num_of_target_concepts,
                    num_learning_problems=args.num_of_training_learning_problems)
        drill.save(directory="pretrained_drill")

    with open(args.path_learning_problem) as json_file:
        examples = json.load(json_file)
    p = examples['positive_examples']
    n = examples['negative_examples']

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
    X = np.array(p + n)
    Y = np.array([1.0 for _ in p] + [0.0 for _ in n])
    dl_render = DLSyntaxObjectRenderer()
    for (ith, (train_index, test_index)) in enumerate(kf.split(X, Y)):
        train_pos = {pos_individual for pos_individual in X[train_index][Y[train_index] == 1]}
        train_neg = {neg_individual for neg_individual in X[train_index][Y[train_index] == 0]}
        test_pos = {pos_individual for pos_individual in X[test_index][Y[test_index] == 1]}
        test_neg = {neg_individual for neg_individual in X[test_index][Y[test_index] == 0]}
        train_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, train_pos))),
                                    neg=set(map(OWLNamedIndividual, map(IRI.create, train_neg))))

        test_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, test_pos))),
                                   neg=set(map(OWLNamedIndividual, map(IRI.create, test_neg))))

        pred_drill = drill.fit(train_lp).best_hypotheses()
        train_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                          pos=train_lp.pos,
                                          neg=train_lp.neg)
        # () Quality on test data
        test_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                         pos=test_lp.pos,
                                         neg=test_lp.neg)
        print(
            f"Prediction: {dl_render.render(pred_drill)} | Train Quality: {train_f1_drill:.3f} | Test Quality: {test_f1_drill:.3f} \n")


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='../KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_embeddings", type=str,
                        default='../embeddings/Keci_entity_embeddings.csv')
    parser.add_argument("--num_of_target_concepts",
                        type=int,
                        default=1)
    parser.add_argument("--num_of_training_learning_problems",
                        type=int,
                        default=1)
    parser.add_argument("--path_pretrained_dir", type=str, default=None)

    parser.add_argument("--path_learning_problem", type=str, default='uncle_lp2.json',
                        help="Path to a .json file that contains 2 properties 'positive_examples' and "
                             "'negative_examples'. Each of this properties should contain the IRIs of the respective"
                             "instances. e.g. 'some/path/lp.json'")
    parser.add_argument("--max_runtime", type=int, default=10, help="Max runtime")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds of cross validation.")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    # DQL related
    parser.add_argument("--num_episode", type=int, default=1, help='Number of trajectories created for a given lp.')

    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=2,
                        help='Number of epochs on experience replay memory')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')

    # NN related
    parser.add_argument("--learning_rate", type=int, default=.01)

    start(parser.parse_args())
