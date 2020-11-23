from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillSample, DrillAverage
import json
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from ontolearn.binders import DLLearnerBinder
from typing import List, Tuple, Set, AnyStr


class Experiments:
    def __init__(self):
        """
        Initialize.
        """

    def run_dl(self, dataset, max_run_time: int = None):
        """

        @param max_run_time:
        @param dataset:
        @return:
        """
        if max_run_time:
            self.max_run_time = max_run_time
        # Create Config file

        for (target_concept, positives, negatives) in dataset:

            for algorithm in ['ocel', 'eltl',
                              'celoe']:  # Although accuracyMethod.type = fmeasure, ocel reports accuracy.

                # @todos: create file to store .conf files.
                str_best_concept, f_1score = self.dl_leaner.pipeline(
                    knowledge_base_path=self.kb_path,
                    algorithm=algorithm,
                    positives=positives,
                    negatives=negatives,
                    path_name=target_concept,
                    max_run_time=self.max_run_time)

                # print('JAVA:{0}:BestPrediction:\t{1}\tF-score:{2}'.format(
                #    algorithm, str_best_concept, f_1score))

    @staticmethod
    def start_KFold(k=None, dataset: List[Tuple[AnyStr, Set, Set]] = None, models: List = None, max_run_time=3):
        """
        Perform KFold cross validation
        @param models:
        @param k:
        @param dataset: A list of tuples where a tuple (i,j,k) where i denotes the target concept
        j denotes the set of positive examples and k denotes the set of negative examples.
        @param max_run_time: in seconds.
        @return:
        """
        assert len(models) > 0
        assert len(dataset) > 0
        assert isinstance(dataset[0], tuple)
        assert isinstance(dataset[0], tuple)
        assert k
        assert isinstance(max_run_time, int)
        dataset = np.array(dataset)  # due to indexing feature required in the sklearn.KFold.

        kf = KFold(n_splits=k, random_state=1)

        for train_index, test_index in kf.split(dataset):
            train, test = dataset[train_index].tolist(), dataset[train_index].tolist()
            # one could even parallelize the following computation.
            for i in models:
                i.train(train)
                test_report = i.test(test, max_run_time=max_run_time)
                for res in test_report:
                    print('TARGET CONCEPT:', res['TargetConcept'])
                    print('Prediction:', res['Prediction'])
                f1_scores = np.array([i['Quality'] for i in test_report])  # Quality of Predictions.
                print('Mean:{0:.2f}+std:{1:.2f} of F1-scores in predictions'.format(f1_scores.mean(),f1_scores.std()))
                exit(1)

PATH_FAMILY = '/home/demir/Desktop/Onto-learn_dev/data/family-benchmark_rich_background.owl'
family_embeddings_path = '../embeddings/dismult_family_benchmark/instance_emb.csv'
dllerner_path = '/home/demir/Desktop/DL/dllearner-1.4.0/'

kb = KnowledgeBase(PATH_FAMILY)
emb = pd.read_csv(family_embeddings_path, index_col=0)

balanced_examples = LearningProblemGenerator(knowledge_base=kb, num_problems=10, min_num_ind=15).balanced_examples

# Initialize concept learners.
celoe = DLLearnerBinder(path=dllerner_path, model='CELOE')
ocel = DLLearnerBinder(path=dllerner_path, model='OCEL')
eltl = DLLearnerBinder(path=dllerner_path, model='ELTL')

drill_average = DrillAverage(knowledge_base=kb, instance_embeddings=emb, num_episode=1)
drill_sample = DrillSample(knowledge_base=kb, instance_embeddings=emb, num_episode=1)

Experiments().start_KFold(k=10, dataset=balanced_examples,
                          models=[drill_average, drill_sample, celoe, ocel, eltl],
                          max_run_time=10)
