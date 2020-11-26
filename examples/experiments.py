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
    def start_KFold(k=None, dataset: List[Tuple[AnyStr, Set, Set]] = None, models: List = None, max_runtime=3):
        """
        Perform KFold cross validation
        @param max_runtime:
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
        assert isinstance(max_runtime, int)
        dataset = np.array(dataset)  # due to indexing feature required in the sklearn.KFold.

        kf = KFold(n_splits=k, random_state=1)

        for train_index, test_index in kf.split(dataset):
            train, test = dataset[train_index].tolist(), dataset[train_index].tolist()
            # one could even parallelize the following computation.
            for m in models:
                m.train(train)
                test_report = m.fit_from_iterable(test, max_runtime=max_runtime)
                f1_scores = np.array([i['F-measure'] for i in test_report])
                accuracy = np.array([i['Accuracy'] for i in test_report])
                runtime = np.array([i['Runtime'] for i in test_report])

                print('{}\t F-measure:(avg.{:.2f} | std.{:.2f})\t'
                      'Accuracy:(avg.{:.2f} | std.{:.2f})\t'
                      'Runtime:(avg.{:.2f} | std.{:.2f})'.format(m.name,
                                                                 f1_scores.mean(), f1_scores.std(),
                                                                 accuracy.mean(),
                                                                 accuracy.std(),
                                                                 runtime.mean(), runtime.std()))


PATH_FAMILY = '/home/demir/Desktop/Onto-learn_dev/data/family-benchmark_rich_background.owl'
family_embeddings_path = '../embeddings/dismult_family_benchmark/instance_emb.csv'
dl_learner_path = '/home/demir/Desktop/DL/dllearner-1.4.0/'

kb = KnowledgeBase(PATH_FAMILY)
emb = pd.read_csv(family_embeddings_path, index_col=0)

balanced_examples = LearningProblemGenerator(knowledge_base=kb, num_problems=2, min_num_ind=15).balanced_examples

# Initialize concept learners.
celoe = DLLearnerBinder(binary_path=dl_learner_path, kb_path=PATH_FAMILY, model='celoe')
ocel = DLLearnerBinder(binary_path=dl_learner_path, kb_path=PATH_FAMILY, model='ocel')
eltl = DLLearnerBinder(binary_path=dl_learner_path, kb_path=PATH_FAMILY, model='eltl')

drill_average = DrillAverage(knowledge_base=kb, instance_embeddings=emb, num_episode=1)
drill_sample = DrillSample(knowledge_base=kb, instance_embeddings=emb, num_episode=1)

Experiments().start_KFold(k=2, dataset=balanced_examples,
                          models=[drill_average, drill_sample, celoe, ocel, eltl], max_runtime=1)
