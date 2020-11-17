from ontolearn import KnowledgeBase, LearningProblemGenerator, DrillSample, DrillAverage
import json
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from ontolearn.binders import DLLearnerBinder


class Experiments:
    def __init__(self, path_of_kb, path_of_emb, dataset, max_run_time):
        self.kb_path = path_of_kb
        self.kb = KnowledgeBase(self.kb_path)
        self.dataset = np.array(dataset)
        self.instance_embeddings = pd.read_csv(path_of_emb, index_col=0)
        self.max_run_time = max_run_time
        assert set(self.kb.uri_individuals) == set(self.instance_embeddings.index.tolist())
        self.dl_leaner = DLLearnerBinder(path='/home/demir/Desktop/DL/dllearner-1.4.0/')

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

            for algorithm in ['ocel','eltl', 'celoe']: # Although accuracyMethod.type = fmeasure, ocel reports accuracy.

                # @todos: create file to store .conf files.
                str_best_concept, f_1score = self.dl_leaner.pipeline(
                    knowledge_base_path=self.kb_path,
                    algorithm=algorithm,
                    positives=positives,
                    negatives=negatives,
                    path_name=target_concept,
                    max_run_time=self.max_run_time)

                #print('JAVA:{0}:BestPrediction:\t{1}\tF-score:{2}'.format(
                #    algorithm, str_best_concept, f_1score))

    def start(self):
        kf = KFold(n_splits=10, random_state=1)

        model_sub = DrillSample(knowledge_base=self.kb, instance_embeddings=self.instance_embeddings)

        for train_index, test_index in kf.split(self.dataset):
            train, test = self.dataset[train_index].tolist(), self.dataset[train_index].tolist()
            # model_sub.train(train)
            print('testing starts')
            res = model_sub.test(test, max_run_time=self.max_run_time)
            f1_scores = np.array([i['Quality'] for i in res])
            print(f1_scores.mean())

            self.run_dl(test, max_run_time=self.max_run_time)


PATH_FAMILY = '/home/demir/Desktop/Onto-learn_dev/data/family-benchmark_rich_background.owl'
family_embeddings_path = '/home/demir/Desktop/Onto-learn_dev/embeddings/Dismult_family_benchmark/instance_emb.csv'

balanced_examples = LearningProblemGenerator(knowledge_base=KnowledgeBase(PATH_FAMILY),
                                             num_problems=10, min_num_ind=15).balanced_examples

exp = Experiments(path_of_kb=PATH_FAMILY,
                  path_of_emb=family_embeddings_path,
                  max_run_time=10,
                  dataset=balanced_examples)
exp.start()
