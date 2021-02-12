from typing import List, Tuple, Set,Dict,Any
import numpy as np
import json
from sklearn.model_selection import KFold

import time

class Experiments:
    def __init__(self):
        self.random_state_k_fold = 1
        self.max_run_time = 3

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
    def store_report(model, learning_problems: List[List], predictions: List[dict]) -> Tuple[str, Dict[str, Any]]:
        """

        @param model: concept learner
        @param learning_problems: A list of learning problems (lps) where lp corresponds to [target concept, positive
        and negative examples, respectively.
        @param predictions: A list of predictions (preds) where
        pred => { 'Prediction': str, 'F-measure': float, 'Accuracy', 'Runtime':float}
        @return:
        """
        assert len(learning_problems) == len(predictions)
        assert isinstance(learning_problems, list) and isinstance(learning_problems[0], list)
        assert isinstance(predictions, list) and isinstance(predictions[0], dict)

        store_json = dict()
        for (th, lp, pred) in zip(range(len(learning_problems)), learning_problems, predictions):
            report = dict()
            report['TargetConcept'] = lp[0]
            report['Positives'], report['Negatives'] = list(lp[1]), list(lp[2])  # 'set' is not JSON serializable.
            report.update(pred)
            store_json[th] = report

        # json serialize
        with open(model.storage_path + '/classification_reports.json', 'w') as file_descriptor:
            json.dump(store_json, file_descriptor, indent=3)

        del store_json

        # json serialize
        with open(model.storage_path + '/classification_reports.json', 'r') as read_file:
            report = json.load(read_file)
        array_res = np.array([[v['F-measure'], v['Accuracy'], v['Runtime']] for k, v in report.items()])
        f1, acc, time = array_res[:, 0], array_res[:, 1], array_res[:, 2]
        del array_res
        m = '{}\t F-measure:(avg.{:.2f} | std.{:.2f})\tAccuracy:(avg.{:.2f} | std.{:.2f})\t' \
            'Runtime:(avg.{:.2f} | std.{:.2f})'.format(model.name,
                                                       f1.mean(), f1.std(),
                                                       acc.mean(),
                                                       acc.std(),
                                                       time.mean(), time.std())
        return m, {'F-measure': f1, 'Accuracy': acc, 'Runtime': time}

    def start_KFold(self, k=None, dataset: List[Tuple[str, Set, Set]] = None, models: List = None,
                    max_runtime_per_problem=3) -> dict:
        """
        Perform KFold cross validation
        @param max_runtime_per_problem:
        @param models:
        @param k:
        @param dataset: A list of tuples where a tuple (i,j,k) where i denotes the target concept
        j denotes the set of positive examples and k denotes the set of negative examples.
        @param max_runtime: in seconds.
        @return:
        """
        assert len(models) > 0
        assert len(dataset) > 0
        assert isinstance(dataset[0], tuple)
        assert isinstance(dataset[0], tuple)
        assert k
        assert isinstance(max_runtime_per_problem, int)
        dataset = np.array(dataset)  # due to indexing feature required in the sklearn.KFold.

        kf = KFold(n_splits=k, random_state=self.random_state_k_fold)

        k_fold_summary = dict()
        results = dict()
        counter = 1
        for train_index, test_index in kf.split(dataset):
            train, test = dataset[train_index].tolist(), dataset[train_index].tolist()
            fold = dict()
            # one could even parallelize the following computation.
            print(f'##### FOLD:{counter} #####')
            start_time_fold = time.time()
            for m in models:
                m.train(train)
                test_report: List[dict] = m.fit_from_iterable(test, max_runtime=max_runtime_per_problem)
                stats_report, dict_report = self.store_report(m, test, test_report)
                print(stats_report)
                # Store stats
                fold.update({m.name: dict_report})
                k_fold_summary.setdefault(m.name, []).append(dict_report)

            print(f'##### FOLD:{counter} took {round(time.time() - start_time_fold)} seconds #####')

            results[counter] = fold
            counter += 1

        return_results = dict()
        for mode_name, stats in k_fold_summary.items():
            test_stats = np.array([[i['F-measure'], i['Accuracy'], i['Runtime']] for i in stats])
            return_results[mode_name] = {'F-measure': test_stats[:, 0],
                                         'Accuracy': test_stats[:, 1],
                                         'Runtime': test_stats[:, 2]}

        return return_results