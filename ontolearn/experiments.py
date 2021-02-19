from typing import List, Tuple, Set, Dict, Any, Iterable
import numpy as np
import json
from sklearn.model_selection import KFold

import time


class Experiments:
    def __init__(self, max_test_time_per_concept=3):
        self.random_state_k_fold = 1
        self.max_test_time_per_concept = max_test_time_per_concept

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

    def start_KFold(self, k=None, dataset: List[Tuple[str, Set, Set]] = None, models: Iterable = None):
        """
        Perform KFold cross validation
        @param models:
        @param k:
        @param dataset: A list of tuples where a tuple (i,j,k) where i denotes the target concept
        j denotes the set of positive examples and k denotes the set of negative examples.
        @return:
        """
        models = {i for i in models}
        assert len(models) > 0
        assert len(dataset) > 0
        assert isinstance(dataset[0], tuple)
        assert isinstance(dataset[0], tuple)
        assert k
        dataset = np.array(dataset)  # due to indexing feature required in the sklearn.KFold.

        kf = KFold(n_splits=k, random_state=self.random_state_k_fold)

        results = dict()
        counter = 1
        for train_index, test_index in kf.split(dataset):
            train, test = dataset[train_index].tolist(), dataset[test_index].tolist()
            print(f'##### FOLD:{counter} #####')
            start_time_fold = time.time()
            for m in models:
                m.train(train)
                test_report: List[dict] = m.fit_from_iterable(test, max_runtime=self.max_test_time_per_concept)
                str_report, dict_report = self.store_report(m, test, test_report)
                print(str_report)
                results.setdefault(m.name, []).append((counter, dict_report))
            print(f'##### FOLD:{counter} took {round(time.time() - start_time_fold)} seconds #####')
            counter += 1

        self.report_results(results)

    @staticmethod
    def report_results(k_fold_cross_validation):
        print('\n##### K-FOLD CROSS VALUATION RESULTS #####')
        for learner_name, v in k_fold_cross_validation.items():
            r=np.array([[report['F-measure'],report['Accuracy'],report['Runtime']] for (fold, report) in v])
            f1_mean, f1_std = r[:,0].mean(), r[:,0].std()
            acc_mean, acc_std = r[:,1].mean(), r[:,1].std()
            runtime_mean, runtime_std = r[:,2].mean(), r[:,2].std()
            print(
                f'{learner_name}\t F-measure:(avg.{f1_mean:.2f} | std.{f1_std:.2f})\tAccuracy:(avg.{acc_mean:.2f} | std.{acc_std:.2f})\tRuntime:(avg.{runtime_mean:.2f} | std.{runtime_std:.2f})')
