from typing import Tuple
import numpy as np
from sklearn import preprocessing
from typing import List
import random
from .metrics import F1
from .util import create_experiment_folder, create_logger


class Data:
    """
    A class for constructing supervised learning problem.
    """

    def __init__(self, knowledge_base, logger=None):
        self.kb = knowledge_base
        if logger:
            self.logger = logger
        else:
            self.storage_path, _ = create_experiment_folder(folder_name='../Log')
            self.logger = create_logger(name='Data', p=self.storage_path)

        self.individuals = list(self.kb.thing.instances)

        self.num_individuals = len(self.individuals)

        self.labels = None

        self.num_of_outputs = None
        self.lb = preprocessing.LabelBinarizer()
        self.all_concepts = None
        self.indx = dict(zip(self.kb.thing.instances, list(range(len(self.kb.thing.instances)))))

        self.quality_func = F1()

    def pos_neg_sampling_from_concept(self, c, number):
        x_pos = random.sample(c.instances, number // 2)
        x_neg = random.sample(self.kb.thing.instances - c.instances,
                              number // 2)
        return x_pos, x_neg

    def generate_data(self, **params):
        """

        @param params:
        @return:
        """

        # Define the learning problem
        X = []
        y = []

        self.all_concepts, kw = self.generate_concepts(**params)

        params.update(kw)

        # Important decision:
        self.labels = random.sample(self.all_concepts, params['num_of_outputs'])

        params['num_of_outputs'] = len(self.labels)
        # Generate Training Data
        for _ in range(params['num_of_times_sample_per_concept']):
            for c in self.all_concepts:
                try:
                    x_pos, x_neg = self.pos_neg_sampling_from_concept(c, params['num_of_inputs_for_model'])
                except ValueError:
                    # self.logger.info('During training,{0} is ignored due due to number of instances {1}.'
                    #                 ''.format(c.str,len(c.instances)))
                    continue
                vec_of_f_scores = self.score_with_labels(pos=x_pos, neg=x_neg, labels=self.labels)
                y.append(vec_of_f_scores)
                X.append([self.indx[i] for i in x_pos + x_neg])

        assert len(X) == len(y)
        return X, y, params

    def concepts_for_training(self, m):
        self.labels = np.array(m)

    def score(self, *, pos, neg):
        assert isinstance(pos, list)
        assert isinstance(neg, list)

        pos = set(pos)
        neg = set(neg)

        y = []
        for j in self.labels:  # training data corresponds to number of outputs.
            individuals = {self.individuals.index(j) for j in j.instances}

            tp = len(pos.intersection(individuals))
            tn = len(neg.difference(individuals))

            fp = len(neg.intersection(individuals))
            fn = len(pos.difference(individuals))
            try:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f_1 = 2 * ((precision * recall) / (precision + recall))
            except:
                f_1 = 0

            y.append(round(f_1, 5))
        return y

    def score_with_instances(self, *, pos, neg, instances):
        return self.quality_func.score(pos=pos, neg=neg, instances=instances)

    def score_with_labels(self, *, pos, neg, labels):
        assert isinstance(pos, list)
        assert isinstance(neg, list)

        pos = set(pos)
        neg = set(neg)

        y = []
        for j in labels:
            individuals = j.instances

            tp = len(pos.intersection(individuals))
            tn = len(neg.difference(individuals))

            fp = len(neg.intersection(individuals))
            fn = len(pos.difference(individuals))
            try:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f_1 = 2 * ((precision * recall) / (precision + recall))
            except:
                f_1 = 0

            y.append(round(f_1, 5))
        return y

    def train(self, num):
        print('Training starts')
        for i in self.labels:
            if len(i.instances) < num or (len(i.instances) >= len(self.kb.thing.instances) - num):
                continue

            all_positives = {self.individuals.index(_) for _ in i.instances}

            all_negatives = {self.individuals.index(_) for _ in self.kb.thing.instances - i.instances}

            yield all_positives, all_negatives, i

    @staticmethod
    def __generate_concepts(*, root_concept, rho, max_concept: int) -> List:
        """
        Generate a list of concepts that are randomly generated given root_concept
        Given a root concept, and refinement operator

        # ->If memory usage needs to be optimized,then one could leverage generators."""
        concepts_to_be_refined = set()
        refined_concepts = set()

        concepts_to_be_refined.add(root_concept)
        while len(refined_concepts) < max_concept:
            try:
                c = concepts_to_be_refined.pop()
            except KeyError:
                print('Break')
                break
            if c in refined_concepts:
                continue
            # print(len(refined_concepts), '.th Concept ', c.str, ' is refined.')
            for i in rho.refine(c):
                concepts_to_be_refined.add(i)
            refined_concepts.add(c)

        concepts = []
        concepts.extend(refined_concepts)
        concepts.extend(concepts_to_be_refined)
        return concepts

    def generate(self, **kwargs):

        x = self.__generate_concepts(root_concept=kwargs['root_concept'],
                                     rho=kwargs['refinement_operator'],
                                     max_concept=kwargs['num_of_concepts_refined'])

        # prune concepts that do not satisfy the provided constraint.
        x = [concept for concept in x if len(concept.instances) > kwargs['num_of_inputs_for_model']]

        if kwargs['learning_problem'] == 'concept_learning':
            self.target_idx = dict()
            y = []
            for i in x:
                self.target_idx.setdefault(i.str, len(self.target_idx))
                y.append(self.target_idx[i.str])

            y = np.array(y).reshape(len(y), 1)

        elif kwargs['learning_problem'] == 'length_prediction':
            y = [len(concept) for concept in x]
            # One might need to omit binary representations for outputs due to sparsity and memory usage.
            self.lb.fit(y)
            y = self.lb.transform(y)

        return x, y, {'num_instances': self.num_individuals,
                      'num_of_outputs': y.shape[1]}

    def generate_concepts(self, **kwargs):
        """
        Generate concepts according to the provided settings.
        Root concept:
        An upperbound on the number of concepts.
        """

        x = self.__generate_concepts(root_concept=kwargs['root_concept'],
                                     rho=kwargs['refinement_operator'],
                                     max_concept=kwargs['num_of_concepts_refined'])

        self.logger.info('Number of concepts generated:{0}'.format(len(x)))
        return x

    def convert_data(self, concepts, labels, params):

        X, y = [], []
        # Generate Training Data
        for _ in range(params['num_of_times_sample_per_concept']):
            for c in concepts:
                try:
                    x_pos, x_neg = self.pos_neg_sampling_from_concept(c, params['num_of_inputs_for_model'])
                except ValueError:
                    # self.logger.info('During training,{0} is ignored due due to number of instances {1}.'
                    #                 ''.format(c.str,len(c.instances)))
                    continue
                vec_of_f_scores = self.score_with_labels(pos=x_pos, neg=x_neg, labels=labels)
                y.append(vec_of_f_scores)
                X.append([self.indx[i] for i in x_pos + x_neg])

        return X, y

    def __get_index_from_iterable(self, x):

        res = []
        for i in x:
            res.append(self.individuals.index(i))

        if len(res) == 0:
            raise ValueError
        return res

    def get_mini_batch(self, X, y, jth, kwargs):

        sampled_instances = []
        concepts = []
        for x in X[jth:jth + kwargs['batch_size']]:
            sample = random.sample(x.instances, k=kwargs['num_of_inputs_for_model'])
            sampled_instances.append(self.__get_index_from_iterable(sample))

            concepts.append(x)

        return concepts, sampled_instances, y[jth:jth + kwargs['batch_size']]


class PriorityQueue:
    def __init__(self):
        from queue import PriorityQueue
        self.struct = PriorityQueue()
        self.items_in_queue = set()
        self.expressionTests = 0

    def __len__(self):
        return len(self.items_in_queue)

    def add_into_queue(self, t: Tuple):
        assert not isinstance(t[0], str)
        if not (t[1].ce.expression in self.items_in_queue):
            self.struct.put(t)
            self.items_in_queue.add(t[1].ce.expression)
            self.expressionTests += 1

    def get_from_queue(self):
        assert len(self.items_in_queue) > 0
        node_to_return = self.struct.get()[1]
        self.items_in_queue.remove(node_to_return.ce.expression)
        return node_to_return
