# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

from typing import Dict, Set, Tuple, List, Union, Callable
import numpy as np
import pandas as pd
from ontolearn.verbalizer import verbalize_learner_prediction
from owlapy.class_expression import (
    OWLObjectIntersectionOf,
    OWLClassExpression,
    OWLObjectUnionOf,
)
from owlapy.owl_individual import OWLNamedIndividual
import ontolearn.triple_store
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
import sklearn
from sklearn import tree

from ..utils.static_funcs import plot_umap_reduced_embeddings, plot_decision_tree_of_expressions, \
    plot_topk_feature_importance

import itertools
from owlapy import owl_expression_to_dl
from ..utils.static_funcs import make_iterable_verbose


def explain_inference(clf, X: pd.DataFrame):
    """
    Given a trained Decision Tree, extract the paths from root to leaf nodes for each entities
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#understanding-the-decision-tree-structure

    """
    np_X = X.values
    # () feature[i] denotes a feature id used for splitting node i.
    # feature represents the feature id OWLClassExpressions used for splitting nodes of decision tree.
    feature: np.ndarray
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    owl_class_expression_features: List[OWLClassExpression]
    owl_class_expression_features = X.columns.to_list()

    node_indicator = clf.decision_path(np_X)
    # node_indicator:
    # () Tuple of integers denotes the index of example and the index of node.
    # () The last integer denotes the class (1/0)
    #   (0, 0)	1
    #   (0, 8)	1
    #   (0, 9)	1
    #   (0, 10)	1

    # Explanation of selection over csr_matrix
    # The column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
    # For more :https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    reports = []
    leaf_id = clf.apply(np_X)
    for sample_id in range(len(np_X)):
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ]

        # print("Rules used to predict sample {id}:\n".format(id=sample_id))

        decision_path = []
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if np_X[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            """
            print(
                "decision node {node} : (X[{sample}, {feature}] = {value}) "
                "{inequality} {threshold})\t OWL:{expression}".format(
                    node=node_id,
                    sample=sample_id,
                    feature=feature[node_id],
                    value=np_X[sample_id, feature[node_id]],
                    inequality=threshold_sign,
                    threshold=threshold[node_id],
                    expression=owl_class_expression_features[feature[node_id]]
                )
            )
            """
            decision_path.append({"node_id": node_id,
                                  "feature_id": feature[node_id],
                                  "feature_value_of_individual": np_X[sample_id, feature[node_id]],
                                  "inequality": threshold_sign,
                                  "threshold_value": threshold[node_id],
                                  "owl_expression": owl_class_expression_features[feature[node_id]]})

        reports.append(decision_path)
    return reports


def concepts_reducer(
        concepts: List[OWLClassExpression], reduced_cls: Callable
) -> Union[OWLObjectUnionOf, OWLObjectIntersectionOf]:
    """Reduces a list of OWLClassExpression instances into a single instance of OWLObjectUnionOf or OWLObjectIntersectionOf"""
    dl_concept_path = None
    for c in concepts:
        assert isinstance(c, OWLClassExpression), f"c is not OWL: {type(c)}"
        if dl_concept_path is None:
            dl_concept_path = c
        else:
            dl_concept_path = reduced_cls((dl_concept_path, c))
    return dl_concept_path


class TDL:
    """Tree-based Description Logic Concept Learner"""

    def __init__(self, knowledge_base,
                 use_inverse: bool = False,
                 use_data_properties: bool = False,
                 use_nominals: bool = False,
                 use_card_restrictions: bool = False,
                 kwargs_classifier: dict = None,
                 max_runtime: int = 1,
                 grid_search_over: dict = None,
                 grid_search_apply: bool = False,
                 report_classification: bool = True,
                 plot_tree: bool = False,
                 plot_embeddings: bool = False,
                 plot_feature_importance: bool = False,
                 verbose: int = 10,
                 verbalize: bool = False):

        assert use_inverse is False, "use_inverse not implemented"
        assert use_data_properties is False, "use_data_properties not implemented"
        assert use_card_restrictions is False, "use_card_restrictions not implemented"
        self.use_nominals = use_nominals
        self.use_card_restrictions = use_card_restrictions

        if grid_search_over is None and grid_search_apply:
            grid_search_over = {
                "criterion": ["entropy", "gini", "log_loss"],
                "splitter": ["random", "best"],
                "max_features": [None, "sqrt", "log2"],
                "min_samples_leaf": [1, 2, 3, 4, 5, 10],
                "max_depth": [1, 2, 3, 4, 5, 10, None],
            }
        else:
            grid_search_over = dict()
        assert (
                isinstance(knowledge_base, KnowledgeBase)
                or isinstance(knowledge_base, ontolearn.triple_store.TripleStore)
                or isinstance(knowledge_base)
        ), "knowledge_base must be a KnowledgeBase instance"
        print(f"Knowledge Base: {knowledge_base}")
        self.grid_search_over = grid_search_over
        self.knowledge_base = knowledge_base
        self.report_classification = report_classification
        self.plot_tree = plot_tree
        self.plot_embeddings = plot_embeddings
        self.plot_feature_importance = plot_feature_importance
        # Keyword arguments for sklearn Decision tree.
        # Initialize classifier
        self.clf = None
        self.kwargs_classifier = kwargs_classifier if kwargs_classifier else dict()
        self.max_runtime = max_runtime
        self.features = None
        # best pred
        self.disjunction_of_conjunctive_concepts = None
        self.conjunctive_concepts = None
        self.owl_class_expressions = set()
        self.cbd_mapping: Dict[str, Set[Tuple[str, str]]]
        self.types_of_individuals = dict()
        self.verbose = verbose
        self.verbalize = verbalize
        self.data_property_cast = dict()
        self.__classification_report = None
        self.X = None
        self.y = None

    def extract_expressions_from_owl_individuals(self, individuals: List[OWLNamedIndividual]) -> (
            Tuple)[Dict[str, OWLClassExpression],Dict[str, str]]:
        # () Store mappings from str dl concept to owl class expression objects.
        features = dict()
        # () Grouped str dl concepts given str individuals.
        individuals_to_feature_mapping = dict()
        for owl_named_individual in make_iterable_verbose(individuals,
                                       verbose=self.verbose,
                                       desc="Extracting information about examples"):
            for owl_class_expression in self.knowledge_base.abox(individual=owl_named_individual, mode="expression"):
                str_dl_concept=owl_expression_to_dl(owl_class_expression)
                individuals_to_feature_mapping.setdefault(owl_named_individual.str,set()).add(str_dl_concept)
                if str_dl_concept not in features:
                    # A mapping from str dl representation to owl object.
                    features[str_dl_concept] = owl_class_expression

        assert len(features) > 0, "First hop features cannot be extracted. Ensure that there are axioms about the examples."
        if self.verbose > 0:
            print("Unique OWL Class Expressions as features :", len(features))
        # () Iterate over features/extracted owl expressions.
        # TODO:CD: We need to use parse tensor representation that we can use to train decision tree
        X = []
        features = [ v for k,v in features.items()]
        for owl_named_individual in make_iterable_verbose(individuals,
                                       verbose=self.verbose,
                                       desc="Constructing Training Data"):
            binary_sparse_representation = []

            features_of_owl_named_individual=individuals_to_feature_mapping[owl_named_individual.str]

            for owl_class_expression in features:
                if owl_expression_to_dl(owl_class_expression) in features_of_owl_named_individual:
                    binary_sparse_representation.append(1.0)
                else:
                    binary_sparse_representation.append(0.0)
            X.append(binary_sparse_representation)
        X = np.array(X)
        return X, features

    def construct_sparse_binary_representations(self,
                                                features: List[OWLClassExpression],
                                                examples: List[OWLNamedIndividual], examples_to_features) -> np.array:
        # () Constructing sparse binary vector representations for examples.
        # () Iterate over features/extracted owl expressions.
        X = []
        # ()
        str_owl_named_individual:str
        for str_owl_named_individual, list_of_owl_expressions in examples_to_features.items():
            for kk in list_of_owl_expressions:
                assert kk in features
        #  number of rows
        for i in examples:
            print(i.str)

        exit(1)

        assert len(X)==len(examples)
        for f in make_iterable_verbose(features,
                                       verbose=self.verbose,
                                       desc="Creating sparse binary representations for the training"):
            # () Retrieve instances belonging to a feature/owl class expression
            # TODO: Very inefficient.
            feature_retrieval = {_ for _ in self.knowledge_base.individuals(f)}
            # () Add 1.0 if positive example found otherwise 0.
            feature_value_per_example = []
            for e in examples:
                if e in feature_retrieval:
                    feature_value_per_example.append(1.0)

                else:
                    feature_value_per_example.append(0.0)
            X.append(feature_value_per_example)
        # Transpose to obtain the training data.
        X = np.array(X).T
        return X

    def create_training_data(self, learning_problem: PosNegLPStandard) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # (1) Initialize ordering over positive and negative examples.
        if self.verbose > 0:
            print("Creating a Training Dataset")
        positive_examples: List[OWLNamedIndividual]
        negative_examples: List[OWLNamedIndividual]
        positive_examples = [i for i in learning_problem.pos]
        negative_examples = [i for i in learning_problem.neg]
        # (2) Initialize labels for (1).
        y = [1.0 for _ in positive_examples] + [0.0 for _ in negative_examples]
        # (3) Iterate over examples to extract unique features.
        examples = positive_examples + negative_examples
        # For the sake of convenience. sort features in ascending order of string lengths of DL representations.
        X, features = self.extract_expressions_from_owl_individuals(examples)
        # (4) Creating a tabular data for the binary classification problem.
        # X = self.construct_sparse_binary_representations(features, examples, examples_to_features)
        self.features = features
        X = pd.DataFrame(data=X, index=examples, columns=self.features)
        y = pd.DataFrame(data=y, index=examples, columns=["label"])
        # Remove redundant columns
        same_value_columns = X.apply(lambda col: col.nunique() == 1)
        X = X.loc[:, ~same_value_columns]
        self.features = X.columns.values.tolist()
        return X, y

    def construct_owl_expression_from_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLObjectIntersectionOf]:
        """ Construct an OWL class expression from a decision tree"""

        positive_examples: List[OWLNamedIndividual]
        positive_examples = y[y.label == 1].index.tolist()
        vector_representation_of_positive_examples = X.loc[positive_examples]
        prediction_per_example = []
        # () Iterate over reasoning steps of predicting a positive example
        pos: OWLNamedIndividual

        for sequence_of_reasoning_steps, pos in zip(make_iterable_verbose(explain_inference(self.clf,
                                            X=vector_representation_of_positive_examples),
                                            verbose=self.verbose,
                                            desc="Constructing Description Logic Concepts"), positive_examples):
            concepts_per_reasoning_step = []
            for i in sequence_of_reasoning_steps:
                if i["inequality"] == ">":
                    owl_class_expression = i["owl_expression"]
                else:
                    owl_class_expression = i["owl_expression"].get_object_complement_of()

                concepts_per_reasoning_step.append(owl_class_expression)
                # TODO : CD: No need to perform retrieval.
                """
                print(i,owl_class_expression)
                retrival_result = pos in {_ for _ in self.knowledge_base.individuals(owl_class_expression)}

                if retrival_result:
                    concepts_per_reasoning_step.append(owl_class_expression)
                else:
                    raise RuntimeError("Incorrect retrival")
                """
            pred = concepts_reducer(concepts=concepts_per_reasoning_step, reduced_cls=OWLObjectIntersectionOf)
            prediction_per_example.append((pred, pos))

        # From list to set to remove identical paths from the root to leafs.
        prediction_per_example = {pred for pred, positive_example in prediction_per_example}
        return list(prediction_per_example)

    def fit(self, learning_problem: PosNegLPStandard = None, max_runtime: int = None):
        """Fit the learner to the given learning problem

        (1) Extract multi-hop information about E^+ and E^-.
        (2) Create OWL Class Expressions from (1)
        (3) Build a binary sparse training data X where
            first |E+| rows denote the binary representations of positives
            Remaining rows denote the binary representations of E⁻
        (4) Create binary labels.
        (4) Construct a set of DL concept for each e \in E^+
        (5) Union (4)

        :param learning_problem: The learning problem
        :param max_runtime:total runtime of the learning

        """
        assert learning_problem is not None, "Learning problem cannot be None."
        assert isinstance(
            learning_problem, PosNegLPStandard
        ), f"Learning problem must be PosNegLPStandard. Currently:{learning_problem}."

        if max_runtime is not None:
            self.max_runtime = max_runtime
        X: pd.DataFrame
        y: Union[pd.DataFrame, pd.Series]
        X, y = self.create_training_data(learning_problem=learning_problem)
        # CD: Remember so that if user wants to use them
        self.X, self.y = X, y
        if self.plot_embeddings:
            plot_umap_reduced_embeddings(X, y.label.to_list(), "umap_visualization.pdf")
        if self.grid_search_over:
            grid_search = sklearn.model_selection.GridSearchCV(
                tree.DecisionTreeClassifier(**self.kwargs_classifier),
                param_grid=self.grid_search_over, cv=10, ).fit(X.values, y.values)
            print(grid_search.best_params_)
            self.kwargs_classifier.update(grid_search.best_params_)
        # Training
        if self.verbose>0:
            print("Training starts!")
        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values)

        if self.report_classification:

            if self.verbose > 0:
                self.__classification_report = "Classification Report: Negatives: -1 and Positives 1 \n"
                self.__classification_report += sklearn.metrics.classification_report(y.values,
                                                                                      self.clf.predict(X.values),
                                                                                      target_names=["Negative",
                                                                                                    "Positive"])
                print(self.__classification_report)
        if self.plot_tree:
            plot_decision_tree_of_expressions(feature_names=[owl_expression_to_dl(f) for f in self.features],
                                              cart_tree=self.clf)
        if self.plot_feature_importance:
            plot_topk_feature_importance(feature_names=[owl_expression_to_dl(f) for f in self.features],
                                         cart_tree=self.clf)

        self.owl_class_expressions.clear()
        # Each item can be considered is a path of OWL Class Expressions
        # starting from the root node in the decision tree and
        # ending in a leaf node.
        self.conjunctive_concepts: List[OWLObjectIntersectionOf]
        if self.verbose >0:
            print("Computing conjunctive_concepts...")
        self.conjunctive_concepts = self.construct_owl_expression_from_tree(X, y)
        for i in self.conjunctive_concepts:
            self.owl_class_expressions.add(i)
        if self.verbose >0:
            print("Computing disjunction_of_conjunctive_concepts...")
        self.disjunction_of_conjunctive_concepts = concepts_reducer(concepts=self.conjunctive_concepts,  reduced_cls=OWLObjectUnionOf)

        if self.verbalize:
            verbalize_learner_prediction(self.disjunction_of_conjunctive_concepts)

        return self

    @property
    def classification_report(self) -> str:
        return self.__classification_report

    def best_hypotheses(
            self, n=1
    ) -> Tuple[OWLClassExpression, List[OWLClassExpression]]:
        """Return the prediction"""
        if n == 1:
            return self.disjunction_of_conjunctive_concepts
        else:
            return [self.disjunction_of_conjunctive_concepts] + [
                i for i in itertools.islice(self.owl_class_expressions, n)
            ]

    def predict(self, X: List[OWLNamedIndividual], proba=True) -> np.ndarray:

        """ Predict the likelihoods of individuals belonging to the classes"""
        raise NotImplementedError("Unavailable. Predict the likelihoods of individuals belonging to the classes")
