from typing import Dict, Set, Tuple, List, Union, Callable, Iterable
import numpy as np
import pandas as pd
from owlapy.class_expression import (
    OWLObjectIntersectionOf,
    OWLClassExpression,
    OWLObjectUnionOf,
    OWLDataHasValue,
    OWLDataSomeValuesFrom,
    OWLClass,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLDataProperty
import ontolearn.triple_store
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.class_expression import OWLDataOneOf
from ontolearn.learning_problem import PosNegLPStandard
from tqdm import tqdm
import sklearn
from sklearn import tree
from owlapy.render import DLSyntaxObjectRenderer, ManchesterOWLSyntaxOWLObjectRenderer

from ..utils.static_funcs import plot_umap_reduced_embeddings, plot_decision_tree_of_expressions, \
    plot_topk_feature_importance

import itertools
from owlapy.class_expression import (
    OWLDataMinCardinality,
    OWLDataMaxCardinality,
    OWLObjectOneOf,
)
from owlapy.class_expression import (
    OWLDataMinCardinality,
    OWLDataOneOf,
    OWLDataSomeValuesFrom,
)
from owlapy.providers import (
    owl_datatype_min_inclusive_restriction,
    owl_datatype_max_inclusive_restriction,
)
from owlapy.providers import (
    owl_datatype_min_exclusive_restriction,
    owl_datatype_max_exclusive_restriction,
    owl_datatype_min_inclusive_restriction,
)
import scipy
from owlapy import owl_expression_to_dl, owl_expression_to_sparql
from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectMinCardinality
from owlapy.providers import owl_datatype_min_max_exclusive_restriction
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
                 report_classification: bool = False,
                 plot_tree: bool = False,
                 plot_embeddings: bool = False,
                 plot_feature_importance: bool = False,
                 verbose: int = 1):

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
        self.data_property_cast = dict()


    def extract_expressions_from_owl_individuals(self, individuals: List[OWLNamedIndividual]) -> List[
        OWLClassExpression]:
        features = []
        for i in make_iterable_verbose(individuals,
                                       verbose=self.verbose,
                                       desc="Extracting information about examples"):
            for expression in self.knowledge_base.abox(individual=i, mode="expression"):
                features.append(expression)
        assert len(
            features) > 0, f"First hop features cannot be extracted. Ensure that there are axioms about the examples."
        # (5) Obtain unique features from (4).
        if self.verbose > 0:
            print("Total extracted features:", len(features))
        features = set(features)
        if self.verbose > 0:
            print("Unique features:", len(features))
        return list(features)

    def construct_sparse_binary_representations(self, features: List[OWLClassExpression],
                                                examples: List[OWLNamedIndividual]) -> np.array:
        # () Constructing sparse binary vector representations for examples.
        # () Iterate over features/extracted owl expressions.
        X = []
        for f in features:
            # () Retrieve instances belonging to a feature/owl class expression
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
        positive_examples: List[OWLNamedIndividual]
        negative_examples: List[OWLNamedIndividual]
        positive_examples = [i for i in learning_problem.pos]
        negative_examples = [i for i in learning_problem.neg]
        # (2) Initialize labels for (1).
        y = [1.0 for _ in positive_examples] + [0.0 for _ in negative_examples]
        # (3) Iterate over examples to extract unique features.
        examples = positive_examples + negative_examples
        features = self.extract_expressions_from_owl_individuals(examples)
        # (4) Creating a tabular data for the binary classification problem.
        X = self.construct_sparse_binary_representations(features, examples)


        self.features = features
        X = pd.DataFrame(data=X, index=examples, columns=self.features)
        y = pd.DataFrame(data=y, index=examples, columns=["label"])

        same_value_columns = X.apply(lambda col: col.nunique() == 1)
        X = X.loc[:, ~same_value_columns]
        self.features=X.columns.values.tolist()
        return X, y

        """
        for ith_row, i in enumerate(make_iterable_verbose(examples,
                                                          verbose=self.verbose,
                                                          desc="Creating supervised binary classification data")):

            # IMPORTANT: None existence is described as 0.0 features.
            X_i = [0.0 for _ in range(len(mapping_features))]
            expression: [
                OWLClass,
                OWLObjectSomeValuesFrom,
                OWLObjectMinCardinality,
                OWLDataSomeValuesFrom,
            ]
            # Filling the features
            for expression in self.knowledge_base.abox(individual=i, mode="expression"):
                if isinstance(expression, OWLDataSomeValuesFrom):
                    fillers: OWLDataOneOf[OWLLiteral]
                    fillers = expression.get_filler()
                    datavalues_in_fillers = list(fillers.values())
                    if datavalues_in_fillers[0].is_boolean():
                        X_i[mapping_features[expression]] = 1
                    elif datavalues_in_fillers[0].is_double():
                        X_i[mapping_features[expression]] = 1.0
                    else:
                        raise RuntimeError(
                            f"Type of literal in OWLDataSomeValuesFrom is not understood:{datavalues_in_fillers}"
                        )
                elif isinstance(expression, OWLClass) or isinstance(
                    expression, OWLObjectSomeValuesFrom
                ):
                    assert expression in mapping_features, expression
                    X_i[mapping_features[expression]] = 1.0
                elif isinstance(expression, OWLObjectMinCardinality):
                    X_i[mapping_features[expression]] = expression.get_cardinality()
                else:
                    raise RuntimeError(
                        f"Unrecognized type:{expression}-{type(expression)}"
                    )

            X.append(X_i)
            # Filling the label
            if ith_row < len(positive_examples):
                # Sanity checking for positive examples.
                assert i in positive_examples and i not in negative_examples
                label = 1.0
            else:
                # Sanity checking for negative examples.
                assert i in negative_examples and i not in positive_examples
                label = 0.0
            y.append(label)

        self.features = features
        X = pd.DataFrame(data=X, index=examples, columns=self.features)
        y = pd.DataFrame(data=y, index=examples, columns=["label"])
        return X, y
        """

    def construct_owl_expression_from_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLObjectIntersectionOf]:
        """ Construct an OWL class expression from a decision tree"""

        positive_examples: List[OWLNamedIndividual]
        positive_examples = y[y.label == 1].index.tolist()
        vector_representation_of_positive_examples = X.loc[positive_examples]
        prediction_per_example = []
        # () Iterate over reasoning steps of predicting a positive example
        pos: OWLNamedIndividual
        for sequence_of_reasoning_steps, pos in zip(

                explain_inference(self.clf,
                                  X=vector_representation_of_positive_examples), positive_examples):
            concepts_per_reasoning_step = []
            for i in sequence_of_reasoning_steps:

                if i["inequality"] == ">":
                    owl_class_expression = i["owl_expression"]
                else:
                    owl_class_expression = i["owl_expression"].get_object_complement_of()


                retrival_result = pos in {_ for _ in self.knowledge_base.individuals(owl_class_expression)}

                if retrival_result:
                    concepts_per_reasoning_step.append(owl_class_expression)
                else:
                    raise RuntimeError("Incorrect retrival")

            pred = concepts_reducer(
                concepts=concepts_per_reasoning_step,
                reduced_cls=OWLObjectIntersectionOf,
            )
            prediction_per_example.append((pred, pos))

        # From list to set to remove identical paths from the root to leafs.
        prediction_per_example = {
            pred for pred, positive_example in prediction_per_example
        }
        return list(prediction_per_example)

    def fit(self, learning_problem: PosNegLPStandard = None, max_runtime: int = None):
        """Fit the learner to the given learning problem

        (1) Extract multi-hop information about E^+ and E^- denoted by \mathcal{F}.
        (1.1) E = list of (E^+ \sqcup E^-).
        (2) Build a training data \mathbf{X} \in  \mathbb{R}^{ |E| \times |\mathcal{F}| } .
        (3) Create binary labels \mathbf{X}.

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

        if self.plot_embeddings:
            plot_umap_reduced_embeddings(X, y.label.to_list(), "umap_visualization.pdf")

        if self.grid_search_over:
            grid_search = sklearn.model_selection.GridSearchCV(
                tree.DecisionTreeClassifier(**self.kwargs_classifier),
                param_grid=self.grid_search_over,
                cv=10,
            ).fit(X.values, y.values)
            print(grid_search.best_params_)
            self.kwargs_classifier.update(grid_search.best_params_)

        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(
            X=X.values, y=y.values
        )

        if self.report_classification:

            if self.verbose > 0:
                print("Classification Report: Negatives: -1 and Positives 1 ")
            print(sklearn.metrics.classification_report(y.values, self.clf.predict(X.values),
                                                        target_names=["Negative", "Positive"]))
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
        self.conjunctive_concepts = self.construct_owl_expression_from_tree(X, y)
        for i in self.conjunctive_concepts:
            self.owl_class_expressions.add(i)

        self.disjunction_of_conjunctive_concepts = concepts_reducer(
            concepts=self.conjunctive_concepts, reduced_cls=OWLObjectUnionOf
        )

        return self

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

