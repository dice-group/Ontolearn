import numpy as np
import owlapy.model
import pandas as pd
import requests
import json

import ontolearn.triple_store
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.base import OWLOntologyManager_Owlready2
from owlapy.model import OWLEquivalentClassesAxiom, OWLOntologyManager, OWLOntology, AddImport, OWLImportsDeclaration, \
    IRI, OWLDataOneOf, OWLObjectProperty, OWLObjectOneOf, OWLDataProperty

from typing import Dict, Set, Tuple, List, Union, TypeVar, Callable, Generator
from ontolearn.learning_problem import PosNegLPStandard
import collections
from tqdm import tqdm
import sklearn
from sklearn import tree

from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLDataHasValue, OWLObjectHasValue, OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer, ManchesterOWLSyntaxOWLObjectRenderer

import time
from ..utils.static_funcs import plot_umap_reduced_embeddings, plot_decision_tree_of_expressions


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def compute_quality(instances, pos, neg, conf_matrix=False, quality_func=None):
    assert isinstance(instances, set)
    tp = len(pos.intersection(instances))
    tn = len(neg.difference(instances))

    fp = len(neg.intersection(instances))
    fn = len(pos.difference(instances))

    _, f1_score = quality_func.score2(tp=tp, fn=fn, fp=fp, tn=tn)
    if conf_matrix:
        return f1_score, f"TP:{tp}\tFN:{fn}\tFP:{fp}\tTN:{tn}"
    return f1_score


def extract_cbd(dataframe) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract concise bounded description for each entity, where the entity is a subject entity.
    Create a mapping from a node to out-going edges and connected nodes
    :param dataframe:
    :return:
    """
    # Extract concise bounded description for each entity, where the entity is a subject entity.
    data = dict()
    for i in dataframe.values.tolist():
        subject_, predicate_, object_ = i
        data.setdefault(subject_, []).append((predicate_, object_))
    return data


def explain_inference(clf, X_test, features, only_shared):
    reports = []
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    # Positives
    node_indicator = clf.decision_path(X_test)
    leaf_id = clf.apply(X_test)

    if only_shared:
        sample_ids = range(len(X_test))
        # boolean array indicating the nodes both samples go through
        common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
        # obtain node ids using position in array
        common_node_id = np.arange(n_nodes)[common_nodes]

        print(
            "The following samples {samples} share the node(s) {nodes} in the tree.".format(
                samples=sample_ids, nodes=common_node_id
            )
        )
        print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
        return None

    for sample_id in range(len(X_test)):
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
            if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            # report = f"decision node {node_id} : ({features[feature[node_id]]} = {X_test[sample_id, feature[node_id]]}) {threshold_sign} {threshold[node_id]})"
            decision_path.append({"decision_node": node_id, "feature": features[feature[node_id]],
                                  "value": X_test[sample_id, feature[node_id]]})
        reports.append(decision_path)
    return reports


def concepts_reducer(concepts: List[OWLClassExpression], reduced_cls: Callable) -> Union[
    OWLObjectUnionOf, OWLObjectIntersectionOf]:
    """ Reduces a list of OWLClassExpression instances into a single instance of OWLObjectUnionOf or OWLObjectIntersectionOf """
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
                 quality_func: Callable = None,
                 kwargs_classifier: dict = None,
                 max_runtime: int = 1,
                 grid_search_over: dict = None,
                 grid_search_apply: bool = False,
                 report_classification: bool = False,
                 plot_tree: bool = False,
                 plot_embeddings: bool = False):
        assert use_inverse is False, "use_inverse not implemented"
        assert use_data_properties is False, "use_data_properties not implemented"
        assert use_card_restrictions is False, "use_card_restrictions not implemented"

        self.use_nominals = use_nominals
        self.use_card_restrictions = use_card_restrictions

        if grid_search_over is None and grid_search_apply:
            grid_search_over = {'criterion': ["entropy", "gini", "log_loss"],
                                "splitter": ["random", "best"],
                                "max_features": [None, "sqrt", "log2"],
                                "min_samples_leaf": [1, 2, 3, 4, 5, 10],
                                "max_depth": [1, 2, 3, 4, 5, 10, None]}
        else:
            grid_search_over = dict()
        assert isinstance(knowledge_base, KnowledgeBase) or isinstance(knowledge_base,
                                                                       ontolearn.triple_store.TripleStore), "knowledge_base must be a KnowledgeBase instance"
        print(f"Knowledge Base: {knowledge_base}")
        self.grid_search_over = grid_search_over
        self.knowledge_base = knowledge_base
        self.report_classification = report_classification
        self.plot_tree = plot_tree
        self.plot_embeddings = plot_embeddings
        self.dl_render = DLSyntaxObjectRenderer()
        self.manchester_render = ManchesterOWLSyntaxOWLObjectRenderer()
        # Keyword arguments for sklearn Decision tree.
        # Initialize classifier
        self.clf = None
        self.kwargs_classifier = kwargs_classifier if kwargs_classifier else dict()
        self.max_runtime = max_runtime
        self.features = None
        # best pred
        self.disjunction_of_conjunctive_concepts = None
        self.conjunctive_concepts = None
        self.cbd_mapping: Dict[str, Set[Tuple[str, str]]]
        self.types_of_individuals = dict()

    def create_training_data(self, learning_problem: PosNegLPStandard) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a training data (X,y) for binary classification problem, where
        X is a sparse binary matrix and y is a binary vector.

        X: shape (n,d)
        y: shape (n,1).

        n denotes the number of examples
        d denotes the number of features extracted from n examples.
        """
        # (1) Initialize features.
        features = set()
        # (2) Initialize ordered examples.
        positive_examples = [i for i in learning_problem.pos]
        negative_examples = [i for i in learning_problem.neg]
        examples = positive_examples + negative_examples

        # (3) Extract all features from (2).
        for i in tqdm(examples, desc="Extracting information about examples"):
            expression: [OWLClassExpression, Tuple[OWLDataProperty, OWLLiteral]]
            sub_features = set()

            for expression in self.knowledge_base.abox(individual=i, mode="expression"):
                # @TODO: expression should not be
                if isinstance(expression, tuple):
                    p, _ = expression
                    sub_features.add(p)
                else:
                    sub_features.add(expression)
            features = features | sub_features
        assert len(
            features) > 0, f"First hop features cannot be extracted. Ensure that there are axioms about the examples."
        features = list(features)
        # (4) Order features: create a mapping from tuple of predicate and objects to integers starting from 0.
        mapping_features = {predicate_object_pair: index_ for index_, predicate_object_pair in enumerate(features)}
        print(f"\n{len(features)} features are extracted")
        # (5) Creating a tabular data for the binary classification problem.
        X = np.zeros(shape=(len(examples), len(features)), dtype=float)
        y = []
        for ith_row, i in enumerate(tqdm(examples, desc="Creating supervised binary classification data")):
            expression: [OWLClassExpression, Tuple[OWLDataProperty, OWLLiteral]]
            # Filling the features
            for expression in self.knowledge_base.abox(individual=i, mode="expression"):
                if isinstance(expression, tuple):
                    o: OWLLiteral
                    p, o = expression
                    assert p in mapping_features
                    if o.is_double():
                        value: float
                        value = o.parse_double()
                        assert isinstance(value, float)
                        X[ith_row, mapping_features[p]] = value
                    elif o.is_boolean():
                        value: bool
                        value = o.parse_boolean()
                        X[ith_row, mapping_features[p]] = float(value)
                    else:
                        raise RuntimeError(f"{o} type not requi ")

                else:
                    assert expression in mapping_features
                    assert isinstance(expression, OWLClassExpression)
                    X[ith_row, mapping_features[expression]] = 1.0

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

        X = pd.DataFrame(data=X, index=examples, columns=features)
        y = pd.DataFrame(data=y, index=examples, columns=["label"])
        return X, y

    def construct_owl_expression_from_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLObjectIntersectionOf]:
        """ Construct an OWL class expression from a decision tree """
        positive_examples: List[OWLNamedIndividual]
        positive_examples = y[y.label == 1].index.tolist()

        prediction_per_example = []
        # () Iterate over E^+
        for sequence_of_reasoning_steps, pos in zip(
                explain_inference(self.clf,
                                  X_test=X.loc[positive_examples].values,
                                  features=X.columns.to_list(),
                                  only_shared=False), positive_examples):
            concepts_per_reasoning_step = []
            for i in sequence_of_reasoning_steps:
                feature: Union[OWLClassExpression, OWLDataProperty]
                feature = i["feature"]
                # sanity checking about the decision.
                if isinstance(feature, OWLClassExpression):
                    assert 1.0 >= i["value"] >= 0.0
                    value = bool(i["value"])
                    if value is False:
                        owl_class_expression = feature.get_object_complement_of()
                    else:
                        owl_class_expression = feature
                else:
                    from owlapy.model import OWLDataRange
                    assert isinstance(feature, OWLDataProperty)
                    # {'decision_node': 0, 'feature': OWLDataProperty(IRI('http://dl-learner.org/mutagenesis#','act')), 'value': 4.99}
                    # We need https://www.w3.org/TR/2004/REC-owl-semantics-20040210/#owl_minCardinality
                    # https://www.w3.org/TR/owl-ref/#ValueRestriction
                    # @TODO:CD: Is this really correct ?!
                    owl_class_expression = OWLDataHasValue(property=feature, value=OWLLiteral(i["value"]))

                concepts_per_reasoning_step.append(owl_class_expression)

            pred = concepts_reducer(concepts=concepts_per_reasoning_step, reduced_cls=OWLObjectIntersectionOf)
            prediction_per_example.append((pred, pos))

        # From list to set to remove identical paths from the root to leafs.
        prediction_per_example = {pred for pred, positive_example in prediction_per_example}
        return list(prediction_per_example)

    def fit(self, learning_problem: PosNegLPStandard = None, max_runtime: int = None):
        """ Fit the learner to the given learning problem

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
        assert isinstance(learning_problem,
                          PosNegLPStandard), f"Learning problem must be PosNegLPStandard. Currently:{learning_problem}."

        if max_runtime is not None:
            self.max_runtime = max_runtime
        X: pd.DataFrame
        y: Union[pd.DataFrame, pd.Series]
        X, y = self.create_training_data(learning_problem=learning_problem)

        if self.plot_embeddings:
            plot_umap_reduced_embeddings(X, y.label.to_list(), "umap_visualization.pdf")

        if self.grid_search_over:
            grid_search = sklearn.model_selection.GridSearchCV(tree.DecisionTreeClassifier(**self.kwargs_classifier),
                                                               param_grid=self.grid_search_over, cv=10).fit(X.values,
                                                                                                            y.values)
            print(grid_search.best_params_)
            self.kwargs_classifier.update(grid_search.best_params_)

        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values)

        if self.report_classification:
            print("Classification Report: Negatives: -1 and Positives 1 ")
            print(sklearn.metrics.classification_report(y.values, self.clf.predict(X.values),
                                                        target_names=["Negative", "Positive"]))
        if self.plot_tree:
            plot_decision_tree_of_expressions(feature_names=[self.dl_render.render(f) for f in self.features],
                                              cart_tree=self.clf, topk=10)

        # Each item can be considered is a path of OWL Class Expressions
        # starting from the root node in the decision tree and
        # ending in a leaf node.
        self.conjunctive_concepts: List[OWLObjectIntersectionOf]
        self.conjunctive_concepts = self.construct_owl_expression_from_tree(X, y)
        self.disjunction_of_conjunctive_concepts = concepts_reducer(concepts=self.conjunctive_concepts,
                                                                    reduced_cls=OWLObjectUnionOf)

        return self

    def best_hypotheses(self, n=1):
        """ Return the prediction"""
        assert n == 1, "Only one hypothesis is supported"
        return self.disjunction_of_conjunctive_concepts

    def predict(self, X: List[OWLNamedIndividual], proba=True) -> np.ndarray:
        """ Predict the likelihoods of individuals belonging to the classes"""
        raise NotImplementedError("Unavailable. Predict the likelihoods of individuals belonging to the classes")
        owl_individuals = [i.get_iri().as_str() for i in X]
        hop_info, _ = self.construct_hop(owl_individuals)
        Xraw = self.built_sparse_training_data(entity_infos=hop_info,
                                               individuals=owl_individuals,
                                               feature_names=self.feature_names)
        # corrupt some infos
        Xraw_numpy = Xraw.values

        if proba:
            return self.clf.predict_proba(Xraw_numpy)
        else:
            return self.clf.predict(Xraw_numpy)
