from typing import Dict, Set, Tuple, List, Union, Callable, Iterable
import numpy as np
import pandas as pd
from owlapy.class_expression import OWLObjectIntersectionOf, OWLClassExpression, OWLObjectUnionOf, OWLDataHasValue, \
    OWLDataSomeValuesFrom, OWLClass
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
from ..utils.static_funcs import plot_umap_reduced_embeddings, plot_decision_tree_of_expressions
import itertools
from owlapy.class_expression import OWLDataMinCardinality, OWLDataMaxCardinality, \
    OWLObjectOneOf
from owlapy.class_expression import OWLDataMinCardinality, OWLDataOneOf, OWLDataSomeValuesFrom
from owlapy.providers import owl_datatype_min_inclusive_restriction, owl_datatype_max_inclusive_restriction
from owlapy.providers import owl_datatype_min_exclusive_restriction, \
    owl_datatype_max_exclusive_restriction, owl_datatype_min_inclusive_restriction
import scipy
from owlapy import owl_expression_to_dl, owl_expression_to_sparql
from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectMinCardinality
from owlapy.providers import owl_datatype_min_max_exclusive_restriction


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


def make_iterable_verbose(iterable_object, verbose, desc="Default") -> Iterable:
    if verbose > 0:
        return tqdm(iterable_object, desc=desc)
    else:
        return iterable_object


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


def explain_inference(clf, X_test: pd.DataFrame):
    """
    Given a trained Decision Tree, extract the paths from root to leaf nodes for each entities
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#understanding-the-decision-tree-structure

    """
    reports = []

    # i-th feature_tree represent a feature used in the i-th node
    feature_tree = clf.tree_.feature

    # i-th item denotes the threshold in the i-th node.
    threshold_value_in_nodes = clf.tree_.threshold
    # Positives
    node_indicator: scipy.sparse._csr.csr_matrix
    node_indicator = clf.decision_path(X_test)
    #  the summary of the training samples that reached node i for class j and output k

    features: List[Tuple[OWLClassExpression, OWLDataProperty]]
    features = X_test.columns.to_list()
    # Leaf id for each example
    leaf_id: np.ndarray
    leaf_id = clf.apply(X_test)
    # node_indicator: tuple of integers denotes the index of example and the index of node.
    # the last integer denotes the class
    #   (0, 0)	1
    #   (0, 8)	1
    #   (0, 9)	1
    #   (0, 10)	1
    # i-th item in leaf_id denotes the leaf node of the i-th example [10, ...., 10]

    np_X_test = X_test.values

    for i, np_individual in enumerate(np_X_test):
        # (1) Extract nodes relating to the classification of the i-th example
        node_indices = node_indicator.indices[node_indicator.indptr[i]: node_indicator.indptr[i + 1]]

        decision_path = []
        for th_node, node_id in enumerate(node_indices):
            if leaf_id[i] == node_id:
                continue
            index_of_feature_owl_ce = feature_tree[node_id]

            decision_path.append({  # "decision_node": node_id,
                # OWLClassExpression or OWLDataProperty
                "feature": features[index_of_feature_owl_ce],
                # Feature value of an individual, e.g. 1.0 or 0.0 for booleans
                "feature_value_of_individual": np_individual[index_of_feature_owl_ce],
                #
                "threshold_value": threshold_value_in_nodes[node_id],
            })
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
                 plot_embeddings: bool = False,
                 verbose: int = 1):
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
        self.owl_class_expressions = set()
        self.cbd_mapping: Dict[str, Set[Tuple[str, str]]]
        self.types_of_individuals = dict()
        self.verbose = verbose
        self.data_property_cast = dict()

    def create_training_data(self, learning_problem: PosNegLPStandard) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a training data (X:pandas.DataFrame of (n,d) , y:pandas.Series of (n,1)) for binary class problem.
        n denotes the number of examples
        d denotes the number of features extracted from n examples.

        return X, y
        """
        # (1) Initialize features.
        features: List[OWLClassExpression]
        features = list()
        # (2) Initialize ordered examples.
        positive_examples: List[OWLNamedIndividual]
        negative_examples: List[OWLNamedIndividual]
        positive_examples = [i for i in learning_problem.pos]
        negative_examples = [i for i in learning_problem.neg]
        examples = positive_examples + negative_examples
        # TODO: Asyncio ?!
        for i in make_iterable_verbose(examples,
                                       verbose=self.verbose,
                                       desc="Extracting information about examples"):
            for expression in self.knowledge_base.abox(individual=i, mode="expression"):
                features.append(expression)
        assert len(
            features) > 0, f"First hop features cannot be extracted. Ensure that there are axioms about the examples."
        print("Total extracted features:", len(features))
        features = set(features)
        print("Unique features:", len(features))
        binary_features = []
        # IMPORTANT: our features either
        for i in features:
            if isinstance(i, OWLClass) or isinstance(i, OWLObjectSomeValuesFrom) or isinstance(i,
                                                                                               OWLObjectMinCardinality):
                # Person, \exist hasChild Female, < 2
                binary_features.append(i)
            elif isinstance(i, OWLDataSomeValuesFrom):
                # (Currently) \exist r. {True, False} =>
                owl_literals = [i for i in i.get_filler().operands()]
                if owl_literals[0].is_boolean():
                    binary_features.append(i)
                elif owl_literals[0].is_double():
                    binary_features.append(i)

                else:
                    raise RuntimeError(f"Unrecognized type:{i}")
            else:
                raise RuntimeError(f"Unrecognized type:{i}")

        features = binary_features
        # (4) Order features: create a mapping from tuple of predicate and objects to integers starting from 0.
        mapping_features = {predicate_object_pair: index_ for index_, predicate_object_pair in enumerate(features)}
        # (5) Creating a tabular data for the binary classification problem.
        X, y = [], []
        for ith_row, i in enumerate(make_iterable_verbose(examples,
                                                          verbose=self.verbose,
                                                          desc="Creating supervised binary classification data")):
            # IMPORTANT: None existence is described as 0.0 features.
            X_i = [0.0 for _ in range(len(mapping_features))]
            expression: [OWLClass, OWLObjectSomeValuesFrom, OWLObjectMinCardinality, OWLDataSomeValuesFrom]
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
                            f"Type of literal in OWLDataSomeValuesFrom is not understood:{datavalues_in_fillers}")
                elif isinstance(expression, OWLClass) or isinstance(expression, OWLObjectSomeValuesFrom):
                    assert expression in mapping_features, expression
                    X_i[mapping_features[expression]] = 1.0
                elif isinstance(expression, OWLObjectMinCardinality):
                    X_i[mapping_features[expression]] = expression.get_cardinality()
                else:
                    raise RuntimeError(f"Unrecognized type:{expression}-{type(expression)}")

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

    def construct_owl_expression_from_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLObjectIntersectionOf]:
        """ Construct an OWL class expression from a decision tree """
        positive_examples: List[OWLNamedIndividual]
        positive_examples = y[y.label == 1].index.tolist()

        prediction_per_example = []
        # () Iterate over reasoning steps of predicting a positive example
        pos: OWLNamedIndividual
        for sequence_of_reasoning_steps, pos in zip(
                explain_inference(self.clf,
                                  X_test=X.loc[positive_examples]), positive_examples):
            concepts_per_reasoning_step = []
            for i in sequence_of_reasoning_steps:
                # sanity checking about the decision.
                if isinstance(i["feature"], OWLDataProperty):
                    # Detect the type of literal
                    owl_literal = OWLLiteral(self.data_property_cast[i["feature"]](i["feature_value_of_individual"]))
                    if owl_literal.is_boolean():
                        # Feature: Dataproperty amesTestPositive
                        # Condition value: {False, True}
                        assert i["feature_value_of_individual"] in [0.0, 1.0]
                        assert i["threshold_value"] == 0.5
                        if i["feature_value_of_individual"] <= 0.5:
                            # Two options for conditions holding:
                            # (1) Either (pos amesTestPositive False) in KG.
                            # (2) Or (pos amesTestPositive, ?) not in KG
                            owl_class_expression = OWLDataHasValue(property=i["feature"], value=OWLLiteral(False))
                            # Checking whether (1) holds
                            if pos in {i in self.knowledge_base.individuals(owl_class_expression)}:
                                "p \in Retrieval(∃ amesTestPositive.{False})"
                            else:
                                "p \in Retrieval(\not(∃ amesTestPositive.{False}))"
                                owl_class_expression = owl_class_expression.get_object_complement_of()
                        else:
                            # Two options for conditions not holding:
                            # (1) (pos amesTestPositive True) in KG.
                            # (2) (pos amesTestPositive, ?) not in.
                            owl_class_expression = OWLDataHasValue(property=i["feature"], value=OWLLiteral(True))

                    else:
                        raise NotImplementedError
                        # DONE!

                elif type(i["feature"]) in [OWLClass, OWLObjectSomeValuesFrom, OWLObjectMinCardinality]:
                    ####################################################################################################
                    # DONE
                    # Feature: Female, ≥ 3 hasStructure.owl:NamedIndividual
                    # Condition Feature(individual) <= 0.5
                    # Explanation: Feature does not hold for the individual
                    if i["feature_value_of_individual"] <= i["threshold_value"]:
                        # Condition holds: Feature(individual)==0.0
                        # Therefore, neg Feature(individual)==1.0
                        owl_class_expression = i["feature"].get_object_complement_of()
                    else:
                        owl_class_expression = i["feature"]
                elif type(i["feature"]) == OWLDataSomeValuesFrom:
                    if i["feature_value_of_individual"] <= i["threshold_value"]:
                        owl_class_expression = i["feature"].get_object_complement_of()
                    else:
                        owl_class_expression = i["feature"]
                else:
                    raise RuntimeError(f"Unrecognized feature:{i['feature']}-{type(i['feature'])}")

                    ####################################################################################################
                    # Expensive Sanity Checking:
                    # The respective positive example should be one of the the retrieved individuals
                    ########################################################################################################
                    """
                    try:
                        indvs={_ for _ in self.knowledge_base.individuals(owl_class_expression)}
                        assert pos in {_ for _ in self.knowledge_base.individuals(owl_class_expression)}
                    except AssertionError:
                        print(i)
                        raise AssertionError(f"{pos} is not founded in the retrieval of {owl_expression_to_dl(owl_class_expression)}\n{owl_expression_to_sparql(expression=owl_class_expression)}\nSize:{len(indvs)}")

                    """

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
            plot_decision_tree_of_expressions(feature_names=[owl_expression_to_dl(f) for f in self.features],
                                              cart_tree=self.clf, topk=10)

        self.owl_class_expressions.clear()
        # Each item can be considered is a path of OWL Class Expressions
        # starting from the root node in the decision tree and
        # ending in a leaf node.
        self.conjunctive_concepts: List[OWLObjectIntersectionOf]
        self.conjunctive_concepts = self.construct_owl_expression_from_tree(X, y)
        for i in self.conjunctive_concepts:
            self.owl_class_expressions.add(i)

        self.disjunction_of_conjunctive_concepts = concepts_reducer(concepts=self.conjunctive_concepts,
                                                                    reduced_cls=OWLObjectUnionOf)

        return self

    def best_hypotheses(self, n=1) -> Tuple[OWLClassExpression, List[OWLClassExpression]]:
        """ Return the prediction"""
        if n == 1:
            return self.disjunction_of_conjunctive_concepts
        else:
            return [self.disjunction_of_conjunctive_concepts] + [i for i in
                                                                 itertools.islice(self.owl_class_expressions, n)]

    def predict(self, X: List[OWLNamedIndividual], proba=True) -> np.ndarray:
        """ Predict the likelihoods of individuals belonging to the classes"""
        raise NotImplementedError("Unavailable. Predict the likelihoods of individuals belonging to the classes")
        owl_individuals = [i.str for i in X]
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
