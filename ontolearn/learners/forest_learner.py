from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
import ontolearn.triple_store
from ontolearn.verbalizer import verbalize_learner_prediction
import numpy as np
import pandas as pd

from typing import Dict, Set, Tuple, List, Union, Callable

from ontolearn.knowledge_base import KnowledgeBase
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from ..utils.static_funcs import plot_umap_reduced_embeddings, plot_decision_tree_of_expressions, \
    plot_topk_feature_importance

import itertools
from owlapy.utils import HasFiller, HasOperands
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_dl
from ..utils.static_funcs import make_iterable_verbose

from owlapy.class_expression import (
    OWLObjectIntersectionOf,
    OWLClassExpression,
    OWLObjectUnionOf,
    OWLObjectOneOf,
    OWLObjectHasValue
)



class FTDL(TDL):

    def __init__(self, knowledge_base,
                 n_estimators: int = 10,
                 use_inverse: bool = False,
                 use_data_properties: bool = False,
                 use_nominals: bool = True,
                 use_card_restrictions: bool = False,
                 kwargs_classifier: dict = None,
                 max_runtime: int = 1,
                 grid_search_over: dict = None,
                 grid_search_apply: bool = False,
                 kwargs_grid_search: dict = {},
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
        elif grid_search_apply and grid_search_over is not None:
            pass
        else:
            grid_search_over = dict()

        kwargs_grid_search.setdefault("cv", 10)
        
        assert (
                isinstance(knowledge_base, KnowledgeBase)
                or isinstance(knowledge_base, ontolearn.triple_store.TripleStore)
                or isinstance(knowledge_base)
        ), "knowledge_base must be a KnowledgeBase instance"
        print(f"Knowledge Base: {knowledge_base}")
        self.grid_search_over = grid_search_over
        self.kwargs_grid_search = kwargs_grid_search
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


    def construct_owl_expression_from_tree(self,c: any, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLObjectIntersectionOf]:
        """ Construct an OWL class expression from a decision tree"""

        positive_examples: List[OWLNamedIndividual]
        positive_examples = y[y.label == 1].index.tolist()
        vector_representation_of_positive_examples = X.loc[positive_examples]
        prediction_per_example = []
        # () Iterate over reasoning steps of predicting a positive example
        pos: OWLNamedIndividual

        for sequence_of_reasoning_steps, pos in zip(make_iterable_verbose(super().explain_inference(c.clf,
                                            X=vector_representation_of_positive_examples),
                                            verbose=c.clf.verbose,
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
            pred = super().concepts_reducer(concepts=concepts_per_reasoning_step, reduced_cls=OWLObjectIntersectionOf)
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
                    param_grid=self.grid_search_over, **self.kwargs_grid_search).fit(X.values, y.values)
                print(grid_search.best_params_)
                self.kwargs_classifier.update(grid_search.best_params_)
            # Training
            if self.verbose>0:
                print("Training starts!")
            self.clf = RandomForestClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values).estimators_

            print("self clf" + str(type(self.clf[0])))

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
            self.disjunction_of_conjunctive_concepts = super().concepts_reducer(concepts=self.conjunctive_concepts,  reduced_cls=OWLObjectUnionOf)

            if self.verbalize:
                verbalize_learner_prediction(self.disjunction_of_conjunctive_concepts)

            return self
    def best_hypotheses(self, n=1):
        return super().best_hypotheses(n)