from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.verbalizer import verbalize_learner_prediction
import pandas as pd
from .tree_learner import explain_inference, concepts_reducer
from ontolearn.metrics import F1


from typing import Tuple, List, Union, Callable

import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from ..quality_funcs import evaluate_concept
from ..utils.static_funcs import plot_umap_reduced_embeddings

from owlapy.owl_individual import OWLNamedIndividual
from ..utils.static_funcs import make_iterable_verbose

from owlapy.class_expression import (
    OWLObjectIntersectionOf,
    OWLClassExpression,
    OWLObjectUnionOf
)



class FTDL(TDL):

    def __init__(self, knowledge_base,
                 n_estimators: int = 10,
                 quality_func: Callable = F1(),
                 reduce:bool  = False,
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
                 verbalize: bool = False, *args, **kwargs):
        super().__init__(knowledge_base, use_inverse, use_data_properties,use_nominals, use_card_restrictions, kwargs_classifier, max_runtime, grid_search_over, grid_search_apply, kwargs_grid_search, report_classification,plot_tree, plot_embeddings, plot_feature_importance, verbose, verbalize)
        self.n_estimators = n_estimators
        self.quality_func = quality_func
        self.reduce = reduce


    def construct_owl_expression_from_tree(self,c: any, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLObjectIntersectionOf]:
        """ Construct an OWL class expression from a decision tree"""

        positive_examples: List[OWLNamedIndividual]
        positive_examples = y[y.label == 1].index.tolist()
        vector_representation_of_positive_examples = X.loc[positive_examples]
        prediction_per_example = []
        # () Iterate over reasoning steps of predicting a positive example
        pos: OWLNamedIndividual

        for sequence_of_reasoning_steps, pos in zip(make_iterable_verbose(explain_inference(c,
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
            self._learning_problem = learning_problem.encode_kb(self.knowledge_base)
            # CD: Remember so that if user wants to use them
            self.X, self.y = X, y
            if self.plot_embeddings:
                plot_umap_reduced_embeddings(X, y.label.to_list(), "umap_visualization.pdf")
            if self.grid_search_over:
                grid_search = sklearn.model_selection.GridSearchCV(
                    RandomForestClassifier(**self.kwargs_classifier),
                    param_grid=self.grid_search_over, **self.kwargs_grid_search).fit(X.values, y.values)
                print(grid_search.best_params_)
                self.kwargs_classifier.update(grid_search.best_params_)
            # Training
            if self.verbose>0:
                print("Training starts!")
            self.clf = ExtraTreesClassifier(self.n_estimators, **self.kwargs_classifier, ).fit(X=X.values, y=y.values)

            if self.report_classification:

                if self.verbose > 0:
                    self.__classification_report = "Classification Report: Negatives: -1 and Positives 1 \n"
                    self.__classification_report += sklearn.metrics.classification_report(y.values,
                                                                                        self.clf.predict(X.values),
                                                                                        target_names=["Negative",
                                                                                                        "Positive"])
                    print(self.__classification_report)
            #if self.plot_tree:
              #  plot_decision_tree_of_expressions(feature_names=[owl_expression_to_dl(f) for f in self.features],
               #                                 cart_tree=self.clf)
            #if self.plot_feature_importance:
             #   plot_topk_feature_importance(feature_names=[owl_expression_to_dl(f) for f in self.features],
                #                            cart_tree=self.clf)

            self.owl_class_expressions.clear()
            # Each item can be considered is a path of OWL Class Expressions
            # starting from the root node in the decision tree and
            # ending in a leaf node.
            self.conjunctive_concepts: List[List[OWLObjectIntersectionOf]]
            self.conjunctive_concepts = []

            if self.verbose >0:
                print("Computing conjunctive_concepts...")
              
                
            
            for c in self.clf.estimators_:
                self.conjunctive_concepts.append(self.construct_owl_expression_from_tree(c, X, y))
            #print(self.conjunctive_concepts)
            
            self.tree_disjunctive_concepts = []
            for tree_conjunctive_concepts in self.conjunctive_concepts:
                for i in tree_conjunctive_concepts:
                    self.owl_class_expressions.add(i)
                if self.verbose >0:
                    print("Computing disjunction_of_conjunctive_concepts...")
                self.disjunction_of_conjunctive_concepts = concepts_reducer(concepts=tree_conjunctive_concepts,  reduced_cls=OWLObjectUnionOf)
                self.tree_disjunctive_concepts.append(self.disjunction_of_conjunctive_concepts)
            
            #print(len(self.tree_disjunctive_concepts))
            for tdc in self.tree_disjunctive_concepts:
                if self.verbalize:
                    verbalize_learner_prediction(tdc)
                   
            return self
    
 
    def best_hypotheses(
            self,n,reduce:bool = False 
    ) -> Tuple[OWLClassExpression, List[OWLClassExpression]]:
        """Return the prediction"""

        if(n == 1):
            # self.tree_disjunctive_concepts
            scores = []
            if(reduce):
                return concepts_reducer(self.tree_disjunctive_concepts, reduced_cls=OWLObjectUnionOf)
            for tdc in self.tree_disjunctive_concepts:
                print("Computing score")
                #scores.append(evaluate_concept(self.knowledge_base, tdc, self.quality_func, self.encoded_learning_problem() ))
                #scores.append((compute_f1_score(individuals=frozenset({i for i in self.knowledge_base.individuals(tdc)}), pos=self._learning_problem.pos, neg=self._learning_problem.neg), tdc))
                #scores.append((compute_f1_score(individuals=frozenset({i for i in self.knowledge_base.individuals(tdc)}), pos=self._learning_problem.pos, neg=self._learning_problem.neg), tdc))
                scores.append((evaluate_concept(self.knowledge_base,tdc,self.quality_func, self._learning_problem).q, tdc))
                
            for i in scores:
                print("score:")
                print(i[0])
            max_tuple = max(scores, key=lambda x: x[0])
            
            print("best score" + str(max_tuple[0]))
            return max_tuple[1]

            
            
        else:          
            return self.tree_disjunctive_concepts
        