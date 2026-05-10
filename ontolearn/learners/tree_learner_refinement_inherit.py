from shap import TreeExplainer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from typing import Dict, Set, Tuple, List, Union, Callable
from fitter import Fitter, get_common_distributions
import logging
import scipy as sc
from scipy.stats._distn_infrastructure import rv_frozen
from .tree_learner import concepts_reducer
from ontolearn.learners import TDL
logging.getLogger("fitter").setLevel(logging.CRITICAL)
import numpy as np
import pandas as pd
from ontolearn.verbalizer import verbalize_learner_prediction
from owlapy.class_expression import (
    OWLObjectIntersectionOf,
    OWLClassExpression,
    OWLObjectUnionOf,
    OWLObjectComplementOf,
    OWLObjectOneOf,
    OWLObjectHasValue,
    OWLDataOneOf,
    OWLDataHasValue,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectExactCardinality,
    OWLDataSomeValuesFrom,
    OWLDataAllValuesFrom,
    OWLClass,
)
from owlapy.utils import HasFiller
from owlapy.owl_individual import OWLNamedIndividual
import ontolearn.triple_store
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
import sklearn
from sklearn import tree
from collections import Counter

from ..utils.static_funcs import (
    plot_umap_reduced_embeddings,
    plot_decision_tree_of_expressions,
    plot_topk_feature_importance,
)

import itertools
from owlapy import owl_expression_to_dl
from ..utils.static_funcs import make_iterable_verbose
from owlapy.class_expression.restriction import (
    OWLDataSomeValuesFrom,
    OWLFacetRestriction,
    OWLDatatypeRestriction,
)
from owlapy.vocab import OWLFacet, XSDVocabulary
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_datatype import OWLDatatype
from collections.abc  import Sequence


class TDL_refinement(TDL):
    def __init__(self, knowledge_base,
                 use_inverse: bool = False,
                 use_data_properties: bool = True,
                 use_nominals: bool = True,
                 use_card_restrictions: bool = True,
                 kwargs_classifier: dict = None,
                 max_runtime: int = 1,
                 grid_search_over: dict = None,
                 grid_search_apply: bool = False,
                 kwargs_grid_search: dict = {},
                 report_classification: bool = False,
                 plot_tree: bool = False,
                 plot_embeddings: bool = False,
                 plot_feature_importance: bool = False,
                 verbose: int = 10,
                 verbalize: bool = False,
                 feature_refinement:bool = True,
                 plot_importance_evo:bool = False
                 ):
        super().__init__(knowledge_base,use_inverse,use_data_properties,use_nominals,use_card_restrictions,kwargs_classifier,max_runtime,grid_search_over,grid_search_apply,kwargs_grid_search,report_classification,
                       plot_tree, plot_embeddings,plot_feature_importance,verbose, verbalize)
        self.feature_refinement = feature_refinement
        self.plot_importance_evo = plot_importance_evo
        self.concept_per_iteration:List = list()
        self.top_feature_dicts:List = list()
        self.initial_importance_dict:Dict = dict()
 
    def _pack_data_property_with_range_to_dl_concept(self, property: OWLDataProperty, literal_range: tuple) -> str:
        """Repack the ranged data property into an DL Concept String"""
        min_restr = OWLFacetRestriction(
            OWLFacet.MIN_INCLUSIVE,
            OWLLiteral(literal_range[0], OWLDatatype(XSDVocabulary.DOUBLE)),
            #OWLLiteral(literal_range[0], property),
        )
        if literal_range is None:
            max_restr = OWLFacetRestriction(
                OWLFacet.MAX_INCLUSIVE,
                OWLLiteral(literal_range[0], OWLDatatype(XSDVocabulary.DOUBLE)),
                #OWLLiteral(literal_range[0], property),
            )
        else:
            max_restr = OWLFacetRestriction(
                OWLFacet.MAX_INCLUSIVE,
                OWLLiteral(literal_range[1], OWLDatatype(XSDVocabulary.DOUBLE)),
            )

        dt_range = OWLDatatypeRestriction(
            # here some way to auto detect datatype is needed
            type_=OWLDatatype(XSDVocabulary.DOUBLE),
            facet_restrictions=[min_restr, max_restr],
        )

        packed_owl_class_expression = OWLDataSomeValuesFrom(property, dt_range)
        return packed_owl_class_expression
    
    def _compute_dt_pdf_ranges(self, best_dist: dict) -> List[tuple]:
        """Compute ranges based on the fitted distribution. """
        frozen_dist: rv_frozen
        dist_name: list
        dist_name = list(best_dist.keys())[0]
        distribution = getattr(sc.stats, dist_name)
        frozen_dist = distribution(**best_dist[dist_name])
        mean: np.float64
        std: np.float64
        mean = frozen_dist.stats(moments="m")
        std = frozen_dist.std()
        ranges = [
            (mean - std, mean),
            (mean, mean + std),
            (mean - 2 * std, mean - 1 * std),
            (mean + std, mean + 2 * std),
        ]
        return ranges
    
    def _find_best_distribution_for_dp(self, dt_values:List[float], plot:bool = False )->dict:
        """Iteratively fit distributions on the data and choose the best one based on K-S statistic."""
        assert len(dt_values) > 0, "Data property values list cannot be empty for distribution fitting."
        f = Fitter(dt_values, distributions=[
        "cauchy",
        "chi2",
        "expon",
        "exponpow",
        "gamma",
        "lognorm",
        "norm",
        "powerlaw",
        "rayleigh",
    ])
        f.fit()
        best_dist = f.get_best(method="ks_statistic")
        print(best_dist)
        if(plot):
            #distribution fit plot
            f.summary(Nbest=1, method="ks_statistic")
            plt.legend(["function", " data (Histogram)"])
            plt.xlabel(" values")
            plt.ylabel("Density (Data & Probability)")
            plt.title("Best Distribution Fit (Ranked by K-S)")
            plt.show()

        return best_dist
        

    def _extract_refined_ranges_from_data_properties(self, data_property_values:dict, range_values):
        """Extract refined ranges from data property values based on the original range values."""
        lb,ub = range_values
        prop_values = np.sort(np.array(data_property_values))
        #where can maybe be runtime optimized
        lb_index = np.where(prop_values[np.abs(prop_values-lb).argmin()] == prop_values)
        ub_index = np.where(prop_values[np.abs(prop_values-ub).argmin()] == prop_values)
        #if self.verbose>10:
        print("lb:", lb_index[0][0], " ub: ", ub_index[0][0])
        if lb_index[0][0] >= ub_index[0][0]:
            if self.verbose > 0:
                print("Warning: Lower bound and upper bound indices are the same. Refinement may not be effective.")
            return None
        dist = self._find_best_distribution_for_dp(prop_values[lb_index[0][0]:ub_index[0][0]],plot=False)
        return self._compute_dt_pdf_ranges(dist)

    def _extract_ranges_from_data_properties(
        self,
        features:Dict[str, OWLClassExpression],
        individuals_to_feature_mapping,
        data_properties_dict:dict[OWLDataProperty,List[float]],
        per_individual_data_properties:dict[OWLNamedIndividual,dict[OWLDataProperty,float]],):
        """Extract refined ranges from data properties and create new DL concepts based on these ranges."""
        if len(data_properties_dict) == 0:
            return
        ranges_dict:dict[OWLDataProperty,List[Tuple[float,float]]] = dict()
        generated_dt_class_expressions = dict()
        best_dists = dict()

        
        for prop in data_properties_dict:
            best_dist=self._find_best_distribution_for_dp(data_properties_dict[prop])

            print("prop" + prop.__repr__())
            print(best_dist)
            best_dists[prop._iri] = best_dist
            ranges_dict.setdefault(prop, self._compute_dt_pdf_ranges(best_dist),)

        for r in ranges_dict:
            for interval in range(len(ranges_dict[r])):
                new_class_expression = self._pack_data_property_with_range_to_dl_concept(r, ranges_dict[r][interval])
                str_dl_concept = owl_expression_to_dl(new_class_expression)
                # add concept to features
                if str_dl_concept not in features:
                    features[str_dl_concept] = new_class_expression
        # assign concept per individual, if satisfied

        for ind, prop in itertools.product(per_individual_data_properties.keys(), data_properties_dict.keys()):
            if prop in per_individual_data_properties.get(ind, {}):
                for v in per_individual_data_properties[ind][prop]:
                    for interval in ranges_dict[prop]:
                        if interval[0] <= v and v <= interval[1]:
                            new_class_expression = self._pack_data_property_with_range_to_dl_concept(prop, interval)
                            str_dl_concept = owl_expression_to_dl(new_class_expression)
                            individuals_to_feature_mapping[ind.str].add(str_dl_concept)
                            if new_class_expression not in generated_dt_class_expressions:
                                generated_dt_class_expressions[new_class_expression] = set()
                            generated_dt_class_expressions[new_class_expression].add(ind)
                            break
        return generated_dt_class_expressions
    
    def _extract_data_property_features(
        self,
        individual: OWLNamedIndividual,
        features: Dict[str, OWLClassExpression],
        individuals_to_feature_mapping: Dict[str, Set[str]],
        data_properties_dict,
        per_individual_data_properties,
        numeric_ranges: bool = True,
    ):
        """Extract features based on data properties."""
        try:
            # Get data properties for this individual

            for data_prop in self.knowledge_base.get_data_properties_for_ind(individual):
                # Get data property values
                data_values = list(self.knowledge_base.get_data_property_values(individual, data_prop))

                if data_values:
                    # For each data value, we already have features from abox(mode="expression")
                    # This method can be extended to add additional data property features
                    # such as numeric ranges, etc.
                    # TODO: Create new OWL CLassExpressions based on data property values
                    if numeric_ranges:
                        for v in data_values:
                            # check if literal is a numeric
                            if v.is_decimal() or v.is_float() or v.is_integer() or v.is_double():
                                literal = float(v.get_literal())

                                # save data_properties per individual
                                if individual not in per_individual_data_properties:
                                    per_individual_data_properties.setdefault(individual, {})
                                if data_prop not in per_individual_data_properties[individual]:
                                    per_individual_data_properties[individual].setdefault(data_prop, [])

                                per_individual_data_properties[individual][data_prop].append(literal)
                                # save global data property values per property
                                if data_prop not in data_properties_dict:
                                    data_properties_dict[data_prop] = list()
                                data_properties_dict[data_prop].append(literal)
                    # print(f"Data property values for {data_prop}: {data_values}")

        #     print("DEBUG")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Error extracting data property features: {e}")




    
    def extract_expressions_from_owl_individuals(self, individuals: List[OWLNamedIndividual]) -> (Tuple)[np.ndarray, List[OWLClassExpression]]:
        # () Store mappings from str dl concept to owl class expression objects.
        features_dict = dict()
        # () Grouped str dl concepts given str individuals.
        generated_dt_classexpressions_per_individual = dict()
        data_properties_dict = dict()
        per_individual_data_properties = dict()
        individuals_to_feature_mapping = dict()
        for owl_named_individual in make_iterable_verbose(
            individuals,
            verbose=self.verbose,
            desc="Extracting information about examples",
        ):
            # Extract base expressions from ABox
            for owl_class_expression in self.knowledge_base.abox(individual=owl_named_individual, mode="expression"):
                # Apply filters based on configuration flags
                if not self._should_include_expression(owl_class_expression):
                    continue

                # Add the expression as a feature
                self._add_feature(
                    owl_class_expression,
                    owl_named_individual,
                    features_dict,
                    individuals_to_feature_mapping,
                )

            # Generate additional features based on flags
            if self.use_inverse:
                self._extract_inverse_property_features(owl_named_individual, features_dict, individuals_to_feature_mapping)

            if self.use_data_properties:
                self._extract_data_property_features(
                    owl_named_individual,
                    features_dict,
                    individuals_to_feature_mapping,
                    data_properties_dict,
                    per_individual_data_properties,
                )

            if self.use_card_restrictions:
                self._extract_cardinality_features(owl_named_individual, features_dict, individuals_to_feature_mapping)

        # map individuals to additional data property features
        #if self.use_data_properties:
        #    generated_dt_classexpressions_per_individual = self._extract_ranges_from_data_properties(
        #        features_dict,
        #        individuals_to_feature_mapping,
        #        data_properties_dict,
        #        per_individual_data_properties,
        #    )

        if len(features_dict) == 0:
            num_individuals = len(list(make_iterable_verbose(individuals)))
            error_msg = (
                "First hop features cannot be extracted.\n"
                f"  - Number of individuals processed: {num_individuals}\n"
                "  - Number of features extracted: 0\n"
                f"  - use_inverse: {self.use_inverse}\n"
                f"  - use_data_properties: {self.use_data_properties}\n"
                f"  - use_card_restrictions: {self.use_card_restrictions}\n"
                "Possible causes:\n"
                "  - The knowledge base is empty or contains no relevant axioms about the individuals.\n"
                "  - All features were filtered out by configuration flags.\n"
                "  - The individuals provided do not exist in the knowledge base.\n"
                "Please check your configuration and input data."
            )
            raise AssertionError(error_msg)
        if self.verbose > 0:
            print(f"Unique OWL Class Expressions as features: {len(features_dict)}")
            if self.use_inverse:
                print("  - Including inverse property features")
            if self.use_data_properties:
                print("  - Including data property features")
            if self.use_card_restrictions:
                print("  - Including cardinality restriction features")

  
       # Convert features dict to list
        features_list = [v for k, v in features_dict.items()]
        # Construct binary feature matrix
        X = []
        for owl_named_individual in make_iterable_verbose(individuals, verbose=self.verbose, desc="Constructing Training Data"):
            binary_sparse_representation = []
            features_of_owl_named_individual = individuals_to_feature_mapping[owl_named_individual.str]

            for owl_class_expression in features_list:
                if owl_expression_to_dl(owl_class_expression) in features_of_owl_named_individual:
                    binary_sparse_representation.append(1.0)
                else:
                    binary_sparse_representation.append(0.0)
            X.append(binary_sparse_representation)

        X = np.array(X)
        #save needed information
        self.individuals_to_feature_mapping = individuals_to_feature_mapping
        self.per_individual_data_properties = per_individual_data_properties
        #self.generated_dt_classexpressions_per_individual = generated_dt_classexpressions_per_individual
        self.data_properties_dict = data_properties_dict
        return X, features_list
    
    def plot_perm_feature_importances(self, perm_importance_result, feat_name):
        """bar plot the permutation feature importance"""
        feat_name = np.array(feat_name)
        fig, ax = plt.subplots()

        indices = perm_importance_result["importances_mean"].argsort()
        perm_importance = perm_importance_result["importances_mean"][indices[-40:]]
        perm_err = perm_importance_result["importances_std"][indices[-40:]]

        plt.barh(
            range(len(indices[-40:])),
            perm_importance,
            xerr=perm_err,
        )

        ax.set_yticks(range(len(indices[-40:])))
        _ = ax.set_yticklabels(feat_name[indices[-40:]])
        print("show plot")
        plt.show()

    def plot_shap_feature_importances(self, shap_vals, feat_name):
        """Bar plot the mean absolute SHAP feature importance"""
        feat_name = np.array(feat_name)
        indices = np.argsort(shap_vals)
        num_to_plot = min(len(indices), 40)
        top_indices = indices[-num_to_plot:]
        fig, ax = plt.subplots(figsize=(10, 8))

        plt.barh(range(num_to_plot), shap_vals[top_indices], color="skyblue")

        ax.set_yticks(range(num_to_plot))
        ax.set_yticklabels(feat_name[top_indices])
        ax.set_xlabel("Global feature importance")
        ax.set_title("Top DL Feature Importance (SHAP)")

        plt.tight_layout()
        print("Displaying SHAP Importance Plot")
        plt.show()

    def refine_numerical_features(self, topk_expressions:list[OWLClassExpression], iteration:int):
        """Refine numerical features based on data property values and fitted distributions."""
        individuals_to_refined_feature_mapping = dict()
        refined_features = dict()

        topk_expressions = set(topk_expressions)
        for e in topk_expressions:
            if type(e) == OWLDataSomeValuesFrom:
                filler = e.get_filler()
                if type(filler) == OWLDatatypeRestriction:
                    borders = tuple(r.get_facet_value().parse_double() for r in filler.get_facet_restrictions())
                    if self.verbose > 10:
                        print("Range to be refined :", borders)
                        print(e)

                    refined_ranges = self._extract_refined_ranges_from_data_properties(
                        self.data_properties_dict[e.get_property()], borders
                    )

                    if refined_ranges is None:
                        continue
                    
                    for r in refined_ranges:
                        new_class_expression = self._pack_data_property_with_range_to_dl_concept(e.get_property(), r)
                        str_dl_concept = owl_expression_to_dl(new_class_expression)
                        refined_features[str_dl_concept] = new_class_expression

                    individuals_for_e = self.generated_dt_classexpressions_per_individual.get(e)
                    if individuals_for_e is None:
                        prop = e.get_property()
                        lb, ub = borders
                        individuals_for_e = {
                            ind
                            for ind, prop_dict in self.per_individual_data_properties.items()
                            if prop in prop_dict and any(lb <= v <= ub for v in prop_dict[prop])
                        }
                        if self.verbose > 0 and not individuals_for_e:
                            print(f"Warning: no individuals found for expression {owl_expression_to_dl(e)}, skipping.")

                    for ind in individuals_for_e:
                        for v in self.per_individual_data_properties[ind][e.get_property()]:
                            for interval in refined_ranges:
                                if interval[0] <= v <= interval[1]:
                                    new_class_expression = self._pack_data_property_with_range_to_dl_concept(e.get_property(), interval)
                                    string_dl_new_concept = owl_expression_to_dl(new_class_expression)
                                    individuals_to_refined_feature_mapping.setdefault(ind.str, set()).add(string_dl_new_concept)
                                    self.individuals_to_feature_mapping[ind.str].add(string_dl_new_concept)
                                    if new_class_expression not in self.generated_dt_classexpressions_per_individual:
                                        self.generated_dt_classexpressions_per_individual[new_class_expression] = set()
                                    self.generated_dt_classexpressions_per_individual[new_class_expression].add(ind)
        return refined_features,individuals_to_refined_feature_mapping

    def create_training_data(self, learning_problem: PosNegLPStandard) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
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
        # X = self.sparse_binary_representations(features, examples, examples_to_features)
        self.features = features
        X = pd.DataFrame(data=X, index=examples, columns=self.features)
        y = pd.DataFrame(data=y, index=examples, columns=["label"])
        # Remove redundant columns
        same_value_columns = X.apply(lambda col: col.nunique() == 1)
        X = X.loc[:, ~same_value_columns]
        self.features = X.columns.values.tolist()
        return X, y
    
    def plot_importance_evolution(self, initial_dict, refined_dicts, top_k=10):
    #prepare data
        def to_str(obj):
            return str(obj).strip()

        initial_set = {to_str(k) for k in initial_dict.keys()}
        feature_birthdays = {feat: 0 for feat in initial_set}

        for i, d in enumerate(refined_dicts):
            iteration_idx = i + 1
            for feat_obj in d.keys():
                feat_str = to_str(feat_obj)
                if feat_str not in feature_birthdays:
                    feature_birthdays[feat_str] = iteration_idx

        
        last_dict = refined_dicts[-1]
        sorted_feats = sorted(last_dict.items(), key=lambda x: x[1], reverse=True)
        features_to_plot = [to_str(f[0]) for f in sorted_feats[:int(top_k)]]

        plt.figure(figsize=(12, 8))
        labels = ["Initial"] + [f"Iteration {i+1}" for i in range(len(refined_dicts))]

        
        cmap = plt.cm.get_cmap('tab20', len(features_to_plot))
        feature_colors = {feat: cmap(i) for i, feat in enumerate(features_to_plot)}

        for feat in features_to_plot:
            birth_idx = feature_birthdays.get(feat, 0)
            v = []
            color = feature_colors[feat]  #unique color per feature
            is_new = birth_idx > 0

            if birth_idx == 0:
                val = next((_v for _k, _v in initial_dict.items() if to_str(_k) == feat), 0)
                v.append(val)
            else:
                v.append(None)

            for i, d in enumerate(refined_dicts):
                current_step = i + 1
                if current_step < birth_idx:
                    v.append(None)
                else:
                    step_val = next((_v for _k, _v in d.items() if to_str(_k) == feat), 0)
                    v.append(step_val)
           
            plt.plot(labels, v, marker='o', color=color,
                     alpha=0.9 if is_new else 0.6,
                     linewidth=3 if is_new else 1.5,
                     linestyle='--' if is_new else '-',  #dashes for new features
                     zorder=5 if is_new else 2)

            if v[-1] is not None:
                plt.text(len(labels) - 1, v[-1], f" {feat}",
                         va='center', fontsize=9, color=color,
                         fontweight='bold' if is_new else 'normal')

        plt.title(f"Feature Importance Shift (Top {top_k})")
        plt.ylabel("Global SHAP Importance")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _get_shap_importance_dict(self, df):
        """Calculate mean absolute SHAP values for each feature and return as a dictionary.(Per iteration)"""
        tree_explainer = TreeExplainer(self.clf)
        shap_values = tree_explainer.shap_values(df.values)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            importances = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            importances = np.abs(shap_values).mean(axis=0)

        # Map to names: self.features must be synced with the current df columns
        return dict(zip([owl_expression_to_dl(f) for f in self.features], importances))

    def _merge_binary_feature_matrices(self, lp:PosNegLPStandard, X: pd.DataFrame, new_features:list, new_individuals_to_feature_mapping ) -> pd.DataFrame:
        """ Merge the original binary feature matrix with new features extracted from data properties."""
        new_features_list = [v for k,v in new_features.items()]
        positive_examples = [i for i in lp.pos]
        negative_examples = [i for i in lp.neg]
        examples = positive_examples + negative_examples
        X_nf = []
        for owl_named_individual in make_iterable_verbose(examples, verbose=self.verbose, desc="Constructing Training Data for Refined Features"):
            binary_sparse_representation = []
            features_of_owl_named_individual = None
            if owl_named_individual.str in new_individuals_to_feature_mapping:
                features_of_owl_named_individual = new_individuals_to_feature_mapping[owl_named_individual.str]
            for owl_class_expression in new_features_list:
                if features_of_owl_named_individual is not None and owl_expression_to_dl(owl_class_expression) in features_of_owl_named_individual:
                    binary_sparse_representation.append(1.0)
                else:
                    binary_sparse_representation.append(0.0)
            X_nf.append(binary_sparse_representation)
        X_nf = np.array(X_nf)
        X_new = pd.DataFrame(data=X_nf, index=examples, columns=new_features_list)
        X = pd.concat([X, X_new], axis=1)
        self.features = X.columns.to_list()
        same_value_columns = X.apply(lambda col: col.nunique() == 1)
        X = X.loc[:, ~same_value_columns]
        return X

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
        assert isinstance(learning_problem, PosNegLPStandard), f"Learning problem must be PosNegLPStandard. Currently:{learning_problem}."


        if max_runtime is not None:
            self.max_runtime = max_runtime
        self.concept_per_iteration = []
        self.top_feature_dicts = []
        self.initial_importance_dict = {}

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
                param_grid=self.grid_search_over,
                **self.kwargs_grid_search,
            ).fit(X.values, y.values)
            print(grid_search.best_params_)
            self.kwargs_classifier.update(grid_search.best_params_)
        # Training
        if self.verbose > 0:
            print("Training starts!")
        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values)
        self.concept_per_iteration.append(concepts_reducer(concepts=self.construct_owl_expression_from_tree(X, y), reduced_cls=OWLObjectUnionOf))
        if self.feature_refinement:
            tree_explainer = TreeExplainer(self.clf)
            sVal = tree_explainer.shap_values(self.X.values)
            self.initial_importance_dict = self._get_shap_importance_dict(X)

            #refine base features here ( extract from data properties)
            dp_features:dict = dict()
            generated_dt_class_expressions = self._extract_ranges_from_data_properties(dp_features, self.individuals_to_feature_mapping,self.data_properties_dict,self.per_individual_data_properties)
            self.generated_dt_classexpressions_per_individual = generated_dt_class_expressions
            X = self._merge_binary_feature_matrices(learning_problem, X, dp_features, self.individuals_to_feature_mapping)
            self.X = X
            self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=self.X.values, y=self.y.values)
            self.concept_per_iteration.append(concepts_reducer(concepts=self.construct_owl_expression_from_tree(X, y), reduced_cls=OWLObjectUnionOf))
            self.top_feature_dicts.append(self._get_shap_importance_dict(self.X))
            topk = 30
            refined_expressions:Set[OWLClassExpression] = set()
            for i in range(1,4):
                #calculate SHAP global feature importance
                tree_explainer = TreeExplainer(self.clf)
                sVal = tree_explainer.shap_values(self.X.values)
                if isinstance(sVal, np.ndarray):
                    if sVal.ndim == 3:
                        global_importance = np.abs(sVal[:, :, 1]).mean(axis=0)
                    else:
                        global_importance = np.abs(sVal).mean(axis=0)                  
                topk_id = np.argsort(global_importance)
                # top expressions sorted low to high
                top_expressions: list[OWLClassExpression]
                top_expressions = [self.features[i] for i in topk_id[-topk:].tolist()]
                
                #print(top_expressions[-5:])
                refined_features, refined_individuals_to_feature_mapping = self.refine_numerical_features(set(top_expressions)-refined_expressions,iteration=i+1)
                print("Expressions to be refined:" + str(len(set(top_expressions)-refined_expressions)))
                refined_expressions = refined_expressions.union(set(top_expressions))
                X = self._merge_binary_feature_matrices(learning_problem, X, refined_features, refined_individuals_to_feature_mapping)
                self.X = X
                self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=self.X.values, y=self.y.values)
                self.top_feature_dicts.append(self._get_shap_importance_dict(self.X))
                self.concept_per_iteration.append(concepts_reducer(concepts=self.construct_owl_expression_from_tree(X, y), reduced_cls=OWLObjectUnionOf))
            if self.plot_importance_evo:
                self.plot_importance_evolution(self.initial_importance_dict, self.top_feature_dicts, top_k=10)

        if self.report_classification:
            self.__classification_report = "Classification Report: Negatives: -1 and Positives 1 \n"
            self.__classification_report += sklearn.metrics.classification_report(
                y.values,
                self.clf.predict(X.values),
                target_names=["Negative", "Positive"],
            )
            if self.verbose > 0:
                print(self.__classification_report)
        if self.plot_tree:
            plot_decision_tree_of_expressions(
                feature_names=[owl_expression_to_dl(f) for f in self.features],
                cart_tree=self.clf,
            )
        if self.plot_feature_importance:
            plot_topk_feature_importance(
                feature_names=[owl_expression_to_dl(f) for f in self.features],
                cart_tree=self.clf,
                topk=100,
            )
        perm_feature_topk = False
        if perm_feature_topk:
            # plot permutation feature importance
            res = permutation_importance(self.clf, X.values, y.values, n_repeats=10, random_state=0, n_jobs=12)
            self.plot_perm_feature_importances(res, [owl_expression_to_dl(f) for f in self.features])
        shap_feature_topk = False
        # calculate SHAP values, to get topk features
        if shap_feature_topk:
            tree_explainer = TreeExplainer(self.clf)
            sVal = tree_explainer.shap_values(X.values)
            if isinstance(sVal, np.ndarray):
                if sVal.ndim == 3:
                    global_importance = np.abs(sVal[:, :, 1]).mean(axis=0)
                else:
                    global_importance = np.abs(sVal).mean(axis=0)

            self.plot_shap_feature_importances(global_importance, [owl_expression_to_dl(f) for f in self.features])
        self.owl_class_expressions.clear()
        # Each item can be considered is a path of OWL Class Expressions
        # starting from the root node in the decision tree and
        # ending in a leaf node.
        self.conjunctive_concepts: List[OWLObjectIntersectionOf]
        if self.verbose > 0:
            print("Computing conjunctive_concepts...")
        self.conjunctive_concepts = self.construct_owl_expression_from_tree(X, y)
        for i in self.conjunctive_concepts:
            self.owl_class_expressions.add(i)
        if self.verbose > 0:
            print("Computing disjunction_of_conjunctive_concepts...")
        self.disjunction_of_conjunctive_concepts = concepts_reducer(concepts=self.conjunctive_concepts, reduced_cls=OWLObjectUnionOf)
        if self.verbalize:
            verbalize_learner_prediction(self.disjunction_of_conjunctive_concepts)

        return self
