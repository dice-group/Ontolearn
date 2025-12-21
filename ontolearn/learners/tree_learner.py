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
from fitter import Fitter, get_common_distributions
import logging
import scipy as sc
from scipy.stats._distn_infrastructure import rv_frozen
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
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
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
            decision_path.append(
                {
                    "node_id": node_id,
                    "feature_id": feature[node_id],
                    "feature_value_of_individual": np_X[sample_id, feature[node_id]],
                    "inequality": threshold_sign,
                    "threshold_value": threshold[node_id],
                    "owl_expression": owl_class_expression_features[feature[node_id]],
                }
            )

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


def contains_data_properties(expr: OWLClassExpression) -> bool:
    if isinstance(expr, (OWLObjectOneOf, OWLDataHasValue)):
        return False

    """Returns True if the OWL expression contains a literal (OwlDataOneOf)"""
    if isinstance(expr, (OWLDataOneOf, OWLDataHasValue)):
        return True
    if isinstance(expr, HasFiller):
        return contains_data_properties(expr.get_filler())
    # if isinstance(expr, HasOperands):
    #    return any(contains_data_properties(op) for op in expr.get_operands())


def contains_nominal(expr: OWLClassExpression) -> bool:
    """Returns True if the OWL expression contains a nominal (OWLObjectOneOf, OWLObjectHasValue)."""
    if isinstance(expr, (OWLObjectOneOf, OWLObjectHasValue)):
        return True

    # Check operands (for unions, intersections, complements)
    if isinstance(
        expr, (OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectComplementOf)
    ):
        try:
            return any(contains_nominal(op) for op in expr.operands())
        except (AttributeError, TypeError):
            pass

    # Check filler (for restrictions)
    if isinstance(expr, HasFiller):
        return contains_nominal(expr.get_filler())

    return False


def contains_cardinality(expr: OWLClassExpression) -> bool:
    """Returns True if the OWL expression contains a cardinality restriction."""
    if isinstance(
        expr,
        (OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality),
    ):
        return True

    # Check operands (for unions, intersections, complements)
    if isinstance(
        expr, (OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectComplementOf)
    ):
        try:
            return any(contains_cardinality(op) for op in expr.operands())
        except (AttributeError, TypeError):
            pass

    # Check filler (for restrictions)
    if isinstance(expr, HasFiller):
        try:
            return contains_cardinality(expr.get_filler())
        except (AttributeError, TypeError):
            pass

    return False


def contains_data_property(expr: OWLClassExpression) -> bool:
    """Returns True if the OWL expression contains a data property."""
    if isinstance(expr, (OWLDataSomeValuesFrom, OWLDataAllValuesFrom)):
        return True

    # Check operands (for unions, intersections, complements)
    if isinstance(
        expr, (OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectComplementOf)
    ):
        try:
            return any(contains_data_property(op) for op in expr.operands())
        except (AttributeError, TypeError):
            pass

    # Check filler (for restrictions)
    if isinstance(expr, HasFiller):
        try:
            return contains_data_property(expr.get_filler())
        except (AttributeError, TypeError):
            pass

    return False


class TDL:
    """Tree-based Description Logic Concept Learner"""

    def __init__(
        self,
        knowledge_base,
        use_inverse: bool = False,
        use_data_properties: bool = True,
        use_nominals: bool = True,
        use_card_restrictions: bool = True,
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
        verbalize: bool = False,
    ):
        self.use_inverse = use_inverse
        self.use_data_properties = use_data_properties
        self.use_nominals = use_nominals
        self.use_data_properties = use_data_properties
        self.use_card_restrictions = use_card_restrictions
        self.verbose = verbose

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

        assert isinstance(knowledge_base, KnowledgeBase) or isinstance(
            knowledge_base, ontolearn.triple_store.TripleStore
        ), "knowledge_base must be a KnowledgeBase or TripleStore instance"
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
        self.verbalize = verbalize
        self.data_property_cast = dict()
        self.__classification_report = None
        self.X = None
        self.y = None

    def _get_data_property_range(self, values: set) -> tuple:
        """Get the data property range of an OWL class expression"""
        values = list(values)
        min_value = min(values)
        max_value = max(values)
        return [(min_value, max_value)]

    def _pack_data_property_with_range_to_dl_concept(
        self, property: OWLDataProperty, literal_range: tuple
    ) -> str:
        """Repack the ranged data property into an DL Concept String"""
        # print(type(literal_range[0]))
        min_restr = OWLFacetRestriction(
            OWLFacet.MIN_INCLUSIVE,
            OWLLiteral(literal_range[0], OWLDatatype(XSDVocabulary.DOUBLE)),
        )
        if literal_range is None:
            max_restr = OWLFacetRestriction(
                OWLFacet.MAX_INCLUSIVE,
                OWLLiteral(literal_range[0], OWLDatatype(XSDVocabulary.DOUBLE)),
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

    def _should_include_expression(
        self, owl_class_expression: OWLClassExpression
    ) -> bool:
        """Determine if an OWL class expression should be included as a feature based on configuration flags.

        Args:
            owl_class_expression: The OWL class expression to evaluate

        Returns:
            True if the expression should be included, False otherwise
        """
        # Should always include atomic classes
        if isinstance(owl_class_expression, OWLClass):
            return True

        # Exclude expressions containing nominals if flag is disabled
        if not self.use_nominals and contains_nominal(owl_class_expression):
            return False

        # Exclude expressions containing cardinality restrictions if flag is disabled
        if not self.use_card_restrictions and contains_cardinality(
            owl_class_expression
        ):
            return False

        # Exclude expressions containing data properties if flag is disabled
        if not self.use_data_properties and contains_data_property(
            owl_class_expression
        ):
            return False

        return True

    def _add_feature(
        self,
        owl_class_expression: OWLClassExpression,
        owl_named_individual: OWLNamedIndividual,
        features: Dict[str, OWLClassExpression],
        individuals_to_feature_mapping: Dict[str, Set[str]],
    ) -> None:
        """Add an OWL class expression as a feature for the given individual.

        Args:
            owl_class_expression: The OWL class expression to add
            owl_named_individual: The individual this feature applies to
            features: Dictionary mapping DL string representations to OWL expressions
            individuals_to_feature_mapping: Dictionary mapping individuals to their feature sets
        """
        str_dl_concept = owl_expression_to_dl(owl_class_expression)
        individuals_to_feature_mapping.setdefault(owl_named_individual.str, set()).add(
            str_dl_concept
        )
        if str_dl_concept not in features:
            # A mapping from str dl representation to owl object.
            features[str_dl_concept] = owl_class_expression

    def _compute_dt_gaussian_ranges(self, values: set, best_dist:str) -> List[tuple]:
        print(type(list(best_dist.keys())[0]))
        dist_name = list(best_dist.keys())[0]
        distribution = getattr(sc.stats, dist_name)
        frozen_dist: rv_frozen
        frozen_dist = distribution(**best_dist[dist_name])
        mean = frozen_dist.stats(moments='m')
        std = frozen_dist.std()
        ranges = [
            (mean - std, mean),
            (mean, mean + std),
            (mean - 2 * std, mean - 1 * std),
            (mean + std, mean + 2 * std),
        ]
        return ranges

    def _extract_ranges_from_data_properties(
        self,
        individuals,
        features,
        individuals_to_feature_mapping,
        data_properties_dict,
        per_individual_data_properties,
    ):
        if len(data_properties_dict) == 0:
            return
        mode = "gauss"
        ranges_dict = dict()
        for prop in data_properties_dict:
            if mode == "minmax":
                ranges_dict.setdefault(
                    prop, self._get_data_property_range(data_properties_dict[prop])
                )
                
            if mode == "gauss":
                f = Fitter(data_properties_dict[prop], distributions= get_common_distributions())
                f.fit()
                print("prop" + prop.__repr__())
                best_dist = f.get_best(method="sumsquare_error")
                print(best_dist)
                ranges_dict.setdefault(
                    prop, self._compute_dt_gaussian_ranges(data_properties_dict[prop], best_dist)
                )
                
                
        for r in ranges_dict:
            for interval in range(len(ranges_dict[r])):
                new_class_expression = (
                    self._pack_data_property_with_range_to_dl_concept(
                        r, ranges_dict[r][interval]
                    )
                )
                str_dl_concept = owl_expression_to_dl(new_class_expression)
                # add concept to features
                if str_dl_concept not in features:
                    features[str_dl_concept] = new_class_expression
            # assign concept per individual, if satisfied
        for ind in individuals:
            for prop in data_properties_dict:
                if prop in per_individual_data_properties.get(ind, {}):
                    for v in per_individual_data_properties[ind][prop]:
                        for interval in ranges_dict[prop]:
                            if interval[0] <= v and v <= interval[1]:
                                individuals_to_feature_mapping[ind.str].add(
                                    owl_expression_to_dl(
                                        self._pack_data_property_with_range_to_dl_concept(
                                            prop, interval
                                        )
                                    )
                                )
                                break
                            # print(individuals_to_feature_mapping[ind.str])

    def extract_expressions_from_owl_individuals(
        self, individuals: List[OWLNamedIndividual]
    ) -> (Tuple)[np.ndarray, List[OWLClassExpression]]:
        # () Store mappings from str dl concept to owl class expression objects.
        features = dict()
        # () Grouped str dl concepts given str individuals.
        data_properties_dict = dict()
        per_individual_data_properties = dict()
        individuals_to_feature_mapping = dict()
        for owl_named_individual in make_iterable_verbose(
            individuals,
            verbose=self.verbose,
            desc="Extracting information about examples",
        ):
            # Extract base expressions from ABox
            for owl_class_expression in self.knowledge_base.abox(
                individual=owl_named_individual, mode="expression"
            ):
                # Apply filters based on configuration flags
                if not self._should_include_expression(owl_class_expression):
                    continue

                # Add the expression as a feature
                self._add_feature(
                    owl_class_expression,
                    owl_named_individual,
                    features,
                    individuals_to_feature_mapping,
                )

            # Generate additional features based on flags
            if self.use_inverse:
                self._extract_inverse_property_features(
                    owl_named_individual, features, individuals_to_feature_mapping
                )

            if self.use_data_properties:
                self._extract_data_property_features(
                    owl_named_individual,
                    features,
                    individuals_to_feature_mapping,
                    data_properties_dict,
                    per_individual_data_properties,
                )

            if self.use_card_restrictions:
                self._extract_cardinality_features(
                    owl_named_individual, features, individuals_to_feature_mapping
                )
        # map individuals to additional data property features
        if self.use_data_properties:
            self._extract_ranges_from_data_properties(
                individuals,
                features,
                individuals_to_feature_mapping,
                data_properties_dict,
                per_individual_data_properties,
            )

        if len(features) == 0:
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
            print(f"Unique OWL Class Expressions as features: {len(features)}")
            if self.use_inverse:
                print("  - Including inverse property features")
            if self.use_data_properties:
                print("  - Including data property features")
            if self.use_card_restrictions:
                print("  - Including cardinality restriction features")

        # Convert features dict to list
        features_list = [v for k, v in features.items()]

        # Construct binary feature matrix
        X = []
        for owl_named_individual in make_iterable_verbose(
            individuals, verbose=self.verbose, desc="Constructing Training Data"
        ):
            binary_sparse_representation = []
            features_of_owl_named_individual = individuals_to_feature_mapping[
                owl_named_individual.str
            ]

            for owl_class_expression in features_list:
                if (
                    owl_expression_to_dl(owl_class_expression)
                    in features_of_owl_named_individual
                ):
                    binary_sparse_representation.append(1.0)
                else:
                    binary_sparse_representation.append(0.0)
            X.append(binary_sparse_representation)

        X = np.array(X)
        return X, features_list

    def _extract_inverse_property_features(
        self,
        individual: OWLNamedIndividual,
        features: Dict[str, OWLClassExpression],
        individuals_to_feature_mapping: Dict[str, Set[str]],
    ):
        """Extract features based on inverse object properties."""
        try:
            # Get all object properties in the knowledge base
            for obj_prop in self.knowledge_base.get_object_properties():
                # Get inverse property values
                inverse_prop = obj_prop.get_inverse_property()

                # Check if this individual is the object of any property assertion
                # by checking all individuals that have this property pointing to our individual
                for other_ind in self.knowledge_base.individuals():
                    if other_ind == individual:
                        continue
                    try:
                        # Get object property values for the other individual
                        prop_values = list(
                            self.knowledge_base.get_object_property_values(
                                other_ind, obj_prop
                            )
                        )
                        if individual in prop_values:
                            # Create inverse existential restriction: ∃r⁻.⊤
                            inv_exist = OWLObjectSomeValuesFrom(
                                property=inverse_prop,
                                filler=self.knowledge_base.generator.thing,
                            )
                            str_dl_concept = owl_expression_to_dl(inv_exist)
                            individuals_to_feature_mapping.setdefault(
                                individual.str, set()
                            ).add(str_dl_concept)
                            if str_dl_concept not in features:
                                features[str_dl_concept] = inv_exist

                            # Create inverse universal restriction: ∀r⁻.⊤
                            inv_univ = OWLObjectAllValuesFrom(
                                property=inverse_prop,
                                filler=self.knowledge_base.generator.thing,
                            )
                            str_dl_concept = owl_expression_to_dl(inv_univ)
                            individuals_to_feature_mapping.setdefault(
                                individual.str, set()
                            ).add(str_dl_concept)
                            if str_dl_concept not in features:
                                features[str_dl_concept] = inv_univ
                            break  # Found at least one, that's enough for the feature
                    except Exception:
                        continue
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Error extracting inverse property features: {e}")

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

            for data_prop in self.knowledge_base.get_data_properties_for_ind(
                individual
            ):
                # Get data property values
                data_values = list(
                    self.knowledge_base.get_data_property_values(individual, data_prop)
                )

                if data_values:
                    # For each data value, we already have features from abox(mode="expression")
                    # This method can be extended to add additional data property features
                    # such as numeric ranges, etc.
                    # TODO: Create new OWL CLassExpressions based on data property values
                    if numeric_ranges:
                        for v in data_values:
                            # check if literal is a numeric
                            if (
                                v.is_decimal()
                                or v.is_float()
                                or v.is_integer()
                                or v.is_double()
                            ):
                                literal = float(v.get_literal())

                                # save data_properties per individual
                                if individual not in per_individual_data_properties:
                                    per_individual_data_properties.setdefault(
                                        individual, {}
                                    )
                                if (
                                    data_prop
                                    not in per_individual_data_properties[individual]
                                ):
                                    per_individual_data_properties[
                                        individual
                                    ].setdefault(data_prop, [])

                                per_individual_data_properties[individual][
                                    data_prop
                                ].append(literal)
                                # save global data property values per property
                                if data_prop not in data_properties_dict:
                                    data_properties_dict[data_prop] = list()
                                data_properties_dict[data_prop].append(literal)
                    # print(f"Data property values for {data_prop}: {data_values}")

        #     print("DEBUG")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Error extracting data property features: {e}")

    def _extract_cardinality_features(
        self,
        individual: OWLNamedIndividual,
        features: Dict[str, OWLClassExpression],
        individuals_to_feature_mapping: Dict[str, Set[str]],
    ):
        """Extract cardinality restriction features based on object properties."""
        try:
            # Get object properties for this individual
            for obj_prop in self.knowledge_base.get_object_properties_for_ind(
                individual
            ):
                # Count the number of values for this property
                prop_values = list(
                    self.knowledge_base.get_object_property_values(individual, obj_prop)
                )
                count = len(prop_values)

                if count > 0:
                    # Get types of the property values
                    types_counter = Counter()
                    for val in prop_values:
                        val_types = list(
                            self.knowledge_base.get_types(val, direct=True)
                        )
                        for val_type in val_types:
                            types_counter[val_type] += 1

                    # Create min cardinality restrictions for each type
                    for owl_type, type_count in types_counter.items():
                        if type_count >= 2:  # Only create if count >= 2
                            for card in range(2, type_count + 1):
                                min_card = OWLObjectMinCardinality(
                                    cardinality=card, property=obj_prop, filler=owl_type
                                )
                                str_dl_concept = owl_expression_to_dl(min_card)
                                individuals_to_feature_mapping.setdefault(
                                    individual.str, set()
                                ).add(str_dl_concept)
                                if str_dl_concept not in features:
                                    features[str_dl_concept] = min_card

                    # Create general min cardinality with Thing as filler
                    if count >= 2:
                        for card in range(2, count + 1):
                            min_card = OWLObjectMinCardinality(
                                cardinality=card,
                                property=obj_prop,
                                filler=self.knowledge_base.generator.thing,
                            )
                            str_dl_concept = owl_expression_to_dl(min_card)
                            individuals_to_feature_mapping.setdefault(
                                individual.str, set()
                            ).add(str_dl_concept)
                            if str_dl_concept not in features:
                                features[str_dl_concept] = min_card
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Error extracting cardinality features: {e}")

    def create_training_data(
        self, learning_problem: PosNegLPStandard
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        # X = self.sparse_binary_representations(features, examples, examples_to_features)
        self.features = features
        X = pd.DataFrame(data=X, index=examples, columns=self.features)
        y = pd.DataFrame(data=y, index=examples, columns=["label"])
        # Remove redundant columns
        same_value_columns = X.apply(lambda col: col.nunique() == 1)
        X = X.loc[:, ~same_value_columns]
        self.features = X.columns.values.tolist()
        return X, y

    def construct_owl_expression_from_tree(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> List[OWLObjectIntersectionOf]:
        """Construct an OWL class expression from a decision tree"""

        positive_examples: List[OWLNamedIndividual]
        positive_examples = y[y.label == 1].index.tolist()
        vector_representation_of_positive_examples = X.loc[positive_examples]
        prediction_per_example = []
        # () Iterate over reasoning steps of predicting a positive example
        pos: OWLNamedIndividual

        for sequence_of_reasoning_steps, pos in zip(
            make_iterable_verbose(
                explain_inference(
                    self.clf, X=vector_representation_of_positive_examples
                ),
                verbose=self.verbose,
                desc="Constructing Description Logic Concepts",
            ),
            positive_examples,
        ):
            concepts_per_reasoning_step = []
            for i in sequence_of_reasoning_steps:
                if i["inequality"] == ">":
                    owl_class_expression = i["owl_expression"]
                else:
                    owl_class_expression = i[
                        "owl_expression"
                    ].get_object_complement_of()

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
        assert isinstance(learning_problem, PosNegLPStandard), (
            f"Learning problem must be PosNegLPStandard. Currently:{learning_problem}."
        )

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
                param_grid=self.grid_search_over,
                **self.kwargs_grid_search,
            ).fit(X.values, y.values)
            print(grid_search.best_params_)
            self.kwargs_classifier.update(grid_search.best_params_)
        # Training
        if self.verbose > 0:
            print("Training starts!")
        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(
            X=X.values, y=y.values
        )

        if self.report_classification:
            self.__classification_report = (
                "Classification Report: Negatives: -1 and Positives 1 \n"
            )
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
            )

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
        self.disjunction_of_conjunctive_concepts = concepts_reducer(
            concepts=self.conjunctive_concepts, reduced_cls=OWLObjectUnionOf
        )

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
        """Predict the likelihoods of individuals belonging to the classes"""
        raise NotImplementedError(
            "Unavailable. Predict the likelihoods of individuals belonging to the classes"
        )
