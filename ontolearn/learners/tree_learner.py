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
import time
from typing import Dict, Set, Tuple, List, Union, Callable
import numpy as np
import pandas as pd
from ontolearn.verbalizer import verbalize_learner_prediction
from .tdl_pruning import prune_rules
from owlapy.class_expression import (
    OWLObjectIntersectionOf,
    OWLClassExpression,
    OWLObjectUnionOf,
    OWLObjectComplementOf,
    OWLObjectOneOf,
    OWLObjectHasValue,
    OWLObjectSomeValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectExactCardinality,
    OWLDataSomeValuesFrom,
    OWLDataAllValuesFrom,
    OWLClass
)
from owlapy.utils import HasFiller, OWLDataOneOf
from owlapy.owl_individual import OWLNamedIndividual
import ontolearn.triple_store
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
import sklearn
from sklearn import tree
from collections import Counter
from owlapy.owl_datatype import OWLDatatype
from ..utils.static_funcs import plot_umap_reduced_embeddings, plot_decision_tree_of_expressions, \
    plot_topk_feature_importance

import itertools
import logging
from owlapy import owl_expression_to_dl
from ..utils.static_funcs import make_iterable_verbose
from enum import Enum, auto

logger = logging.getLogger(__name__)


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

def contains_nominal(expr: OWLClassExpression) -> bool:
    """Returns True if the OWL expression contains a nominal (OWLObjectOneOf, OWLObjectHasValue)."""
    if isinstance(expr, (OWLObjectOneOf, OWLObjectHasValue)):
        return True

    # Check operands (for unions, intersections, complements)
    if isinstance(expr, (OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectComplementOf)):
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
    if isinstance(expr, (OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality)):
        return True
    
    # Check operands (for unions, intersections, complements)
    if isinstance(expr, (OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectComplementOf)):
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

class PropertyType(Enum):
    BOOLEAN = auto()
    STRING = auto()
    DATE = auto()
    NUMERIC = auto()
    OTHER = auto()

_NUMERIC_TYPES = {
    "integer", "decimal", "float", "double", "long", "int", "short", "byte",
    "nonNegativeInteger", "nonPositiveInteger", "positiveInteger",
    "negativeInteger", "unsignedLong", "unsignedInt", "unsignedShort",
    "unsignedByte", "real", "rational",
}
_STRING_TYPES = {
    "string", "normalizedString", "token", "language",
    "Name", "NCName", "NMTOKEN", "PlainLiteral",
}
_DATE_TYPES = {
    "dateTime", "dateTimeStamp", "date", "time",
    "gYear", "gYearMonth", "gMonthDay", "gDay", "gMonth",
}

def _classify_datatype_name(name: str) -> PropertyType:
    """Map an XSD datatype IRI remainder to a DataPropertyType category."""
    if name == "boolean":
        return PropertyType.BOOLEAN
    if name in _NUMERIC_TYPES:
        return PropertyType.NUMERIC
    if name in _STRING_TYPES:
        return PropertyType.STRING
    if name in _DATE_TYPES:
        return PropertyType.DATE
    return PropertyType.OTHER


def _classify_literal(literal) -> PropertyType:
    """Map an OWLLiteral to a DataPropertyType category."""
    if literal.is_boolean():
        return PropertyType.BOOLEAN
    if literal.is_string():
        return PropertyType.STRING
    if literal.is_integer() or literal.is_double():
        return PropertyType.NUMERIC
    # Fall back to classifying by the literal's datatype IRI
    try:
        return _classify_datatype_name(literal.get_datatype().iri.remainder)
    except AttributeError:
        return PropertyType.OTHER


def contains_data_property(expr: OWLClassExpression, include_boolean:bool = False) -> PropertyType:
    """Returns True if the OWL expression contains a data property."""
    found: Set[PropertyType] = set()
    if isinstance(expr, (OWLDataSomeValuesFrom, OWLDataAllValuesFrom)):
        filler = expr.get_filler()
        if isinstance(filler, OWLDataOneOf):
            found.update(_classify_literal(literal) for literal in filler.values())
        elif isinstance(filler, OWLDatatype):
            found.add(_classify_datatype_name(filler.iri.remainder))
        else:
            found.add(PropertyType.OTHER)
    elif isinstance(expr, OWLObjectHasValue):
        found.add(_classify_literal(expr.get_filler()))

    elif isinstance(expr, (OWLObjectIntersectionOf, OWLObjectUnionOf, OWLObjectComplementOf)):
        for op in expr.operands():
            sub = contains_data_property(op)
            if sub:
                found.update(sub)
    
    elif isinstance(expr, OWLObjectComplementOf):
        sub = contains_data_property(expr.get_operand())
        if sub:
            found.update(sub)
    elif isinstance(expr, HasFiller):
        try:
            sub = contains_data_property(expr.get_filler())
            if sub:
                found.update(sub)
        except (AttributeError, TypeError):
            pass
    return found


HYPOTHESIS_EXTRACTION_MODES = ("positive_paths", "predicted_leaves", "factored_tree", "pruned_leaves")
r"""Ways of turning the trained decision tree into one OWL class expression (``TDL(hypothesis_extraction=...)``).

Every split of the tree tests one *binary* feature, i.e. one OWL class expression ``C`` that an individual either
satisfies (right branch, "yes", sklearn: ``> 0.5``) or not (left branch, "no", sklearn: ``<= 0.5``, read as ``¬C``).
A leaf is labelled with the class that holds the majority of the training examples that reach it
(ties go to the negative class, like ``DecisionTreeClassifier.predict``).

Worked example. Features ``B``, ``C``; leaves are shown as ``(#positives/#negatives) -> predicted class``::

                           C ?
                  no /           \ yes
                   B ?            L3
              no /    \ yes    (8/0) +
             L1        L2
          (5/0) +   (1/3) -

    L1 = ¬C ⊓ ¬B        L2 = ¬C ⊓ B        L3 = C

``"positive_paths"`` (default, the original behaviour)
    For every *positive training example* take the root-to-leaf path it follows, join the decisions with ⊓ and
    union the distinct paths. A leaf is used as soon as one positive example lands in it, whatever class it predicts.
    Example: the lone positive of L2 pulls L2 in::

        (¬C ⊓ ¬B) ⊔ (¬C ⊓ B) ⊔ C          # covers the 3 negatives of L2, too

    This favours recall. It fails for trees without a split (all paths are empty).

``"predicted_leaves"``
    Same shape of result, but a leaf contributes only if it *predicts* the positive class (majority vote), and the
    leaves are visited depth-first, left to right (deterministic order). Example::

        (¬C ⊓ ¬B) ⊔ C                      # L2 is dropped: 1 positive vs 3 negatives

    A tree that is a single leaf yields ⊤ (positive) or ⊥ (negative).

``"factored_tree"``
    Folds the tree bottom-up into a nested expression instead of listing leaves, and simplifies on the way
    (Shannon expansion ``(¬C ⊓ low) ⊔ (C ⊓ high)`` with ⊤/⊥ propagation). The positive region of the example is::

        node B:  low = ⊤ (L1), high = ⊥ (L2)  ->  (¬B ⊓ ⊤) ⊔ (B ⊓ ⊥)             ->  ¬B
        node C:  low = ¬B,     high = ⊤ (L3)  ->  (¬C ⊓ ¬B) ⊔ (C ⊓ ⊤)  =  C ⊔ ¬B   (absorption: C ⊔ (¬C ⊓ x) = C ⊔ x)

    The same fold on the *negative* region gives ¬C ⊓ B, and its complement ``¬(¬C ⊓ B)`` is logically equivalent
    to the positive region (the leaves partition all individuals). The shorter of the two is returned: here
    ``C ⊔ ¬B``; for trees with many positive leaves and few negative ones the complement usually wins
    (e.g. ``¬(¬C ⊓ B)`` instead of a union of three conjunctions). It makes the same predictions as
    ``predicted_leaves``; only the notation is shorter.

``"pruned_leaves"``  (NOT part of the published TDL algorithm)
    The ``predicted_leaves`` expression, derefined on the training examples by reduced-error pruning
    (:func:`ontolearn.learners.tdl_pruning.prune_rules`): literals and whole conjunctions are dropped as long as the
    training F1 does not fall by more than ``prune_tolerance`` (default 0: the training F1 is kept and only
    redundancy is removed). Example: in the tree above, if every training example that satisfies ``¬C`` is positive
    whatever ``B`` says, ``¬C ⊓ ¬B`` is shortened to ``¬C``. This is an addition made in Ontolearn after TDL was published. It is
    not from a peer-reviewed work and has only been evaluated in Ontolearn (issue #672), where it gave expressions
    about 1.2x shorter at the geometric mean and retrieved about 1.6x faster (``prune_tolerance=0``), with no
    significant change in held-out F1 (-0.0006, 20 problems). Opt-in; the default mode is unchanged.

Held-out F1 observed in 5-fold cross-validation (see ``tests/test_tdl_hypothesis_extraction.py``)::

                                    positive_paths  predicted_leaves  factored_tree
    Family, 18 problems, full tree     0.965            0.965            0.965   (factored_tree ~35% shorter)
    Family, 18 problems, depth 3       0.949            0.951            0.951
    Carcinogenesis, full tree          0.620            0.620            0.620   (factored_tree ~2.7x shorter)
    Carcinogenesis, depth 3            0.705            0.592            0.592

The leaf-based modes win when shallow trees have impure leaves that mostly hold negatives; ``positive_paths`` wins when
recall matters (it keeps every leaf that holds a positive example).
"""


class TDL:
    """Tree-based Description Logic Concept Learner"""

    def __init__(self, knowledge_base,
                 use_inverse: bool = True,
                 use_data_properties_boolean:bool = True,
                 use_data_properties_string: bool = True,
                 use_data_properties_date: bool = True,
                 use_data_properties_numeric: bool = True,
                 use_nominals: bool = True,
                 use_card_restrictions: bool = True,
                 kwargs_classifier: dict = None,
                 max_runtime: int = None,
                 grid_search_over: dict = None,
                 grid_search_apply: bool = False,
                 kwargs_grid_search: dict = None,
                 report_classification: bool = True,
                 plot_tree: bool = False,
                 plot_embeddings: bool = False,
                 plot_feature_importance: bool = False,
                 verbose: int = 10,
                 verbalize: bool = False,
                 hypothesis_extraction: str = "positive_paths",
                 prune_tolerance: float = 0.0):

        if hypothesis_extraction not in HYPOTHESIS_EXTRACTION_MODES:
            raise ValueError(f"hypothesis_extraction must be one of {HYPOTHESIS_EXTRACTION_MODES}. "
                             f"Currently: {hypothesis_extraction!r}")
        self.hypothesis_extraction = hypothesis_extraction
        # Only used by hypothesis_extraction="pruned_leaves" (not part of the published TDL, see HYPOTHESIS_EXTRACTION_MODES)
        self.prune_tolerance = prune_tolerance
        self.use_inverse = use_inverse
        self.use_data_properties_boolean = use_data_properties_boolean
        self.use_data_properties_string = use_data_properties_string
        self.use_data_properties_date = use_data_properties_date
        self.use_data_properties_numeric = use_data_properties_numeric
        self.use_nominals = use_nominals
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

        # Fresh copy: avoids sharing/mutating a default (or caller-owned) dict across instances
        kwargs_grid_search = dict(kwargs_grid_search) if kwargs_grid_search is not None else {}
        kwargs_grid_search.setdefault("cv", 10)
        
        assert (
                isinstance(knowledge_base, KnowledgeBase)
                or isinstance(knowledge_base, ontolearn.triple_store.TripleStore)
        ), "knowledge_base must be a KnowledgeBase or TripleStore instance"
        logger.info(f"Knowledge Base: {knowledge_base}")
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

    def _should_include_expression(self, owl_class_expression: OWLClassExpression) -> bool:
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
        if not self.use_card_restrictions and contains_cardinality(owl_class_expression):
            return False
        
        # exclude expressions containing specific data properties
        property_types = contains_data_property(owl_class_expression)
        data_property_toggles = {
                PropertyType.BOOLEAN:  self.use_data_properties_boolean,
                PropertyType.STRING:   self.use_data_properties_string,
                PropertyType.DATE: self.use_data_properties_date,
                PropertyType.NUMERIC:  self.use_data_properties_numeric,
            }
        if not all(data_property_toggles.get(t, False) for t in property_types):
            return False
        
        return True
    
    def _add_feature(self, owl_class_expression: OWLClassExpression, 
                     owl_named_individual: OWLNamedIndividual,
                     features: Dict[str, OWLClassExpression],
                     individuals_to_feature_mapping: Dict[str, Set[str]]) -> None:
        """Add an OWL class expression as a feature for the given individual.
        
        Args:
            owl_class_expression: The OWL class expression to add
            owl_named_individual: The individual this feature applies to
            features: Dictionary mapping DL string representations to OWL expressions
            individuals_to_feature_mapping: Dictionary mapping individuals to their feature sets
        """
        str_dl_concept = owl_expression_to_dl(owl_class_expression)
        individuals_to_feature_mapping.setdefault(owl_named_individual.str, set()).add(str_dl_concept)
        if str_dl_concept not in features:
            # A mapping from str dl representation to owl object.
            features[str_dl_concept] = owl_class_expression

    def extract_expressions_from_owl_individuals(self, individuals: List[OWLNamedIndividual], start_time:float=None) -> (
            Tuple)[np.ndarray, List[OWLClassExpression]]:
        # () Store mappings from str dl concept to owl class expression objects.
        features = dict()
        # () Grouped str dl concepts given str individuals.
        individuals_to_feature_mapping = dict()
        
        for owl_named_individual in make_iterable_verbose(individuals,
                                       verbose=self.verbose,
                                       desc="Extracting information about examples"):
            #check if max_runtime exceeded
            if self.max_runtime and (time.time() - start_time > self.max_runtime):
                raise RuntimeError(f"Execution terminated: max_runtime of {self.max_runtime} seconds exceeded.")

            # Extract base expressions from ABox
            for owl_class_expression in self.knowledge_base.abox(individual=owl_named_individual, mode="expression"):
                # Apply filters based on configuration flags
                if not self._should_include_expression(owl_class_expression):
                    continue
                
                # Add the expression as a feature
                self._add_feature(owl_class_expression, owl_named_individual, features, individuals_to_feature_mapping)

            # Generate additional features based on flags
            if self.use_inverse:
                self._extract_inverse_property_features(owl_named_individual, features, individuals_to_feature_mapping)
            
            if self.use_data_properties_string or self.use_data_properties_date or self.use_data_properties_numeric or self.use_data_properties_boolean:
                self._extract_data_property_features(owl_named_individual, features, individuals_to_feature_mapping)
            
            if self.use_card_restrictions:
                self._extract_cardinality_features(owl_named_individual, features, individuals_to_feature_mapping)

        if len(features) == 0:
            num_individuals = len(list(make_iterable_verbose(individuals)))
            error_msg = (
                "First hop features cannot be extracted.\n"
                f"  - Number of individuals processed: {num_individuals}\n"
                "  - Number of features extracted: 0\n"
                f"  - use_inverse: {self.use_inverse}\n"
                f"  - use_data_properties_boolean: {self.use_data_properties_boolean}\n"
                f"  - use_data_properties_string: {self.use_data_properties_string}\n"
                f"  - use_data_properties_date: {self.use_data_properties_date}\n"
                f"  - use_data_properties_numeric: {self.use_data_properties_numeric}\n"
                f"  - use_card_restrictions: {self.use_card_restrictions}\n"
                "Possible causes:\n"
                "  - The knowledge base is empty or contains no relevant axioms about the individuals.\n"
                "  - All features were filtered out by configuration flags.\n"
                "  - The individuals provided do not exist in the knowledge base.\n"
                "Please check your configuration and input data."
            )
            raise AssertionError(error_msg)
        if self.verbose > 0:
            logger.info(f"Unique OWL Class Expressions as features: {len(features)}")
            feature_options = {
                "inverse properties":        self.use_inverse,
                "nominals":                  self.use_nominals,
                "cardinality restrictions":  self.use_card_restrictions,
                "boolean data properties":   self.use_data_properties_boolean,
                "string data properties":    self.use_data_properties_string,
                "date data properties":      self.use_data_properties_date,
                "numeric data properties":   self.use_data_properties_numeric,
                }
            enabled = [name for name, flag in feature_options.items() if flag]

            if enabled:
                logger.info(f"Enabled feature types: {', '.join(enabled)}")
        
        # Convert features dict to list
        # Sort by the DL string: the insertion order of `features` follows set iteration inside the knowledge base
        # (hash-seed dependent). A fixed column order makes tie-breaking in the decision tree, and hence the learned
        # expression, reproducible across processes.
        features_list = [features[k] for k in sorted(features)]
       
        # Construct binary feature matrix
        X = []
        for owl_named_individual in make_iterable_verbose(individuals,
                                       verbose=self.verbose,
                                       desc="Constructing Training Data"):
            if self.max_runtime and (time.time() - start_time > self.max_runtime):
                raise RuntimeError(f"Execution terminated: max_runtime of {self.max_runtime} seconds exceeded.")

            binary_sparse_representation = []
            features_of_owl_named_individual = individuals_to_feature_mapping[owl_named_individual.str]

            for owl_class_expression in features_list:
                if owl_expression_to_dl(owl_class_expression) in features_of_owl_named_individual:
                    binary_sparse_representation.append(1.0)
                else:
                    binary_sparse_representation.append(0.0)
            X.append(binary_sparse_representation)
        
        X = np.array(X)
        return X, features_list

    def _extract_inverse_property_features(self, individual: OWLNamedIndividual, 
                                          features: Dict[str, OWLClassExpression],
                                          individuals_to_feature_mapping: Dict[str, Set[str]]):
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
                        # Get object property values for the other individual. This is the
                        # only reasoner call in the loop body, so only it is allowed to fail
                        # and be skipped - a bug in the feature-construction code below must
                        # propagate instead of being silently treated as "no result found".
                        prop_values = list(self.knowledge_base.get_object_property_values(other_ind, obj_prop))
                    except Exception:
                        logger.debug("Could not fetch object property values for %s, %s",
                                     other_ind, obj_prop, exc_info=True)
                        continue
                    if individual in prop_values:
                        # Create inverse existential restriction: ∃r⁻.⊤
                        inv_exist = OWLObjectSomeValuesFrom(property=inverse_prop,
                                                           filler=self.knowledge_base.generator.thing)
                        str_dl_concept = owl_expression_to_dl(inv_exist)
                        individuals_to_feature_mapping.setdefault(individual.str, set()).add(str_dl_concept)
                        if str_dl_concept not in features:
                            features[str_dl_concept] = inv_exist

                        # No universal restriction ∀r⁻.⊤ is added: with the filler ⊤ it is a tautology (it holds for
                        # every individual, including those without a predecessor), so a column that is 1 only
                        # for individuals with a predecessor would not describe the expression it is named after.
                        break  # Found at least one, that's enough for the feature
        except Exception as e:
            logger.warning("Error extracting inverse property features for %s: %s", individual, e, exc_info=True)

    def _extract_data_property_features(self, individual: OWLNamedIndividual,
                                       features: Dict[str, OWLClassExpression],
                                       individuals_to_feature_mapping: Dict[str, Set[str]]):
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
                   # print(f"Data property values for {data_prop}: {data_values}")
                    pass
        except Exception as e:
            logger.warning("Error extracting data property features for %s: %s", individual, e, exc_info=True)

    def _extract_cardinality_features(self, individual: OWLNamedIndividual,
                                     features: Dict[str, OWLClassExpression],
                                     individuals_to_feature_mapping: Dict[str, Set[str]]):
        """Extract cardinality restriction features based on object properties."""
        try:
            # Get object properties for this individual
            for obj_prop in self.knowledge_base.get_object_properties_for_ind(individual):
                # Count the number of values for this property
                prop_values = list(self.knowledge_base.get_object_property_values(individual, obj_prop))
                count = len(prop_values)
                
                if count > 0:
                    # Get types of the property values
                    types_counter = Counter()
                    for val in prop_values:
                        val_types = list(self.knowledge_base.get_types(val, direct=True))
                        for val_type in val_types:
                            types_counter[val_type] += 1
                    
                    # Create min cardinality restrictions for each type
                    for owl_type, type_count in types_counter.items():
                        if type_count >= 2:  # Only create if count >= 2
                            for card in range(2, type_count + 1):
                                min_card = OWLObjectMinCardinality(cardinality=card,
                                                                  property=obj_prop,
                                                                  filler=owl_type)
                                str_dl_concept = owl_expression_to_dl(min_card)
                                individuals_to_feature_mapping.setdefault(individual.str, set()).add(str_dl_concept)
                                if str_dl_concept not in features:
                                    features[str_dl_concept] = min_card
                    
                    # Create general min cardinality with Thing as filler
                    if count >= 2:
                        for card in range(2, count + 1):
                            min_card = OWLObjectMinCardinality(cardinality=card,
                                                              property=obj_prop,
                                                              filler=self.knowledge_base.generator.thing)
                            str_dl_concept = owl_expression_to_dl(min_card)
                            individuals_to_feature_mapping.setdefault(individual.str, set()).add(str_dl_concept)
                            if str_dl_concept not in features:
                                features[str_dl_concept] = min_card
        except Exception as e:
            logger.warning("Error extracting cardinality features for %s: %s", individual, e, exc_info=True)


    def create_training_data(self, learning_problem: PosNegLPStandard, start_time:float=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # (1) Initialize ordering over positive and negative examples.
        if self.verbose > 0:
            logger.info("Creating a Training Dataset")
        positive_examples: List[OWLNamedIndividual]
        negative_examples: List[OWLNamedIndividual]
        positive_examples = [i for i in learning_problem.pos]
        negative_examples = [i for i in learning_problem.neg]
        # (2) Initialize labels for (1).
        y = [1.0 for _ in positive_examples] + [0.0 for _ in negative_examples]
        # (3) Iterate over examples to extract unique features.
        examples = positive_examples + negative_examples
        # For the sake of convenience. sort features in ascending order of string lengths of DL representations.
        X, features = self.extract_expressions_from_owl_individuals(examples, start_time)
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

    def construct_owl_expression_from_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLClassExpression]:
        """Construct OWL class expressions from the trained decision tree.

        The strategy is selected by ``self.hypothesis_extraction`` (see ``HYPOTHESIS_EXTRACTION_MODES`` for a worked
        example). The hypothesis is the union of the returned expressions.

        :param X: binary feature matrix the tree was trained on; its columns are the OWL class expressions
        :param y: labels (column ``label``: 1 positive, 0 negative)
        :return: ``positive_paths``/``predicted_leaves``: one conjunction per used leaf; ``factored_tree``: a single
                 (factored) expression
        """
        if self.hypothesis_extraction == "predicted_leaves":
            return self._expressions_from_predicted_leaves(X)
        if self.hypothesis_extraction == "factored_tree":
            return [self._factored_expression_from_tree(X)]
        if self.hypothesis_extraction == "pruned_leaves":
            return self._expressions_from_pruned_leaves(X, y)
        return self._expressions_from_positive_paths(X, y)

    def _tree_feature_expression(self, X: pd.DataFrame, node_id: int) -> OWLClassExpression:
        """The OWL class expression tested at the internal node `node_id` (a column of X)."""
        return X.columns[self.clf.tree_.feature[node_id]]

    def _is_positive_leaf(self, node_id: int) -> bool:
        """True iff the leaf predicts the positive class.

        The class with the most training examples in the leaf wins; ties go to the negative class because
        ``np.argmax`` returns the first maximum and ``classes_`` is sorted ([0, 1]), as in sklearn's ``predict``.
        """
        counts = self.clf.tree_.value[node_id, 0]
        return bool(self.clf.classes_[int(np.argmax(counts))] == 1)

    def _expressions_from_predicted_leaves(self, X: pd.DataFrame) -> List[OWLClassExpression]:
        r"""Mode ``predicted_leaves``: one conjunction per leaf that predicts the positive class.

        Example tree (leaf labels: #positives/#negatives)::

                   C ?
              no /     \ yes
               B ?       (8/0) +
          no /   \ yes
         (5/0) +  (1/3) -

            result: [ ¬C ⊓ ¬B,  C ]       hypothesis = (¬C ⊓ ¬B) ⊔ C    (the (1/3) leaf is dropped)

        Leaves are visited depth-first, left ("no") before right ("yes"), so the output is deterministic.
        A tree consisting of one leaf has an empty path: ⊤ if the leaf is positive, otherwise ⊥ is returned.
        """
        tree_ = self.clf.tree_
        thing, nothing = self.knowledge_base.generator.thing, self.knowledge_base.generator.nothing
        conjunctions: List[OWLClassExpression] = []

        def visit(node_id: int, path: List[OWLClassExpression]):
            # sklearn marks leaves with children_left == -1
            if tree_.children_left[node_id] == -1:
                if self._is_positive_leaf(node_id):
                    # An empty path means the root is the leaf: nothing restricts the individuals -> Thing.
                    conjunctions.append(concepts_reducer(path, OWLObjectIntersectionOf) if path else thing)
                return
            feature = self._tree_feature_expression(X, node_id)
            # Binary features: "<= threshold" means the feature does not hold, "> threshold" means it holds.
            visit(tree_.children_left[node_id], path + [feature.get_object_complement_of()])
            visit(tree_.children_right[node_id], path + [feature])

        visit(0, [])
        # No positive leaf at all: the hypothesis is the empty class.
        return conjunctions if conjunctions else [nothing]

    def _expressions_from_pruned_leaves(self, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLClassExpression]:
        """Mode ``pruned_leaves`` (an Ontolearn addition, not part of the published TDL): the positive leaves of the
        tree as in ``predicted_leaves``, derefined on the training rows with
        :func:`ontolearn.learners.tdl_pruning.prune_rules` (tolerance ``self.prune_tolerance``)."""
        tree_ = self.clf.tree_
        thing, nothing = self.knowledge_base.generator.thing, self.knowledge_base.generator.nothing
        rules = []

        def visit(node_id: int, path: tuple):
            if tree_.children_left[node_id] == -1:
                if self._is_positive_leaf(node_id):
                    rules.append(path)
                return
            feature = int(tree_.feature[node_id])
            visit(tree_.children_left[node_id], path + ((feature, False),))
            visit(tree_.children_right[node_id], path + ((feature, True),))

        visit(0, ())
        if not rules:
            return [nothing]
        rules = prune_rules(rules, X.values, np.asarray(y).ravel().astype(int), self.prune_tolerance)
        return [concepts_reducer([X.columns[c] if holds else X.columns[c].get_object_complement_of() for c, holds in rule],
                                 OWLObjectIntersectionOf) if rule else thing for rule in rules]

    def _factored_expression_from_tree(self, X: pd.DataFrame) -> OWLClassExpression:
        r"""Mode ``factored_tree``: fold the tree into one simplified, nested class expression.

        For a node testing ``C`` with sub-expressions ``low`` (for ¬C) and ``high`` (for C) describing where the
        tree predicts the wanted class::

            region(node) = (¬C ⊓ low) ⊔ (C ⊓ high)

        with the simplifications ``low == high -> low``, ``⊤ ⊓ x -> x``, ``⊥ ⊓ x -> ⊥``, ``⊥ ⊔ x -> x``,
        ``⊤ ⊔ x -> ⊤``, ``¬C ⊔ (C ⊓ h) -> ¬C ⊔ h`` and ``C ⊔ (¬C ⊓ l) -> C ⊔ l``.

        Example (leaf labels #positives/#negatives; ``+`` = predicted positive)::

                   C ?                      region(B) = (¬B ⊓ ⊤) ⊔ (B ⊓ ⊥)  = ¬B
              no /     \ yes               region(C) = (¬C ⊓ ¬B) ⊔ (C ⊓ ⊤) = C ⊔ ¬B
               B ?       (8/0) +
          no /   \ yes
         (5/0) +  (1/3) -

        The negative region (where the tree predicts the negative class) is folded the same way and complemented:
        ``¬(¬C ⊓ B)``. Because the leaves partition all individuals this is equivalent to the positive region, and
        the shorter notation is returned (ties keep the positive region): here ``C ⊔ ¬B``.
        """
        tree_ = self.clf.tree_
        thing, nothing = self.knowledge_base.generator.thing, self.knowledge_base.generator.nothing

        def neg(e):
            """Complement with ⊤/⊥ handled explicitly."""
            return nothing if e == thing else thing if e == nothing else e.get_object_complement_of()

        def conj(a, b):
            """a ⊓ b with ⊤ as neutral and ⊥ as absorbing element."""
            if a == nothing or b == nothing:
                return nothing
            return b if a == thing else a if b == thing else OWLObjectIntersectionOf((a, b))

        def disj(a, b):
            """a ⊔ b with ⊥ as neutral and ⊤ as absorbing element."""
            if a == thing or b == thing:
                return thing
            return b if a == nothing else a if b == nothing else OWLObjectUnionOf((a, b))

        def region(node_id: int, positive: bool):
            """Expression for the individuals the subtree below `node_id` assigns to the wanted class."""
            if tree_.children_left[node_id] == -1:
                return thing if self._is_positive_leaf(node_id) == positive else nothing
            feature = self._tree_feature_expression(X, node_id)
            low = region(tree_.children_left[node_id], positive)  # individuals NOT satisfying `feature`
            high = region(tree_.children_right[node_id], positive)  # individuals satisfying `feature`
            if low == high:  # the split does not matter for the outcome
                return low
            if low == thing:  # ¬C ⊔ (C ⊓ high) = ¬C ⊔ high
                return disj(neg(feature), high)
            if high == thing:  # C ⊔ (¬C ⊓ low) = C ⊔ low
                return disj(feature, low)
            return disj(conj(neg(feature), low), conj(feature, high))

        def size(e):
            return len(owl_expression_to_dl(e))

        positive_region = region(0, True)
        # Equivalent notation: everything that is not predicted negative.
        complemented_negative_region = neg(region(0, False))
        return complemented_negative_region if size(complemented_negative_region) < size(positive_region) else positive_region

    def _expressions_from_positive_paths(self, X: pd.DataFrame, y: pd.DataFrame) -> List[OWLObjectIntersectionOf]:
        """Mode ``positive_paths`` (default): union of the root-to-leaf paths of the positive training examples.

        Example (same tree as in ``_expressions_from_predicted_leaves``; the leaf (1/3) holds one positive)::

            result: [ ¬C ⊓ ¬B,  ¬C ⊓ B,  C ]       hypothesis = (¬C ⊓ ¬B) ⊔ (¬C ⊓ B) ⊔ C

        The leaf is used because a positive example lands in it, although it predicts the negative class and
        covers three negatives. Recall-oriented. Limitation: for a tree without a split every path is empty.
        """

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

    def _maybe_plot_embeddings(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Plot a UMAP-reduced view of the training features, if enabled. Shared by TDL and TDL_refinement."""
        if self.plot_embeddings:
            plot_umap_reduced_embeddings(X, y.label.to_list(), "umap_visualization.pdf")

    def _fit_decision_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Optionally grid-search classifier hyperparameters, then train self.clf on (X, y).
        Shared by TDL and TDL_refinement.

        If no informative feature is left (every extracted feature has the same value for all examples),
        no tree can be trained: self.clf is set to None and the learner falls back to the trivial hypothesis."""
        if X.shape[1] == 0:
            logger.warning("No informative features: all extracted features take the same value for the positive and "
                           "negative examples. Falling back to the trivial hypothesis (Thing). Consider enabling more "
                           "feature extractors (inverse properties, data properties, nominals, cardinality restrictions).")
            self.clf = None
            return
        if self.grid_search_over:
            grid_search = sklearn.model_selection.GridSearchCV(
                tree.DecisionTreeClassifier(**self.kwargs_classifier),
                param_grid=self.grid_search_over, **self.kwargs_grid_search).fit(X.values, y.values)
            logger.info(f"Best grid search parameters: {grid_search.best_params_}")
            self.kwargs_classifier.update(grid_search.best_params_)
        # Training
        if self.verbose > 0:
            logger.info("Training starts!")
        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values)

    def _report_and_plot(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Report classification metrics and render the configured tree/feature-importance plots.
        Shared by TDL and TDL_refinement."""
        if self.clf is None:
            self.__classification_report = "No classification report: no informative features, no tree was trained."
            return
        if self.report_classification:
            self.__classification_report = "Classification Report: Negatives: -1 and Positives 1 \n"
            self.__classification_report += sklearn.metrics.classification_report(y.values,
                                                                                  self.clf.predict(X.values),
                                                                                  target_names=["Negative",
                                                                                                "Positive"])
            if self.verbose > 0:
                logger.info(self.__classification_report)
        if self.plot_tree:
            plot_decision_tree_of_expressions(feature_names=[owl_expression_to_dl(f) for f in self.features],
                                              cart_tree=self.clf)
        if self.plot_feature_importance:
            plot_topk_feature_importance(feature_names=[owl_expression_to_dl(f) for f in self.features],
                                         cart_tree=self.clf)

    def _finalize_predictions(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Build the conjunctive/disjunctive OWL predictions from the trained tree and optionally verbalize them.
        Shared by TDL and TDL_refinement."""
        self.owl_class_expressions.clear()
        if self.clf is None:
            # No tree was trained (no informative features): the only hypothesis available is Thing.
            thing = self.knowledge_base.generator.thing
            self.conjunctive_concepts = [thing]
            self.owl_class_expressions.add(thing)
            self.disjunction_of_conjunctive_concepts = thing
            if self.verbalize:
                verbalize_learner_prediction(thing)
            return
        # Each item can be considered is a path of OWL Class Expressions
        # starting from the root node in the decision tree and
        # ending in a leaf node.
        self.conjunctive_concepts: List[OWLObjectIntersectionOf]
        if self.verbose > 0:
            logger.info("Computing conjunctive_concepts...")
        self.conjunctive_concepts = self.construct_owl_expression_from_tree(X, y)
        for i in self.conjunctive_concepts:
            self.owl_class_expressions.add(i)
        if self.verbose > 0:
            logger.info("Computing disjunction_of_conjunctive_concepts...")
        self.disjunction_of_conjunctive_concepts = concepts_reducer(concepts=self.conjunctive_concepts,  reduced_cls=OWLObjectUnionOf)

        if self.verbalize:
            verbalize_learner_prediction(self.disjunction_of_conjunctive_concepts)

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

        start_time = time.time()
        X: pd.DataFrame
        y: Union[pd.DataFrame, pd.Series]
        X, y = self.create_training_data(learning_problem=learning_problem, start_time=start_time)
        # CD: Remember so that if user wants to use them
        self.X, self.y = X, y
        self._maybe_plot_embeddings(X, y)
        self._fit_decision_tree(X, y)
        self._report_and_plot(X, y)
        self._finalize_predictions(X, y)

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
