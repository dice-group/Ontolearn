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
import os
import pandas
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import traceback

from itertools import chain
from typing import Any, Optional, Callable, Tuple, Generator, List, Union, Final
from tqdm import tqdm
from typing import Set, Iterable

from owlapy.iri import IRI
from owlapy.owl_axiom import OWLEquivalentClassesAxiom
from owlapy.owl_hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from owlapy.utils import OWLClassExpressionLengthMetric, LRUCache
from owlapy.class_expression import (
    OWLClass, OWLClassExpression, OWLObjectUnionOf, OWLObjectIntersectionOf,
    OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLObjectExactCardinality, OWLObjectSomeValuesFrom,
    OWLQuantifiedObjectRestriction, OWLObjectCardinalityRestriction,
    OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLDataAllValuesFrom, OWLDataSomeValuesFrom, OWLDataHasValue, OWLObjectHasValue)

def f1_set_similarity(y: Set[str], yhat: Set[str]) -> float:
    """
    Compute F1 score for two set
    :param y: A set of URIs
    :param yhat: A set of URIs
    :return:
    """
    if len(yhat) == len(y) == 0:
        return 1.0
    if len(yhat) == 0 or len(y) == 0:
        return 0.0

    tp = len(y.intersection(yhat))
    fp = len(yhat.difference(y))
    fn = len(y.difference(yhat))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)


    if precision == 0 and recall == 0:
        return 0.0
    
    return (2 * precision * recall) / (precision + recall)


def concept_reducer(concepts, opt):
    """
    Reduces a set of concepts by applying a binary operation to each pair of concepts.

    Args:
        concepts (set): A set of concepts to be reduced.
        opt (function): A binary function that takes a pair of concepts and returns a single concept.

    Returns:
        set: A set containing the results of applying the binary operation to each pair of concepts.

    Example:
        >>> concepts = {1, 2, 3}
        >>> opt = lambda x: x[0] + x[1]
        >>> concept_reducer(concepts, opt)
        {2, 3, 4, 5, 6}

    Note:
        The operation `opt` should be commutative and associative to ensure meaningful reduction in the context of set operations.
    """
    result = set()
    for i in concepts:
        for j in concepts:
            result.add(opt((i, j)))
    return result


def concept_reducer_properties(
        concepts: Set, properties, cls: Callable = None, cardinality: int = 2
) -> Set[Union[OWLQuantifiedObjectRestriction, OWLObjectCardinalityRestriction]]:
    """
    Map a set of owl concepts and a set of properties into OWL Restrictions

    Args:
        concepts:
        properties:
        cls (Callable): An owl Restriction class
        cardinality: A positive Integer

    Returns: List of OWL Restrictions

    """
    assert isinstance(concepts, Iterable), "Concepts must be an Iterable"
    assert isinstance(properties, Iterable), "properties must be an Iterable"
    assert isinstance(cls, Callable), "cls must be an Callable"
    assert cardinality > 0
    result = set()
    for i in concepts:
        for j in properties:
            if cls == OWLObjectMinCardinality or cls == OWLObjectMaxCardinality:
                result.add(cls(cardinality=cardinality, property=j, filler=i))
                continue
            result.add(cls(j, i))
    return result


def make_iterable_verbose(iterable_object, verbose, desc="Default", position=None, leave=True) -> Iterable:
    if verbose > 0:
        return tqdm(iterable_object, desc=desc, position=position, leave=leave)
    else:
        return iterable_object


def init_length_metric(length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                       length_metric_factory: Optional[Callable[[], OWLClassExpressionLengthMetric]] = None):
    """ Initialize the technique on computing length of a concept"""
    if length_metric is not None:
        pass
    elif length_metric_factory is not None:
        length_metric = length_metric_factory()
    else:
        length_metric = OWLClassExpressionLengthMetric.get_default()

    return length_metric


def concept_len(ce: OWLClassExpression, length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                length_metric_factory: Optional[Callable[[], OWLClassExpressionLengthMetric]] = None):
    length_metric = init_length_metric(length_metric, length_metric_factory)
    return length_metric.length(ce)

def init_hierarchy_instances(reasoner, class_hierarchy, object_property_hierarchy, data_property_hierarchy) -> Tuple[
    ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy]:
    """ Initialize class, object property, and data property hierarchies """
    if class_hierarchy is None:
        class_hierarchy = ClassHierarchy(reasoner)

    if object_property_hierarchy is None:
        object_property_hierarchy = ObjectPropertyHierarchy(reasoner)

    if data_property_hierarchy is None:
        data_property_hierarchy = DatatypePropertyHierarchy(reasoner)

    assert class_hierarchy is None or isinstance(class_hierarchy, ClassHierarchy)
    assert object_property_hierarchy is None or isinstance(object_property_hierarchy, ObjectPropertyHierarchy)
    assert data_property_hierarchy is None or isinstance(data_property_hierarchy, DatatypePropertyHierarchy)

    return class_hierarchy, object_property_hierarchy, data_property_hierarchy


def init_named_individuals(individuals_cache_size, ):
    use_individuals_cache = individuals_cache_size > 0
    if use_individuals_cache:
        _ind_cache = LRUCache(maxsize=individuals_cache_size)
    else:
        _ind_cache = None
    return use_individuals_cache, _ind_cache


def init_individuals_from_concepts(include_implicit_individuals: bool = None, reasoner=None, ontology=None,
                                   individuals_per_concept=None):  # pragma: no cover
    assert isinstance(include_implicit_individuals, bool), f"{include_implicit_individuals} must be boolean"
    if include_implicit_individuals:
        assert isinstance(individuals_per_concept, Generator)
        # get all individuals from concepts
        _ind_set = frozenset(chain.from_iterable(individuals_per_concept))
    else:
        # @TODO: needs to be explained
        individuals = ontology.individuals_in_signature()
        _ind_set = frozenset(individuals)
    return _ind_set


def compute_tp_fn_fp_tn(individuals, pos, neg):  # pragma: no cover
    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))
    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))
    return tp, fn, fp, tn


def compute_f1_score(individuals, pos, neg) -> float:
    """ Compute F1-score of a concept
    """
    assert type(individuals) == type(pos) == type(neg), f"Types must match:{type(individuals)},{type(pos)},{type(neg)}"
    # true positive: |E^+ AND R(C)  |
    tp = len(pos.intersection(individuals))
    # true negative : |E^- AND R(C)|
    tn = len(neg.difference(individuals))

    # false positive : |E^- AND R(C)|
    fp = len(neg.intersection(individuals))
    # false negative : |E^- \ R(C)|
    fn = len(pos.difference(individuals))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0

    if precision == 0 or recall == 0:
        return 0.0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1


def compute_f1_score_from_confusion_matrix(confusion_matrix:dict)->float:
    tp=int(confusion_matrix["tp"])
    fn=int(confusion_matrix["fn"])
    fp=int(confusion_matrix["fp"])
    tn=int(confusion_matrix["tn"])
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0

    if precision == 0 or recall == 0:
        return 0.0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1


def plot_umap_reduced_embeddings(X: pandas.DataFrame, y: List[float], name: str = "umap_visualization.pdf") -> None:  # pragma: no cover
    # TODO:AB: 'umap' is not part of the dependencies !?
    import umap
    reducer = umap.UMAP(random_state=1)
    embedding = reducer.fit_transform(X)
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=["r" if x == 1 else "b" for x in y])
    plt.grid()
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(name)
    plt.show()


def plot_decision_tree_of_expressions(feature_names, cart_tree) -> None:  # pragma: no cover
    """
    Plot the built CART Decision Tree and feature importance.

    Parameters:
        feature_names (list): A list of feature names used in the decision tree.
        cart_tree: The trained CART Decision Tree model.

    Returns:
        None

    Notes:
        This function plots the decision tree using matplotlib and saves it to a PDF file named 'cart_decision_tree.pdf'.
        It also displays the plot.
    """
    plt.figure(figsize=(10, 10))
    sklearn.tree.plot_tree(cart_tree, fontsize=10, feature_names=feature_names, class_names=["Negative", "Positive"],
                           filled=True)
    plt.savefig('cart_decision_tree.pdf')
    plt.show()

def plot_topk_feature_importance(feature_names, cart_tree, topk: int = 10)->None:  # pragma: no cover
    """
    Plot the feature importance of the CART Decision Tree.

    Parameters:
        feature_names (list): A list of feature names used in the decision tree.
        cart_tree: The trained CART Decision Tree model.
        topk (int, optional): The number of top features to display. Defaults to 10.

    Returns:
        None

    Notes:
        This function plots a bar chart showing the importance of each feature in the decision tree,
        with the top k features displayed. The importance is measured by the normalized total reduction.
        The plot is displayed using matplotlib.
    """
    fig, ax = plt.subplots()
    topk_id = np.argsort(cart_tree.feature_importances_)[-topk:]
    expressions = [feature_names[i] for i in topk_id.tolist()]
    feature_importance = cart_tree.feature_importances_[topk_id]
    ax.bar(x=expressions, height=feature_importance)
    ax.set_ylabel('Normalized total reduction')
    ax.set_title('Feature Importance')
    plt.xticks(rotation=90, ha='right')
    fig.tight_layout()
    plt.show()


def save_owl_class_expressions(expressions: Union[OWLClassExpression, List[OWLClassExpression]],
                               path: str = './Predictions',
                               rdf_format: str = 'rdfxml') -> None:  # pragma: no cover
    assert isinstance(expressions, OWLClassExpression) or isinstance(expressions[0],
                                                                     OWLClassExpression), "expressions must be either OWLClassExpression or a list of OWLClassExpression"
    if isinstance(expressions, OWLClassExpression):
        expressions = [expressions]
    NS: Final = 'https://dice-research.org/predictions#'

    if rdf_format != 'rdfxml':
        raise NotImplementedError(f'Format {rdf_format} not implemented.')
    # @TODO: CD: Lazy import. CD: Can we use rdflib to serialize concepts ?!
    from owlapy.owl_ontology import Ontology
    # ()
    ontology = Ontology(IRI.create(NS), load=False)
    # () Iterate over concepts
    for th, i in enumerate(expressions):
        cls_a = OWLClass(IRI.create(NS, str(th)))
        equivalent_classes_axiom = OWLEquivalentClassesAxiom([cls_a, i])
        try:
            ontology.add_axiom(equivalent_classes_axiom)
        except AttributeError:
            print(traceback.format_exc())
            print("Exception at creating OWLEquivalentClassesAxiom")
            print(equivalent_classes_axiom)
            print(cls_a)
            print(i)
            print(expressions)
            exit(1)
    ontology.save(IRI.create(path + '.owl', is_file_path=True))


def verbalize(predictions_file_path: str):  # pragma: no cover
    import xml.etree.ElementTree as ET
    import os
    tree = ET.parse(predictions_file_path)
    root = tree.getroot()
    tmp_file = 'tmp_file_' + predictions_file_path
    owl = 'http://www.w3.org/2002/07/owl#'
    ontology_elem = root.find(f'{{{owl}}}Ontology')
    ontology_elem.remove(ontology_elem.find(f'{{{owl}}}imports'))

    # The commented lines below are needed if you want to use `verbaliser.verbalise_class_expression`
    # They assign labels to classes and properties.

    # rdf = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
    # rdfs = 'http://www.w3.org/2000/01/rdf-schema#'
    # for element in root.iter():
    #     resource = None
    #     if f'{{{rdf}}}about' in element.attrib:
    #         resource = element.attrib[f'{{{rdf}}}about']
    #     elif f'{{{rdf}}}resource' in element.attrib:
    #         resource = element.attrib[f'{{{rdf}}}resource']
    #     if resource is not None:
    #         label = resource.split('#')
    #         if len(label) > 1:
    #             element.set(f'{{{rdfs}}}label', label[1])
    #         else:
    #             element.set(f'{{{rdfs}}}label', resource)

    tree.write(tmp_file)

    try:
        from deeponto.onto import Ontology, OntologyVerbaliser
        from anytree.dotexport import RenderTreeGraph
        from IPython.display import Image
    except Exception as e:
        print("You need to install deeponto to use this feature (pip install deeponto). If you have already, check "
              "whether it's installed properly. \n   ----> Error: " + f'{e}')
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return

    onto = Ontology(tmp_file)
    verbalizer = OntologyVerbaliser(onto)
    complex_concepts = onto.get_asserted_complex_classes()
    try:
        for i, ce in enumerate(complex_concepts):
            tree = verbalizer.parser.parse(str(ce))
            tree.render_image()
            os.rename("range_node.png", f"Prediction_{i}.png")
    except Exception as e:
        print("If you have not installed graphviz, please do so at https://graphviz.org/download/ to make the "
              "verbalization possible. Otherwise check the error message: \n" + f'{e}')
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    if len(complex_concepts) == 0:
        print("No complex classes found!")
    elif len(complex_concepts) == 1:
        print("Image generated successfully!")
    else:
        print("Images generated successfully!")

def get_file_base_name(file_path: str) -> str:
    """
    Extract the base name of a file from a given file path, excluding its extension.

    Parameters:
        file_path (str): The full path to the file.

    Returns:
        str: The base name of the file without its extension.

    Example:
        >>> get_file_base_name("/path/to/file.txt")
        'file'
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def prepare_output_path(base_name: str, output_dir: str = "results", extension: str = ".csv") -> str:
    """
    Create the output directory if it doesn't exist and return the full path for an output file.

    Parameters:
        base_name (str): The base name of the output file (without extension).
        output_dir (str, optional): Directory where the output file will be saved. Defaults to "results".
        extension (str, optional): File extension to use. Defaults to ".csv".

    Returns:
        str: The full path to the output file.

    Example:
        >>> prepare_output_path("summary")
        'results/summary.csv'
    """
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base_name}{extension}")

def assert_class_expression_type(class_expr: Any) -> bool:
    """
    Check whether the given object is a valid OWL class expression.

    This function verifies if the input object is an instance of any recognized OWL class
    expression types (e.g., union, intersection, cardinality restrictions, value restrictions, etc.).

    Parameters:
        class_expr (Any): The object to check.

    Returns:
        bool: True if the object is a valid OWL class expression; False otherwise.

    Example:
        >>> assert_class_expression_type(some_owl_expression)
        True
    """
    expression_types = (
        OWLClassExpression, OWLObjectUnionOf, OWLObjectIntersectionOf,
        OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLObjectExactCardinality,
        OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectComplementOf,
        OWLDataAllValuesFrom, OWLDataSomeValuesFrom, OWLDataHasValue, OWLObjectHasValue,
    )
    return isinstance(class_expr, expression_types)
