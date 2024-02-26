from itertools import chain
from typing import Optional, Callable, Tuple, Generator
from ..base.owl.hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from ..base.owl.utils import OWLClassExpressionLengthMetric
from owlapy.util import LRUCache
from ..base.fast_instance_checker import OWLReasoner_FastInstanceChecker


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
                                   individuals_per_concept=None):
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


def compute_tp_fn_fp_tn(individuals, pos, neg):
    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))
    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))
    return tp, fn, fp, tn


def compute_f1_score(individuals, pos, neg):
    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))

    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0

    if precision == 0 or recall == 0:
        return 0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1