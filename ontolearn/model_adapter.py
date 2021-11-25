import inspect
import logging
from typing import TypeVar

from ontolearn.abstracts import AbstractHeuristic, AbstractScorer, BaseRefinement, AbstractKnowledgeBase, \
    AbstractNode
from ontolearn.base_concept_learner import BaseConceptLearner

logger = logging.getLogger(__name__)


def _get_matching_opts(_Type, optargs, kwargs, *, prefix=None):
    """find the keys in kwargs that are parameters of _Type

    if prefix is specified, the keys in kwargs need to be prefixed with prefix_
    """
    opts = {}
    if prefix is None:
        def p(s):
            return s
    else:
        def p(s):
            return prefix + "_" + s
    sig = set()
    sig.update(inspect.signature(_Type).parameters.keys())
    sig.difference_update({'args', 'kwds'})
    try:
        sig.update(inspect.signature(_Type.__init__).parameters.keys())
        sig.discard('self')
    except AttributeError:
        pass

    for opt in sig:
        if p(opt) in kwargs:
            opts[opt] = kwargs.pop(p(opt))
        elif opt in optargs:
            opts[opt] = optargs[opt]
    return opts


_N = TypeVar('_N', bound=AbstractNode)  #:


def ModelAdapter(*args, **kwargs):  # noqa: C901
    """Create a new Concept learner through the model adapter

    Args:
        knowledge_base (AbstractKnowledgeBase): a knowledge base
        knowledge_base_type: a knowledge base type
        ...: knowledge base arguments
        refinement_operator_type: a refinement operator type
        ...: refinement operator arguments
        quality_type: an Abstract Scorer type
        ...: quality arguments
        heuristic_func (AbstractHeuristic): a heuristic
        heuristic_type: an Abstract Heuristic type
        ...: arguments for the heuristic type
        learner_type: a Base Concept Learner type
        ...: arguments for the learning algorithm
    """

    if "knowledge_base" in kwargs:
        kb = kwargs.pop("knowledge_base")
        if "knowledge_base_type" in kwargs:
            raise ValueError("both knowledge_base and _type specified")
    else:
        kb_type = kwargs.pop("knowledge_base_type", None)
        if kb_type is None:
            from ontolearn import KnowledgeBase
            kb_type = KnowledgeBase
        else:
            kb_type = kb_type

        kb_args = _get_matching_opts(kb_type, {}, kwargs)
        try:
            kb = kb_type(**kb_args)
        except TypeError:
            kb = None
    if kb is not None:
        assert isinstance(kb, AbstractKnowledgeBase)

    if "ignore" in kwargs:
        from ontolearn import KnowledgeBase
        assert isinstance(kb, KnowledgeBase)
        target_kb = kb.ignore_and_copy(ignored_classes=kwargs.pop("ignore"))
    else:
        target_kb = kb

    if "refinement_operator" in kwargs:
        operator = kwargs.pop("refinement_operator")
        if "refinement_operator_type" in kwargs:
            raise ValueError("both refinement_operator and _type specified")
    else:
        op_type = kwargs.pop("refinement_operator_type", None)
        if op_type is None:
            from ontolearn.refinement_operators import ModifiedCELOERefinement
            op_type = ModifiedCELOERefinement
        assert issubclass(op_type, BaseRefinement)
        operator = op_type(**_get_matching_opts(
            op_type, {
                'knowledge_base': target_kb
            }, kwargs))
    assert isinstance(operator, BaseRefinement)

    if "quality_func" in kwargs:
        qual = kwargs.pop("quality_func")
        if "quality_type" in kwargs:
            raise ValueError("both quality_func and _type specified")
    else:
        quality_type = kwargs.pop("quality_type", None)
        if quality_type is None:
            from ontolearn.metrics import F1
            quality_type = F1
        assert issubclass(quality_type, AbstractScorer)
        qual = quality_type(**_get_matching_opts(quality_type, {}, kwargs))
    assert isinstance(qual, AbstractScorer)

    if "heuristic_func" in kwargs:
        heur = kwargs.pop("heuristic_func")
        if "heuristic_type" in kwargs:
            raise ValueError("both heuristic_func and _type specified")
    else:
        heuristic_type = kwargs.pop("heuristic_type", None)
        if heuristic_type is None:
            from ontolearn.heuristics import CELOEHeuristic
            heuristic_type = CELOEHeuristic
        assert issubclass(heuristic_type, AbstractHeuristic)
        heur = heuristic_type(**_get_matching_opts(heuristic_type, {}, kwargs))
    assert isinstance(heur, AbstractHeuristic)

    if "learner" in kwargs:
        learner = kwargs.pop("learner")
        learner_type = type(learner)
        if "learner_type" in kwargs:
            raise ValueError("both learner and _type specified")
    else:
        learner_type = kwargs.pop("learner_type", None)
        if learner_type is None:
            from ontolearn.concept_learner import CELOE
            learner_type = CELOE
        assert issubclass(learner_type, BaseConceptLearner)
        learner_args = _get_matching_opts(learner_type, {}, kwargs)
        learner = None

    other_components = dict()
    clearkeys = set()
    for k in list(kwargs):
        if k in kwargs and k.endswith("_type"):
            clearkeys.add(k)
            cls = kwargs[k]
            assert issubclass(cls, object)
            other_components[k[:-5]] = (cls, _get_matching_opts(cls, {}, kwargs))

    for k in clearkeys:
        kwargs.pop(k)

    if kwargs:
        logger.warning("Unused parameters: %s", kwargs)

    other_instances = dict()
    for k in other_components:
        cls = other_components[k][0]
        logger.debug("Instantiating %s of type %s", k, cls)

        # noinspection PyArgumentList
        inst = cls(**_get_matching_opts(cls, {
            'knowledge_base': target_kb,
            'refinement_operator': operator,
            'quality_func': qual,
            'heuristic_func': heur,
        }, other_components[k][1]))
        other_instances[k] = inst

    if learner is None:
        learner = learner_type(**_get_matching_opts(
            learner_type, {
                **other_instances,
                'knowledge_base': target_kb,
                'refinement_operator': operator,
                'quality_func': qual,
                'heuristic_func': heur,
            }, learner_args
        ))

    return learner
