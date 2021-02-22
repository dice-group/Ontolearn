import inspect
import logging

from ontolearn.abstracts import AbstractHeuristic, AbstractScorer, BaseRefinement, AbstractKnowledgeBase, \
    AbstractLearningProblem
from ontolearn.base_concept_learner import BaseConceptLearner

logger = logging.getLogger(__name__)


def _get_matching_opts(_Type, optargs, topargs, kwargs, prefix=None):
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
        elif opt in topargs:
            opts[opt] = topargs[opt]
        elif opt in optargs:
            opts[opt] = optargs[opt]
    return opts


class ModelAdapter:
    def __init__(self, *args, **kwargs):
        """Create a new Concept learner model adapter

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
        kb_type = kwargs.pop("knowledge_base_type", None)
        if "knowledge_base" in kwargs:
            self.kb = kwargs.pop("knowledge_base")
        else:
            if kb_type is None:
                from ontolearn import KnowledgeBase
                kb_type = KnowledgeBase
            self.kb = kb_type(**_get_matching_opts(kb_type, {}, {}, kwargs))
        assert isinstance(self.kb, AbstractKnowledgeBase)

        self.op_type = kwargs.pop("refinement_operator_type", None)
        if "refinement_operator" in kwargs:
            raise ValueError("please specify refinement_operator_type")
        if self.op_type is None:
            from ontolearn.refinement_operators import ModifiedCELOERefinement
            self.op_type = ModifiedCELOERefinement
        assert issubclass(self.op_type, BaseRefinement)
        self.op_args = _get_matching_opts(self.op_type, {}, {}, kwargs)

        if "quality_func" in kwargs:
            raise ValueError("please specify quality_type")
        self.quality_type = kwargs.pop("quality_type", None)
        if self.quality_type is None:
            from ontolearn.metrics import F1
            self.quality_type = F1
        assert issubclass(self.quality_type, AbstractScorer)
        self.quality_args = _get_matching_opts(self.quality_type, {}, {}, kwargs)

        if "heuristic_func" in kwargs:
            self.heuristic_func = kwargs.pop("heuristic_func")
            assert isinstance(self.heuristic_func, AbstractHeuristic)
        else:
            self.heuristic_type = kwargs.pop("heuristic_type", None)
            if self.heuristic_type is None:
                from ontolearn.heuristics import CELOEHeuristic
                self.heuristic_type = CELOEHeuristic
            assert issubclass(self.heuristic_type, AbstractHeuristic)
            self.heuristic_args = _get_matching_opts(self.heuristic_type, {}, {}, kwargs)

        if "learner" in kwargs:
            raise ValueError("please specify learner_type")
        self.learner_type = kwargs.pop("learner_type", None)
        if self.learner_type is None:
            from ontolearn.concept_learner import CELOE
            self.learner_type = CELOE
        assert issubclass(self.learner_type, BaseConceptLearner)
        self.learner_args = _get_matching_opts(self.learner_type, {}, {}, kwargs)

        if kwargs:
            logger.warning("Unused parameters: %s", kwargs)

    def fit(self, *args, **kwargs):
        """Execute fit function on a model adapter

        Args:
            ignore (Iterable[OWLClass]): list of OWL Classes to ignore
            learning_problem_type: a type of the learning prblem
            ...: learning problem arguments, for example pos and neg
        """
        if "ignore" in kwargs:
            target_kb = self.kb.ignore_and_copy(ignored_classes=kwargs.pop("ignore"))
        else:
            target_kb = self.kb

        lp_type = kwargs.pop("learning_problem_type", None)
        if lp_type is None:
            from ontolearn.learning_problem import PosNegLPStandard
            lp_type = PosNegLPStandard
        assert issubclass(lp_type, AbstractLearningProblem)
        lp = lp_type(**_get_matching_opts(
            lp_type, {
                'knowledge_base': target_kb
            }, {}, kwargs))

        try:
            heur = self.heuristic_func
        except AttributeError:
            heur = self.heuristic_type(**_get_matching_opts(
                self.heuristic_type, {
                    'learning_problem': lp, 'knowledge_base': target_kb
                }, self.heuristic_args, kwargs))

        qual = self.quality_type(**_get_matching_opts(
            self.quality_type, {
                'learning_problem': lp, 'knowledge_base': target_kb
            }, self.quality_args, kwargs
        ))

        opts = _get_matching_opts(
            self.op_type, {
                'knowledge_base': target_kb
            }, self.op_args, kwargs)
        operator = self.op_type(**opts)

        learner = self.learner_type(**_get_matching_opts(
            self.learner_type, {
                'knowledge_base': target_kb,
                'learning_problem': lp,
                'refinement_operator': operator,
                'quality_func': qual,
                'heuristic_func': heur,
            }, self.learner_args, kwargs
        ))

        if kwargs:
            logger.warning("Unused parameters: %s", kwargs)

        return learner.fit()