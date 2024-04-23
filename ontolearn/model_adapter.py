"""Model adapters."""
import inspect
import json
import logging
import re
from typing import TypeVar, List, Optional, Union

from owlapy.class_expression import OWLClassExpression
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLAxiom
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_reasoner import OWLReasoner

from ontolearn.abstracts import AbstractHeuristic, AbstractScorer, BaseRefinement, AbstractKnowledgeBase, \
    AbstractNode
from ontolearn.base_concept_learner import BaseConceptLearner
from ontolearn.base import OWLReasoner_Owlready2_ComplexCEInstances
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner, NCES
from ontolearn.ea_algorithms import EASimple
from ontolearn.ea_initialization import EARandomWalkInitialization, EARandomInitialization, RandomInitMethod
from ontolearn.fitness_functions import LinearPressureFitness
from ontolearn.heuristics import CELOEHeuristic, OCELHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.metrics import Accuracy, F1, Recall, Precision, WeightedAccuracy
from ontolearn.triple_store import TripleStoreKnowledgeBase
from ontolearn.value_splitter import BinningValueSplitter, EntropyValueSplitter

logger = logging.getLogger(__name__)

metrics = {'f1': F1,
           'accuracy': Accuracy,
           'recall': Recall,
           'precision': Precision,
           'weighted_accuracy': WeightedAccuracy
           }

models = {'celoe': CELOE,
          'ocel': OCEL,
          'evolearner': EvoLearner,
          'nces': NCES}

heuristics = {'celoe': CELOEHeuristic,
              'ocel': OCELHeuristic}

def transform_string(input_string):
    """Used to turn camelCase arguments to snake_case"""
    # Use regex to find all capital letters C and replace them with '_C'
    transformed_string = re.sub(r'([A-Z])', r'_\1', input_string).lower()

    # Remove the leading underscore if it exists
    transformed_string = transformed_string.lstrip('_')

    return transformed_string


def compute_quality(KB, solution, pos, neg, qulaity_func="f1"):
    func = metrics[qulaity_func]().score2
    instances = set(KB.individuals(solution))
    if isinstance(list(pos)[0], str):
        instances = {ind.str.split("/")[-1] for ind in instances}
    tp = len(pos.intersection(instances))
    fn = len(pos.difference(instances))
    fp = len(neg.intersection(instances))
    tn = len(neg.difference(instances))
    return func(tp=tp, fn=fn, fp=fp, tn=tn)[-1]

def _get_matching_opts(_Type, optargs, kwargs, *, prefix=None):
    """Find the keys in kwargs that are parameters of _Type.

    If prefix is specified, the keys in kwargs need to be prefixed with prefix_.
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
        elif transform_string(p(opt)) in kwargs:
            opts[opt] = kwargs.pop(transform_string(p(opt)))
        elif opt in optargs:
            opts[opt] = optargs[opt]
    return opts


_N = TypeVar('_N', bound=AbstractNode)  #:


def ModelAdapter(*args, **kwargs):  # noqa: C901
    """Instantiate a model through the model adapter.

    .. warning ::
        You should not specify both: the _type and the object. For
        example, you should not give both 'reasoner' and 'reasoner_type' because the ModelAdapter cant decide
        which one to use, the reasoner object or create a new reasoner instance using 'reasoner_type'.

    Note:
        If you give `_type` for an argument you can pass further arguments to construct the instance of that
        class. The model adapter will arrange every argument automatically and use them to construct an object
        for that certain class type.

    Args:
        knowledge_base (AbstractKnowledgeBase): A knowledge base.
        knowledge_base_type: A knowledge base type.
        ...: Knowledge base arguments.
        reasoner: A reasoner.
        reasoner_type: A reasoner type.
        ...: Reasoner constructor arguments.
        refinement_operator_type: A refinement operator type.
        ...: Refinement operator arguments.
        quality_type: An Abstract Scorer type.
        ...: Quality arguments.
        heuristic_func (AbstractHeuristic): A heuristic.
        heuristic_type: An Abstract Heuristic type.
        ...: arguments For the heuristic type.
        learner_type: A Base Concept Learner type.
        ...: Arguments for the learning algorithm.
    """
    if "knowledge_base" in kwargs:
        kb = kwargs.pop("knowledge_base")
        if "reasoner" in kwargs:
            kwargs["cl_reasoner"] = kwargs["reasoner"]
            kwargs.pop("reasoner")
        if "knowledge_base_type" in kwargs:
            raise ValueError("both knowledge_base and _type specified")
    else:
        kb_type = kwargs.pop("knowledge_base_type", None)
        if kb_type is None:
            kb_type = KnowledgeBase
        else:
            kb_type = kb_type
        if "reasoner" in kwargs:
            kwargs["cl_reasoner"] = kwargs["reasoner"]
        kb_args = _get_matching_opts(kb_type, {}, kwargs)
        try:
            kb = kb_type(**kb_args)
        except TypeError:
            kb = None
    if kb is not None:
        assert isinstance(kb, AbstractKnowledgeBase)

    if "ignore" in kwargs:
        assert isinstance(kb, KnowledgeBase)
        target_kb = kb.ignore_and_copy(ignored_classes=kwargs.pop("ignore"))
    else:
        target_kb = kb

    if "cl_reasoner" in kwargs:
        reasoner = kwargs.pop("cl_reasoner")
        if "reasoner_type" in kwargs:
            raise ValueError("both reasoner and _type specified")
    else:
        reasoner_type = kwargs.pop("reasoner_type", None)
        if reasoner_type is None:
            reasoner_type = OWLReasoner_Owlready2_ComplexCEInstances
        assert issubclass(reasoner_type, OWLReasoner)
        reasoner = reasoner_type(**_get_matching_opts(
            reasoner_type, {'ontology': target_kb.ontology}, kwargs))
    assert isinstance(reasoner, OWLReasoner)

    if "refinement_operator" in kwargs:
        operator = kwargs.pop("refinement_operator")
        if "refinement_operator_type" in kwargs:
            raise ValueError("both refinement_operator and _type specified")
    else:
        op_type = kwargs.pop("refinement_operator_type", None)
        if op_type is None:
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
            'reasoner': reasoner,
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
                'reasoner': reasoner,
                'refinement_operator': operator,
                'quality_func': qual,
                'heuristic_func': heur,
            }, learner_args
        ))

    return learner


class Trainer:
    def __init__(self, learner: BaseConceptLearner, reasoner: OWLReasoner):
        """
        A class to disentangle the learner from its training.

        Args:
            learner: The concept learner.
            reasoner: The reasoner to use (should have the same ontology as the `kb` argument of the learner).
        """
        assert reasoner.get_root_ontology().get_ontology_id().get_ontology_iri().as_str() == \
               learner.kb.ontology.get_ontology_id().get_ontology_iri().as_str(), "New reasoner does not have " + \
                                                                                    "the same ontology as the learner!"
        learner.reasoner = reasoner
        self.learner = learner
        self.reasoner = reasoner

    def fit(self, *args, **kwargs):
        """Run the concept learning algorithm according to its configuration.

        Once finished, the results can be queried with the `best_hypotheses` function."""
        self.learner.fit(*args, **kwargs)

    def best_hypotheses(self, n):
        """Get the current best found hypotheses according to the quality.

        Args:
            n: Maximum number of results.

        Returns:
            Iterable with hypotheses in form of search tree nodes.
        """
        return self.learner.best_hypotheses(n)

    def predict(self, individuals: List[OWLNamedIndividual],
                hypotheses: Optional[List[Union[_N, OWLClassExpression]]] = None,
                axioms: Optional[List[OWLAxiom]] = None, n: int = 10):
        """Creates a binary data frame showing for each individual whether it is entailed in the given hypotheses
                (class expressions). The individuals do not have to be in the ontology/knowledge base yet. In that case,
                axioms describing these individuals must be provided.

        The state of the knowledge base/ontology is not changed, any provided axioms will be removed again.

        Args:
            individuals: A list of individuals/instances.
            hypotheses: (Optional) A list of search tree nodes or class expressions. If not provided, the
                        current :func:`BaseConceptLearner.best_hypothesis` of the concept learner are used.
            axioms: (Optional) A list of axioms that are not in the current knowledge base/ontology.
                    If the individual list contains individuals that are not in the ontology yet, axioms
                    describing these individuals must be provided. The argument can also be used to add
                    arbitrary axioms to the ontology for the prediction.
            n: Integer denoting number of ALC concepts to extract from search tree if hypotheses=None.

        Returns:
            Pandas data frame with dimensions |individuals|*|hypotheses| indicating for each individual and each
            hypothesis whether the individual is entailed in the hypothesis.
        """
        return self.learner.predict(individuals, hypotheses, axioms, n)

    def save_best_hypothesis(self, n: int = 10, path: str = 'Predictions', rdf_format: str = 'rdfxml') -> None:
        """Serialise the best hypotheses to a file.

        Args:
            n: Maximum number of hypotheses to save.
            path: Filename base (extension will be added automatically).
            rdf_format: Serialisation format. currently supported: "rdfxml".
        """
        self.learner.save_best_hypothesis(n, path, rdf_format)


def execute(args):

    args_d = args.__dict__
    learner_type = models[args.model]
    optargs = {}
    if args.sparql_endpoint:
        kb = TripleStoreKnowledgeBase(args.sparql_endpoint)
    else:
        kb = KnowledgeBase(path=args.knowledge_base_path)

    with open(args.path_learning_problem) as json_file:
        examples = json.load(json_file)
    pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
    neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
    lp = PosNegLPStandard(pos=pos, neg=neg)

    if args.model in ["celoe", "ocel"]:
        heur_func = heuristics[args.model](**_get_matching_opts(heuristics[args.model], {}, args_d))
        refinement_op = ModifiedCELOERefinement(**_get_matching_opts(ModifiedCELOERefinement,
                                                {"knowledge_base": kb,
                                                 "value_splitter": BinningValueSplitter(args.max_nr_splits)},
                                                args_d))
        optargs = {"knowledge_base": kb,
                   "quality_func": metrics[args.quality_metric](),
                   "heuristic_func": heur_func,
                   "refinement_operator": refinement_op}
    elif args.model == "evolearner":
        fit_func = LinearPressureFitness(**_get_matching_opts(LinearPressureFitness, {}, args_d))
        init_rw_method = EARandomWalkInitialization(**_get_matching_opts(EARandomWalkInitialization, {}, args_d))
        algorithm = EASimple(**_get_matching_opts(EASimple, {}, args_d))
        mut_uniform_gen = EARandomInitialization(**_get_matching_opts(
            EARandomInitialization, {"method": getattr(RandomInitMethod, args.init_method_type)}, args_d))
        value_splitter = EntropyValueSplitter(**_get_matching_opts(EntropyValueSplitter, {}, args_d))

        optargs = {"knowledge_base": kb,
                   "quality_func": metrics[args.quality_metric](),
                   "fitness_func": fit_func,
                   "init_method": init_rw_method,
                   "algorithm": algorithm,
                   "mut_uniform_gen": mut_uniform_gen,
                   "value_splitter": value_splitter}
    # elif args.model == "drill":
    #     optargs = {"knowledge_base": kb,
    #                "quality_func": metrics[args.quality_metric]()}

    model = learner_type(**_get_matching_opts(learner_type, optargs, args_d))

    if args.model in ["celoe", "evolearner", "ocel"]:
        trainer = Trainer(model, kb.reasoner())
        trainer.fit(lp)
        print(trainer.best_hypotheses(1))
        if args.save:
            trainer.save_best_hypothesis()

    elif args.model in ["nces"]:
        hypothesis = model.fit(pos, neg)  # This will also print the prediction
        # @TODO:CD: model.fit() should return a train model itself, not predictions
        report = f"Quality: {compute_quality(kb, hypothesis, pos, neg, args.quality_metric)} \nIndividuals: " + \
                 f"{kb.individuals_count(hypothesis)}"
        print(report)
