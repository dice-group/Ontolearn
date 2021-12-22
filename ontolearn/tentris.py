import logging
from functools import singledispatchmethod
from types import MappingProxyType
from typing import Optional, Iterable

import httpx as httpx

from ontolearn import KnowledgeBase
from ontolearn.abstracts import AbstractScorer, AbstractLearningProblem, AbstractKnowledgeBase, \
    EncodedPosNegLPStandardKind
from ontolearn.concept_generator import ConceptGenerator
from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric
from ontolearn.knowledge_base import EvaluatedConcept, Factory, _Default_ClassExpressionLengthMetricFactory
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Precision, Accuracy, Recall
from ontolearn.utils import oplogging
from owlapy.ext import OWLReasonerEx
from owlapy.model import OWLClassExpression, OWLEntity, OWLOntology, OWLClass, OWLNamedIndividual, \
    OWLObjectPropertyExpression, OWLDataProperty, OWLObjectProperty, OWLOntologyID, _M, OWLDataPropertyRangeAxiom, \
    IRI, OWLThing, OWLLiteral, OWLObjectPropertyRangeAxiom, OWLObjectPropertyDomainAxiom, OWLDataPropertyDomainAxiom
from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from owlapy.render import ManchesterOWLSyntaxOWLObjectRenderer, DLSyntaxObjectRenderer
from owlapy.util import LRUCache

logger = logging.getLogger(__name__)

_Metric_map = MappingProxyType({
    F1: 'f1_score',
    Precision: 'precision',
    Recall: 'recall',
    Accuracy: 'accuracy',
    # WeightedAccuracy: ,
    # : specificity,
    # : sensitivity,
})

_debug_render = DLSyntaxObjectRenderer().render


def _full_iri_renderer(e: OWLEntity) -> str:
    return f'<{e.to_string_id()}>'


_tentris_render = ManchesterOWLSyntaxOWLObjectRenderer(_full_iri_renderer, no_render_thing=True).render


class EncodedPosNegLPStandardTentris(EncodedPosNegLPStandardKind):
    __slots__ = 'id'

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f'EncodedPosNegLPStandardTentris(id={self.id})'


class TentrisOntology(OWLOntology):
    __slots__ = '_path', '_endpoint_url', '_backing_mgr', '_backing_onto', '_endpoint_timeout'

    def __init__(self, path: str, endpoint_url: str, timeout: float):
        self._path = path
        self._endpoint_url = endpoint_url
        self._endpoint_timeout = timeout
        self._backing_mgr = OWLOntologyManager_Owlready2()
        self._backing_onto = self._backing_mgr.load_ontology(IRI.create('file://' + self._path))

    def classes_in_signature(self) -> Iterable[OWLClass]:
        logger.debug("Calling classes_in_signature from backing onto")
        yield from self._backing_onto.classes_in_signature()

    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        logger.debug("Calling data_properties_in_signature from backing onto")
        yield from self._backing_onto.data_properties_in_signature()

    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        logger.debug("Calling object_properties_in_signature from backing onto")
        yield from self._backing_onto.object_properties_in_signature()

    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        res = httpx.get(self._endpoint_url + '/instances',
                        params={'class_expression': _tentris_render(OWLThing)})
        for i in res.json()['instances']:
            yield OWLNamedIndividual(IRI.create(i))

    def data_property_range_axioms(self, property: OWLDataProperty) -> Iterable[OWLDataPropertyRangeAxiom]:
        raise NotImplementedError

    def data_property_domain_axioms(self, property: OWLDataProperty) -> Iterable[OWLDataPropertyDomainAxiom]:
        logger.debug("Calling data_property_domain_axioms from backing onto")
        yield from self._backing_onto.data_property_domain_axioms(property)

    def object_property_domain_axioms(self, property: OWLObjectProperty) -> Iterable[OWLObjectPropertyDomainAxiom]:
        logger.debug("Calling object_property_domain_axioms from backing onto")
        yield from self._backing_onto.object_property_domain_axioms(property)

    def object_property_range_axioms(self, property: OWLObjectProperty) -> Iterable[OWLObjectPropertyRangeAxiom]:
        logger.debug("Calling object_property_range_axioms from backing onto")
        yield from self._backing_onto.object_property_range_axioms(property)

    def get_owl_ontology_manager(self) -> _M:
        raise NotImplementedError

    def get_ontology_id(self) -> OWLOntologyID:
        raise NotImplementedError

    def __eq__(self, other):
        if type(other) == type(self):
            return self._endpoint_url == other._endpoint_url and self._path == other._path
        return NotImplemented

    def __hash__(self):
        return hash((self._endpoint_url, self._path))

    def __repr__(self):
        return f'TentrisOntology(path={repr(self._path)},endpoint_url={repr(self._endpoint_url)})'


class TentrisReasoner(OWLReasonerEx):
    __slots__ = '_ontology', '_backing_reasoner'

    def __init__(self, ontology: TentrisOntology):
        self._ontology = ontology
        self._backing_reasoner = OWLReasoner_Owlready2(self._ontology._backing_onto)

    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClass]:
        raise NotImplementedError

    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        logger.debug("Calling object_property_domains(%s) from backing reasoner", _debug_render(pe))
        yield from self._backing_reasoner.object_property_domains(pe, direct=direct)

    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClass]:
        logger.debug("Calling object_property_ranges(%s) from backing reasoner", _debug_render(pe))
        yield from self._backing_reasoner.object_property_ranges(pe, direct=direct)

    def equivalent_classes(self, ce: OWLClassExpression) -> Iterable[OWLClass]:
        raise NotImplementedError

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty) -> Iterable[OWLLiteral]:
        raise NotImplementedError

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression) \
            -> Iterable[OWLNamedIndividual]:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def instances(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLNamedIndividual]:
        logger.warning("Instances(%s) method used", _debug_render(ce))
        res = httpx.get(self._ontology._endpoint_url + '/instances',
                        params={'class_expression': _tentris_render(ce)},
                        timeout=self._ontology._endpoint_timeout)
        for i in res.json()['instances']:
            yield OWLNamedIndividual(IRI.create(i))

    def sub_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        logger.debug("Calling sub_classes(%s) from backing reasoner", _debug_render(ce))
        yield from self._backing_reasoner.sub_classes(ce, direct=direct)

    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        logger.debug("Calling sub_data_properties(%s) from backing reasoner", _debug_render(dp))
        yield from self._backing_reasoner.sub_data_properties(dp, direct=direct)

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        logger.debug("Calling sub_object_properties(%s) from backing reasoner", _debug_render(op))
        yield from self._backing_reasoner.sub_object_properties(op, direct=direct)

    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        raise NotImplementedError

    def get_root_ontology(self) -> TentrisOntology:
        return self._ontology

    def super_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        raise NotImplementedError


class EvaluatedConceptTentris(EvaluatedConcept):
    __slots__ = ()

    @property
    def inds(self):
        if logger.isEnabledFor(oplogging.TRACE):
            logger.debug("No individuals")
        return None


class TentrisKnowledgeBase(KnowledgeBase):
    __slots__ = 'endpoint_url', 'endpoint_timeout', 'async_client', 'tasks', '_total_req', '_current_req'

    _ontology: TentrisOntology
    endpoint_url: str
    endpoint_timeout: float

    def __init__(self, path: str, *,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 individuals_cache_size=128,
                 ):
        AbstractKnowledgeBase.__init__(self)
        self.path = path
        self.endpoint_url = 'http://localhost:8131'
        self.endpoint_timeout = 15.0
        self.tasks = 8
        self.async_client = httpx.AsyncClient(timeout=httpx.Timeout(self.endpoint_timeout,
                                                                    pool=self.endpoint_timeout * (1 + self.tasks)),
                                              limits=httpx.Limits(max_connections=self.tasks,
                                                                  max_keepalive_connections=self.tasks))
        self._total_req = 0
        self._current_req = 0

        self._ontology = TentrisOntology(self.path, self.endpoint_url, timeout=50.0)
        self._reasoner = TentrisReasoner(self._ontology)

        if length_metric is not None:
            self._length_metric = length_metric
        elif length_metric_factory is not None:
            self._length_metric = length_metric_factory()
        else:
            self._length_metric = _Default_ClassExpressionLengthMetricFactory()

        ConceptGenerator.__init__(self, reasoner=self._reasoner)

        individuals = self._ontology.individuals_in_signature()
        self._ind_set = frozenset(individuals)

        self.use_individuals_cache = individuals_cache_size > 0
        if self.use_individuals_cache:
            self._ind_cache = LRUCache(maxsize=individuals_cache_size)

        logger.info(f'Tentris Knowledge Base created. Endpoint: {self.endpoint_url}')

    @singledispatchmethod
    def encode_learning_problem(self, lp: AbstractLearningProblem):
        raise NotImplementedError(lp)

    @encode_learning_problem.register
    def _(self, lp: PosNegLPStandard):
        id_ = httpx.post(self.endpoint_url + '/learning_problem', json={
            'positives': list(map(lambda _: _.to_string_id(), sorted(lp.pos))),
            'negatives': list(map(lambda _: _.to_string_id(), sorted(lp.neg))),
        }).text
        logger.debug("LP id: %s", id_)
        return EncodedPosNegLPStandardTentris(id_)

    def evaluate_concept(self, concept: OWLClassExpression, quality_func: AbstractScorer,
                         encoded_learning_problem: EncodedPosNegLPStandardTentris) -> EvaluatedConcept:
        e = EvaluatedConceptTentris()
        ce = _tentris_render(concept.get_nnf())
        metric = _Metric_map.get(type(quality_func))
        # workaround tentris bug
        if metric == 'f1_score':
            metric_kv = dict()
        else:
            metric_kv = {'metric': metric}
        try:
            res = httpx.get(self.endpoint_url + '/class_expression_quality',
                            params={
                                **metric_kv,
                                'class_expression': ce,
                                'learning_problem_id': str(encoded_learning_problem.id)
                            },
                            timeout=self.endpoint_timeout)
        except httpx.ReadTimeout:
            logger.error("Could not resolve << %s >> using Tentris@%s", ce, encoded_learning_problem.id)
            e.q = 0
            return e
        e.q = float(res.text)
        return e

    async def evaluate_concept_async(self, concept: OWLClassExpression, quality_func: AbstractScorer,
                                     encoded_learning_problem: EncodedPosNegLPStandardTentris) -> EvaluatedConcept:
        e = EvaluatedConceptTentris()
        ce = _tentris_render(concept.get_nnf())
        metric = _Metric_map.get(type(quality_func))
        # workaround tentris bug
        id_ = self._total_req + 1
        if metric == 'f1_score':
            metric_kv = dict()
        else:
            metric_kv = {'metric': metric}
        try:
            # async with httpx.AsyncClient() as client:
            self._total_req += 1
            self._current_req += 1
            logger.debug(f"START:{id_} -- total:{self._total_req} current:{self._current_req} -- {ce}")
            res = await self.async_client.get(
                self.endpoint_url + '/class_expression_quality',
                params={
                    **metric_kv,
                    'class_expression': ce,
                    'learning_problem_id': str(encoded_learning_problem.id)
                })
            logger.debug(f"E N D:{id_} -- total:{self._total_req} current:{self._current_req} -- ")
            self._current_req -= 1
        except httpx.ReadTimeout:
            logger.error("Could not resolve << %s >> using Tentris@%s", ce, encoded_learning_problem.id)
            e.q = 0
            return e
        e.q = float(res.text)
        # await res.aclose()
        # logger.debug(f"CLOSE:{id_}")
        return e
