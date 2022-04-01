import logging
from functools import singledispatchmethod
from typing import Optional, Iterable, FrozenSet

import httpx as httpx
from httpx import AsyncClient

from ontolearn import KnowledgeBase
from ontolearn.abstracts import AbstractScorer, AbstractLearningProblem, AbstractKnowledgeBase, \
    EncodedPosNegLPStandardKind
from ontolearn.concept_generator import ConceptGenerator
from ontolearn.core.owl.utils import OWLClassExpressionLengthMetric
from ontolearn.knowledge_base import EvaluatedConcept, Factory, _Default_ClassExpressionLengthMetricFactory
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.metrics import Recall
from ontolearn.utils import oplogging
from owlapy.ext import OWLReasonerEx
from owlapy.model import OWLClassExpression, OWLEntity, OWLOntology, OWLClass, OWLNamedIndividual, \
    OWLObjectPropertyExpression, OWLDataProperty, OWLObjectProperty, OWLOntologyID, _M, OWLDataPropertyRangeAxiom, \
    IRI, OWLThing, OWLLiteral, OWLDatatype, OWLDataPropertyDomainAxiom, OWLObjectPropertyDomainAxiom, \
    OWLObjectPropertyRangeAxiom
from owlapy.owl2sparql.converter import Owl2SparqlConverter
from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from owlapy.render import ManchesterOWLSyntaxOWLObjectRenderer, DLSyntaxObjectRenderer
from owlapy.util import LRUCache
from owlapy.vocab import OWLRDFVocabulary

logger = logging.getLogger(__name__)

_debug_render = DLSyntaxObjectRenderer().render


def _full_iri_renderer(e: OWLEntity) -> str:
    return f'<{e.to_string_id()}>'


class EncodedPosNegLPStandardSparql(EncodedPosNegLPStandardKind):
    __slots__ = 'pos', 'neg'

    pos: FrozenSet[OWLNamedIndividual]
    neg: FrozenSet[OWLNamedIndividual]

    def __init__(self, lp: PosNegLPStandard):
        self.pos = lp.pos
        self.neg = lp.neg

    def __repr__(self):
        return f'EncodedPosNegLPStandardSparql()'


class SparqlClient:
    __slots__ = '_timeout'

    def __init__(self, timeout):
        self._timeout = timeout

    def query(self, url, query) -> httpx.Response:
        return httpx.post(
            url,
            headers={
                'Accept': 'application/sparql-results+json'
            },
            data={
                'query': query
            },
            timeout=self._timeout
        )

    def unwrap(self, result: httpx.Response):
        json = result.json()
        vars_ = list(json['head']['vars'])
        for b in json['results']['bindings']:
            val = []
            for v in vars_:
                if b[v]['type'] == 'uri':
                    val.append(IRI.create(b[v]['value']))
                else:
                    val.append(OWLLiteral(b[v]['value'], OWLDatatype(IRI.create(b[v]['datatype']))))
            if len(val) == 1:
                yield val.pop()
            else:
                yield tuple(val)


class AsyncSparqlClient:
    __slots__ = '_client', 'endpoint_timeout', 'tasks'

    def __init__(self, timeout, tasks):
        self.endpoint_timeout = timeout
        self.tasks = tasks
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.endpoint_timeout,
                                                               pool=self.endpoint_timeout * (1 + self.tasks)),
                                         limits=httpx.Limits(max_connections=self.tasks,
                                                             max_keepalive_connections=self.tasks))

    async def query(self, url, query) -> httpx.Response:
        return await self._client.post(
            url,
            headers={
                'Accept': 'application/sparql-results+json'
            },
            data={
                'query': query
            }
        )

    async def aclose(self):
        return await self._client.aclose()


AsyncSparqlClient.unwrap = SparqlClient.unwrap


class SparqlOntology(OWLOntology):
    __slots__ = '_path', '_endpoint_url', '_backing_mgr', '_backing_onto', 'client'

    def __init__(self, path: str, endpoint_url: str, timeout: float):
        self._path = path
        self._endpoint_url = endpoint_url
        self.client = SparqlClient(timeout)
        self._backing_mgr = OWLOntologyManager_Owlready2()
        self._backing_onto = self._backing_mgr.load_ontology(IRI.create('file://' + self._path))

    def classes_in_signature(self) -> Iterable[OWLClass]:
        res = self.client.query(
            self._endpoint_url,
            f"SELECT DISTINCT ?cls WHERE {{"
            f" ?cls a <{OWLRDFVocabulary.OWL_CLASS.as_str()}> . "
            f"}}"
        )
        for cls in self.client.unwrap(res):
            yield OWLClass(cls)

    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        logger.debug("Calling data_properties_in_signature from backing onto")
        yield from self._backing_onto.data_properties_in_signature()

    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        logger.debug("Calling object_properties_in_signature from backing onto")
        yield from self._backing_onto.object_properties_in_signature()

    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        res = self.client.query(
            self._endpoint_url,
            f"SELECT DISTINCT ?ind WHERE {{"
            f" ?ind a <{OWLRDFVocabulary.OWL_NAMED_INDIVIDUAL.as_str()}> . "
            f"}}"
        )
        for ind in self.client.unwrap(res):
            yield OWLNamedIndividual(ind)

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
        return f'SparqlOntology(path={repr(self._path)},endpoint_url={repr(self._endpoint_url)})'


class SparqlReasoner(OWLReasonerEx):
    __slots__ = '_ontology', '_backing_reasoner', '_conv'

    def __init__(self, ontology: SparqlOntology):
        self._ontology = ontology
        self._conv = Owl2SparqlConverter()
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
        query = self._conv.as_query("?r", ce, False)
        logger.warning("Instances(%s) method used => %s", _debug_render(ce), query)
        res = self._ontology.client.query(
            self._ontology._endpoint_url,
            query
        )
        for i in self._ontology.client.unwrap(res):
            yield OWLNamedIndividual(i)

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

    def get_root_ontology(self) -> SparqlOntology:
        return self._ontology

    def super_classes(self, ce: OWLClassExpression, direct: bool = False) -> Iterable[OWLClass]:
        raise NotImplementedError


class EvaluatedConceptSparql(EvaluatedConcept):
    __slots__ = ()

    @property
    def inds(self):
        if logger.isEnabledFor(oplogging.TRACE):
            logger.debug("No individuals")
        return None


class SparqlKnowledgeBase(KnowledgeBase):
    __slots__ = 'endpoint_url', 'endpoint_timeout', 'async_client', 'tasks', '_total_req', '_current_req'

    _ontology: SparqlOntology
    _reasoner: SparqlReasoner
    endpoint_url: str
    endpoint_timeout: float

    def __init__(self, path: str, endpoint_url: str, *,
                 length_metric_factory: Optional[Factory[[], OWLClassExpressionLengthMetric]] = None,
                 length_metric: Optional[OWLClassExpressionLengthMetric] = None,
                 individuals_cache_size=128,
                 ):
        AbstractKnowledgeBase.__init__(self)
        self.path = path
        self.endpoint_url = endpoint_url

        self._total_req = 0
        self._current_req = 0

        self._ontology = SparqlOntology(self.path, self.endpoint_url, timeout=50.0)
        self._reasoner = SparqlReasoner(self._ontology)
        self.async_client = AsyncSparqlClient(timeout=15.0, tasks=8)

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

        logger.info(f'Sparql Knowledge Base created. Endpoint: {self.endpoint_url}')

    @singledispatchmethod
    def encode_learning_problem(self, lp: AbstractLearningProblem):
        raise NotImplementedError(lp)

    @encode_learning_problem.register
    def _(self, lp: PosNegLPStandard):
        return EncodedPosNegLPStandardSparql(lp)

    def evaluate_concept(self, concept: OWLClassExpression, quality_func: AbstractScorer,
                         encoded_learning_problem: EncodedPosNegLPStandardSparql) -> EvaluatedConcept:
        e = EvaluatedConceptSparql()
        try:
            query = self._reasoner._conv.as_query("?r", concept, count=True, values=encoded_learning_problem.pos)
            res = self._ontology.client.query(
                self.endpoint_url,
                query
            )
            tp = next(iter(self._ontology.client.unwrap(res))).parse_integer()
            fn = len(encoded_learning_problem.pos) - tp
            query = self._reasoner._conv.as_query("?r", concept, count=True, values=encoded_learning_problem.neg)
            res = self._ontology.client.query(
                self.endpoint_url,
                query
            )
            fp = next(iter(self._ontology.client.unwrap(res))).parse_integer()
            tn = len(encoded_learning_problem.neg) - fp
            _, e.q = quality_func.score2(tp=tp, fn=fn, fp=fp, tn=tn)
        except httpx.ReadTimeout:
            logger.error("Could not resolve << %s >>", concept)
            e.q = 0
            return e
        return e

    async def evaluate_concept_async(self, concept: OWLClassExpression, quality_func: AbstractScorer,
                                     encoded_learning_problem: EncodedPosNegLPStandardSparql) -> EvaluatedConcept:
        e = EvaluatedConceptSparql()
        try:
            self._total_req += 2
            self._current_req += 2
            import asyncio
            query_pos = self._reasoner._conv.as_query("?r", concept, count=True, values=encoded_learning_problem.pos)
            query_neg = self._reasoner._conv.as_query("?r", concept, count=True, values=encoded_learning_problem.neg)
            r_pos, r_neg = await asyncio.gather(
                self.async_client.query(
                    self.endpoint_url,
                    query_pos
                ),
                self.async_client.query(
                    self.endpoint_url,
                    query_neg
                )
            )
            tp = next(iter(self.async_client.unwrap(r_pos))).parse_integer()
            fn = len(encoded_learning_problem.pos) - tp
            fp = next(iter(self.async_client.unwrap(r_neg))).parse_integer()
            tn = len(encoded_learning_problem.neg) - fp
            _, e.q = quality_func.score2(tp=tp, fn=fn, fp=fp, tn=tn)
        except httpx.ReadTimeout:
            logger.error("Could not resolve << %s >>", concept)
            e.q = 0
            return e
        return e
