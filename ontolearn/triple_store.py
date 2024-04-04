"""Triple store representations."""
import logging
import re
from itertools import chain
from typing import Iterable, Set, Optional, Generator, Union, FrozenSet, Tuple
import requests
from requests import Response
from requests.exceptions import RequestException, JSONDecodeError
from owlapy.owl2sparql.converter import Owl2SparqlConverter
from ontolearn.base.ext import OWLReasonerEx
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import OWLObjectPropertyRangeAxiom, OWLDataProperty, \
    OWLNamedIndividual, OWLClassExpression, OWLObjectPropertyExpression, OWLOntologyID, OWLOntology, \
    OWLThing, OWLObjectPropertyDomainAxiom, OWLLiteral, \
    OWLObjectInverseOf, OWLClass, \
    IRI, OWLDataPropertyRangeAxiom, OWLDataPropertyDomainAxiom, OWLClassAxiom, \
    OWLEquivalentClassesAxiom, OWLObjectProperty, OWLProperty, OWLDatatype, OWLObjectSomeValuesFrom

from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectOneOf, OWLObjectMinCardinality
import rdflib
from ontolearn.concept_generator import ConceptGenerator
from ontolearn.base.owl.utils import OWLClassExpressionLengthMetric
import traceback
from collections import Counter

logger = logging.getLogger(__name__)

rdfs_prefix = "PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n "
owl_prefix = "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n "
rdf_prefix = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n "
xsd_prefix = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"

# CD: For the sake of efficient software development.
limit_posix = ""


def rdflib_to_str(sparql_result: rdflib.plugins.sparql.processor.SPARQLResult) -> str:
    """
    @TODO: CD: Not quite sure whether we need this continuent function
    """
    for result_row in sparql_result:
        str_iri: str
        yield result_row.x.n3()


def is_valid_url(url) -> bool:
    """
    Check the validity of a URL.

    Args:
        url (str): The url to validate.

    Returns:
        True if url is not None, and it passes the regex check.

    """
    regex = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url is not None and regex.search(url)


def get_results_from_ts(triplestore_address: str, query: str, return_type: type):
    """
    Execute the SPARQL query in the given triplestore_address and return the result as the given return_type.

    Args:
        triplestore_address (str): The triplestore address where the query will be executed.
        query (str): SPARQL query where the root variable should be '?x'.
        return_type (type): OWLAPY class as type. e.g. OWLClass, OWLNamedIndividual, etc.

    Returns:
        Generator containing the results of the query as the given type.
    """
    try:
        response = requests.post(triplestore_address, data={'query': query})
    except RequestException as e:
        raise RequestException(f"Make sure the server is running on the `triplestore_address` = '{triplestore_address}'"
                               f". Check the error below:"
                               f"\n  -->Error: {e}")
    try:
        if return_type == OWLLiteral:
            yield from unwrap(response)
        else:
            yield from [return_type(i) for i in unwrap(response) if i is not None]
        # return [return_type(IRI.create(i['x']['value'])) for i in
        #         response.json()['results']['bindings']]
    except JSONDecodeError as e:
        raise JSONDecodeError(f"Something went wrong with decoding JSON from the response. Check for typos in "
                              f"the `triplestore_address` = '{triplestore_address}' otherwise the error is likely "
                              f"caused by an internal issue. \n  -->Error: {e}")


def unwrap(result: Response):
    json = result.json()
    vars_ = list(json['head']['vars'])
    for b in json['results']['bindings']:
        val = []
        for v in vars_:
            if b[v]['type'] == 'uri':
                val.append(IRI.create(b[v]['value']))
            elif b[v]['type'] == 'bnode':
                continue
            else:
                print(b[v]['type'])
                val.append(OWLLiteral(b[v]['value'], OWLDatatype(IRI.create(b[v]['datatype']))))
        if len(val) == 1:
            yield val.pop()
        else:
            yield None


def suf(direct: bool):
    """Put the star for rdfs properties depending on direct param"""
    return " " if direct else "* "


class TripleStoreOntology(OWLOntology):

    def __init__(self, triplestore_address: str):
        assert (is_valid_url(triplestore_address)), "You should specify a valid URL in the following argument: " \
                                                    "'triplestore_address' of class `TripleStore`"

        self.url = triplestore_address

    def classes_in_signature(self) -> Iterable[OWLClass]:
        query = owl_prefix + "SELECT DISTINCT ?x WHERE {?x a owl:Class.}"
        yield from get_results_from_ts(self.url, query, OWLClass)

    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:DatatypeProperty.}"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:ObjectProperty.}"
        yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:NamedIndividual.}"
        yield from get_results_from_ts(self.url, query, OWLNamedIndividual)

    def equivalent_classes_axioms(self, c: OWLClass) -> Iterable[OWLEquivalentClassesAxiom]:
        query = owl_prefix + "SELECT DISTINCT ?x" + \
                "WHERE { ?x owl:equivalentClass " + f"<{c.get_iri().as_str()}>." + \
                "FILTER(?x != " + f"<{c.get_iri().as_str()}>)}}"
        for cls in get_results_from_ts(self.url, query, OWLClass):
            yield OWLEquivalentClassesAxiom([c, cls])

    def general_class_axioms(self) -> Iterable[OWLClassAxiom]:
        raise NotImplementedError

    def data_property_domain_axioms(self, pe: OWLDataProperty) -> Iterable[OWLDataPropertyDomainAxiom]:
        domains = self._get_property_domains(pe)
        if len(domains) == 0:
            yield OWLDataPropertyDomainAxiom(pe, OWLThing)
        else:
            for dom in domains:
                yield OWLDataPropertyDomainAxiom(pe, dom)

    def data_property_range_axioms(self, pe: OWLDataProperty) -> Iterable[OWLDataPropertyRangeAxiom]:
        raise NotImplementedError

    def object_property_domain_axioms(self, pe: OWLObjectProperty) -> Iterable[OWLObjectPropertyDomainAxiom]:
        domains = self._get_property_domains(pe)
        if len(domains) == 0:
            yield OWLObjectPropertyDomainAxiom(pe, OWLThing)
        else:
            for dom in domains:
                yield OWLObjectPropertyDomainAxiom(pe, dom)

    def object_property_range_axioms(self, pe: OWLObjectProperty) -> Iterable[OWLObjectPropertyRangeAxiom]:
        query = rdfs_prefix + "SELECT ?x WHERE { " + f"<{pe.get_iri().as_str()}>" + " rdfs:range ?x. }"
        ranges = set(get_results_from_ts(self.url, query, OWLClass))
        if len(ranges) == 0:
            yield OWLObjectPropertyRangeAxiom(pe, OWLThing)
        else:
            for rng in ranges:
                yield OWLObjectPropertyRangeAxiom(pe, rng)

    def _get_property_domains(self, pe: OWLProperty):
        if isinstance(pe, OWLObjectProperty) or isinstance(pe, OWLDataProperty):
            query = rdfs_prefix + "SELECT ?x WHERE { " + f"<{pe.get_iri().as_str()}>" + " rdfs:domain ?x. }"
            domains = set(get_results_from_ts(self.url, query, OWLClass))
            return domains
        else:
            raise NotImplementedError

    def get_owl_ontology_manager(self):
        # no manager for this kind of Ontology
        pass

    def get_ontology_id(self) -> OWLOntologyID:
        # query = (rdf_prefix + owl_prefix +
        #          "SELECT ?ontologyIRI WHERE { ?ontology rdf:type owl:Ontology . ?ontology rdf:about ?ontologyIRI .}")
        # return list(get_results_from_ts(self.url, query, OWLOntologyID)).pop()
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.url == other.url
        return NotImplemented

    def __hash__(self):
        return hash(self.url)

    def __repr__(self):
        return f'TripleStoreOntology({self.url})'


class TripleStoreReasoner(OWLReasonerEx):
    __slots__ = 'ontology'

    def __init__(self, ontology: TripleStoreOntology):
        self.ontology = ontology
        self.url = self.ontology.url
        self._owl2sparql_converter = Owl2SparqlConverter()

    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        domains = {d.get_domain() for d in self.ontology.data_property_domain_axioms(pe)}
        sub_domains = set(chain.from_iterable([self.sub_classes(d) for d in domains]))
        yield from domains - sub_domains
        if not direct:
            yield from sub_domains

    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        domains = {d.get_domain() for d in self.ontology.object_property_domain_axioms(pe)}
        sub_domains = set(chain.from_iterable([self.sub_classes(d) for d in domains]))
        yield from domains - sub_domains
        if not direct:
            yield from sub_domains

    def object_property_ranges(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        ranges = {r.get_range() for r in self.ontology.object_property_range_axioms(pe)}
        sub_ranges = set(chain.from_iterable([self.sub_classes(d) for d in ranges]))
        yield from ranges - sub_ranges
        if not direct:
            yield from sub_ranges

    def equivalent_classes(self, ce: OWLClassExpression, only_named: bool = True) -> Iterable[OWLClassExpression]:
        if only_named:
            if isinstance(ce, OWLClass):
                query = owl_prefix + "SELECT DISTINCT ?x " + \
                        "WHERE { {?x owl:equivalentClass " + f"<{ce.get_iri().as_str()}>.}}" + \
                        "UNION {" + f"<{ce.get_iri().as_str()}>" + " owl:equivalentClass ?x.}" + \
                        "FILTER(?x != " + f"<{ce.get_iri().as_str()}>)}}"
                yield from get_results_from_ts(self.url, query, OWLClass)
            else:
                raise NotImplementedError("Equivalent classes for complex class expressions is not implemented")
        else:
            raise NotImplementedError("Finding equivalent complex classes is not implemented")

    def disjoint_classes(self, ce: OWLClassExpression, only_named: bool = True) -> Iterable[OWLClassExpression]:
        if only_named:
            if isinstance(ce, OWLClass):
                query = owl_prefix + " SELECT DISTINCT ?x " + \
                        "WHERE { " + f"<{ce.get_iri().as_str()}>" + " owl:disjointWith ?x .}"
                yield from get_results_from_ts(self.url, query, OWLClass)
            else:
                raise NotImplementedError("Disjoint classes for complex class expressions is not implemented")
        else:
            raise NotImplementedError("Finding disjoint complex classes is not implemented")

    def different_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        query = owl_prefix + rdf_prefix + "SELECT DISTINCT ?x \n" + \
                "WHERE{ ?allDifferent owl:distinctMembers/rdf:rest*/rdf:first ?x.\n" + \
                "?allDifferent owl:distinctMembers/rdf:rest*/rdf:first" + f"<{ind.str}>" + ".\n" + \
                "FILTER(?x != " + f"<{ind.str}>" + ")}"
        yield from get_results_from_ts(self.url, query, OWLNamedIndividual)

    def same_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        query = owl_prefix + "SELECT DISTINCT ?x " + \
                "WHERE {{ ?x owl:sameAs " + f"<{ind.str}>" + " .}" + \
                "UNION { " + f"<{ind.str}>" + " owl:sameAs ?x.}}"
        yield from get_results_from_ts(self.url, query, OWLNamedIndividual)

    def equivalent_object_properties(self, op: OWLObjectPropertyExpression) -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = owl_prefix + "SELECT DISTINCT ?x " + \
                    "WHERE { {?x owl:equivalentProperty " + f"<{op.get_iri().as_str()}>.}}" + \
                    "UNION {" + f"<{op.get_iri().as_str()}>" + " owl:equivalentProperty ?x.}" + \
                    "FILTER(?x != " + f"<{op.get_iri().as_str()}>)}}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = owl_prefix + "SELECT DISTINCT ?x " + \
                    "WHERE {  ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().get_iri().as_str()}> ." + \
                    " {?x owl:equivalentProperty ?inverseProperty .}" + \
                    "UNION { ?inverseProperty owl:equivalentClass ?x.}" + \
                    "FILTER(?x != ?inverseProperty }>)}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def equivalent_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x" + \
                "WHERE { {?x owl:equivalentProperty " + f"<{dp.get_iri().as_str()}>.}}" + \
                "UNION {" + f"<{dp.get_iri().as_str()}>" + " owl:equivalentProperty ?x.}" + \
                "FILTER(?x != " + f"<{dp.get_iri().as_str()}>)}}"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty, direct: bool = True) \
            -> Iterable[OWLLiteral]:
        query = "SELECT ?x WHERE { " + f"<{ind.str}>" + f"<{pe.get_iri().as_str()}>" + " ?x . }"
        yield from get_results_from_ts(self.url, query, OWLLiteral)
        if not direct:
            for prop in self.sub_data_properties(pe):
                yield from self.data_property_values(ind, prop, True)

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression, direct: bool = True) \
            -> Iterable[OWLNamedIndividual]:
        if isinstance(pe, OWLObjectProperty):
            query = "SELECT ?x WHERE { " + f"<{ind.str}> " + f"<{pe.get_iri().as_str()}>" + " ?x . }"
            yield from get_results_from_ts(self.url, query, OWLNamedIndividual)
        elif isinstance(pe, OWLObjectInverseOf):
            query = (owl_prefix + "SELECT ?x WHERE { ?inverseProperty owl:inverseOf " +
                     f"<{pe.get_inverse().get_iri().as_str()}>." +
                     f"<{ind.str}> ?inverseProperty ?x . }}")
            yield from get_results_from_ts(self.url, query, OWLNamedIndividual)
        if not direct:
            for prop in self.sub_object_properties(pe):
                yield from self.object_property_values(ind, prop, True)

    def flush(self) -> None:
        pass

    def instances(self, ce: OWLClassExpression, direct: bool = False, seen_set: Set = None) \
            -> Iterable[OWLNamedIndividual]:
        if not seen_set:
            seen_set = set()
            seen_set.add(ce)
        ce_to_sparql = self._owl2sparql_converter.as_query("?x", ce)
        if not direct:
            ce_to_sparql = ce_to_sparql.replace("?x a ", "?x a ?some_cls. \n ?some_cls "
                                                         "<http://www.w3.org/2000/01/rdf-schema#subClassOf>* ")
        yield from get_results_from_ts(self.url, ce_to_sparql, OWLNamedIndividual)
        if not direct:
            for cls in self.equivalent_classes(ce):
                if cls not in seen_set:
                    seen_set.add(cls)
                    yield from self.instances(cls, direct, seen_set)

    def sub_classes(self, ce: OWLClassExpression, direct: bool = False, only_named: bool = True) \
            -> Iterable[OWLClassExpression]:
        if not only_named:
            raise NotImplementedError("Finding anonymous subclasses not implemented")
        if isinstance(ce, OWLClass):
            query = rdfs_prefix + \
                    "SELECT ?x WHERE { ?x rdfs:subClassOf" + suf(direct) + f"<{ce.get_iri().as_str()}>" + ". }"
            results = list(get_results_from_ts(self.url, query, OWLClass))
            if ce in results:
                results.remove(ce)
            yield from results
        else:
            raise NotImplementedError("Subclasses of complex classes retrieved via triple store is not implemented")
            # query = "PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
            #         "SELECT DISTINCT ?x WHERE { ?x rdfs:subClassOf" + suf(direct) + " ?c. \n" \
            #         "?s a ?c . \n"
            # ce_to_sparql_statements = self._owl2sparql_converter.convert("?s", ce)
            # for s in ce_to_sparql_statements:
            #     query = query + s + "\n"
            # query = query + "}"
            # yield from get_results_from_ts(self._triplestore_address, query, OWLClass)

    def super_classes(self, ce: OWLClassExpression, direct: bool = False, only_named: bool = True) \
            -> Iterable[OWLClassExpression]:
        if not only_named:
            raise NotImplementedError("Finding anonymous superclasses not implemented")
        if isinstance(ce, OWLClass):
            if ce == OWLThing:
                return []
            query = rdfs_prefix + \
                    "SELECT ?x WHERE { " + f"<{ce.get_iri().as_str()}>" + " rdfs:subClassOf" + suf(direct) + "?x. }"
            results = list(get_results_from_ts(self.url, query, OWLClass))
            if ce in results:
                results.remove(ce)
            if (not direct and OWLThing not in results) or len(results) == 0:
                results.append(OWLThing)
            yield from results
        else:
            raise NotImplementedError("Superclasses of complex classes retrieved via triple store is not "
                                      "implemented")

    def disjoint_object_properties(self, op: OWLObjectPropertyExpression) -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = owl_prefix + rdf_prefix + "SELECT DISTINCT ?x \n" + \
                    "WHERE{ ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n" + \
                    "?AllDisjointProperties owl:members/rdf:rest*/rdf:first" + f"<{op.get_iri().as_str()}>" + ".\n" + \
                    "FILTER(?x != " + f"<{op.get_iri().as_str()}>" + ")}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = owl_prefix + " SELECT DISTINCT ?x " + \
                    "WHERE {  ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().get_iri().as_str()}> ." + \
                    " ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n" + \
                    " ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?inverseProperty.\n" + \
                    " FILTER(?x != ?inverseProperty)}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def disjoint_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        query = owl_prefix + rdf_prefix + "SELECT DISTINCT ?x \n" + \
                "WHERE{ ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n" + \
                "?AllDisjointProperties owl:members/rdf:rest*/rdf:first" + f"<{dp.get_iri().as_str()}>" + ".\n" + \
                "FILTER(?x != " + f"<{dp.get_iri().as_str()}>" + ")}"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def all_data_property_values(self, pe: OWLDataProperty, direct: bool = True) -> Iterable[OWLLiteral]:
        query = "SELECT DISTINCT ?x WHERE { ?y" + f"<{pe.get_iri().as_str()}>" + " ?x . }"
        yield from get_results_from_ts(self.url, query, OWLLiteral)
        if not direct:
            for prop in self.sub_data_properties(pe):
                yield from self.all_data_property_values(prop, True)

    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        query = rdfs_prefix + \
                "SELECT ?x WHERE { ?x rdfs:subPropertyOf" + suf(direct) + f"<{dp.get_iri().as_str()}>" + ". }"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def super_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        query = rdfs_prefix + \
                "SELECT ?x WHERE {" + f"<{dp.get_iri().as_str()}>" + " rdfs:subPropertyOf" + suf(direct) + " ?x. }"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (rdfs_prefix + "SELECT ?x WHERE { ?x rdfs:subPropertyOf"
                     + suf(direct) + f"<{op.get_iri().as_str()}> . FILTER(?x != " + f"<{op.get_iri().as_str()}>) }}")
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (rdfs_prefix + "SELECT ?x " +
                     "WHERE { ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().get_iri().as_str()}> ." +
                     " ?x rdfs:subPropertyOf" + suf(direct) + " ?inverseProperty . }")
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def super_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (rdfs_prefix + "SELECT ?x WHERE {" + f"<{op.get_iri().as_str()}>" + " rdfs:subPropertyOf"
                     + suf(direct) + " ?x. FILTER(?x != " + f"<{op.get_iri().as_str()}>) }}")
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (rdfs_prefix + "SELECT ?x " +
                     "WHERE { ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().get_iri().as_str()}> ." +
                     " ?inverseProperty rdfs:subPropertyOf" + suf(direct) + "?x  . }")
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        if direct:
            query = "SELECT ?x WHERE {" + f"<{ind.str}> a" + " ?x. }"
        else:
            query = rdfs_prefix + "SELECT DISTINCT ?x WHERE {" + f"<{ind.str}> a ?cls. " \
                                                                 " ?cls rdfs:subClassOf* ?x}"
        yield from [i for i in get_results_from_ts(self.url, query, OWLClass)
                    if i != OWLClass(IRI('http://www.w3.org/2002/07/owl#', 'NamedIndividual'))]

    def get_root_ontology(self) -> OWLOntology:
        return self.ontology

    def is_isolated(self):
        # not needed here
        pass

    def is_using_triplestore(self):
        """No use! Deprecated."""
        # TODO: Deprecated! Remove after it is removed from OWLReasoner in owlapy
        pass


class TripleStoreKnowledgeBase(KnowledgeBase):
    url: str
    ontology: TripleStoreOntology
    reasoner: TripleStoreReasoner

    def __init__(self, triplestore_address: str):
        self.url = triplestore_address
        self.ontology = TripleStoreOntology(triplestore_address)
        self.reasoner = TripleStoreReasoner(self.ontology)
        super().__init__(ontology=self.ontology, reasoner=self.reasoner)


class TripleStoreReasonerOntology:

    def __init__(self, graph: rdflib.graph.Graph, url: str = None):
        self.g = graph
        self.url = url
        self.converter = Owl2SparqlConverter()
        # A convenience to distinguish type predicate from other predicates in the results of SPARQL query
        self.type_predicate = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"

    def concise_bounded_description(self, str_iri: str) -> Generator[
        Tuple[OWLNamedIndividual, Union[IRI, OWLObjectProperty], Union[OWLClass, OWLNamedIndividual]], None, None]:
        """
        https://www.w3.org/submissions/CBD/
        also see https://docs.aws.amazon.com/neptune/latest/userguide/sparql-query-hints-for-describe.html

        Given a particular node (the starting node) in a particular RDF graph (the source graph),
        a subgraph of that particular graph, taken to comprise a concise bounded description of the resource denoted by the starting node, can be identified as follows:

        Include in the subgraph all statements in the source graph where the subject of the statement is the starting node;
        Recursively, for all statements identified in the subgraph thus far having a blank node object, include in the subgraph all statements in the source graph
        where the subject of the statement is the blank node in question and which are not already included in the subgraph.
        Recursively, for all statements included in the subgraph thus far, for all reifications of each statement in the source graph, include the concise bounded description beginning from the rdf:Statement node of each reification.
        his results in a subgraph where the object nodes are either URI references, literals, or blank nodes not serving as the subject of any statement in the graph.
        """
        # CD: We can allivate the object creations by creating a dictionary of created instances of
        for (s, p, o) in self.query(sparql_query=f"""DESCRIBE <{str_iri}>"""):
            if p.n3() == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                assert isinstance(p, rdflib.term.URIRef)
                assert isinstance(o, rdflib.term.URIRef)
                yield OWLNamedIndividual(IRI.create(s.n3()[1:-1])), IRI.create(p.n3()[1:-1]), OWLClass(
                    IRI.create(o.n3()[1:-1]))
            else:
                assert isinstance(p, rdflib.term.URIRef)
                assert isinstance(o, rdflib.term.URIRef)
                # @TODO: CD: Can we safely assume that the object always be owl individuals ?
                # @TODO: CD: Can we safely assume that the property always be Objet property?
                yield OWLNamedIndividual(IRI.create(s.n3()[1:-1])), OWLObjectProperty(
                    IRI.create(p.n3()[1:-1])), OWLNamedIndividual(IRI.create(o.n3()[1:-1]))

    def abox(self, str_iri: str) -> Generator[
        Tuple[OWLNamedIndividual, Union[IRI, OWLObjectProperty], Union[OWLClass, OWLNamedIndividual]], None, None]:
        """
        Get all axioms of a given individual being a subject entity

        Args:
            str_iri (str): An individual
            mode (str): The return format.
             1) 'native' -> returns triples as tuples of owlapy objects,
             2) 'iri' -> returns triples as tuples of IRIs as string,
             3) 'axiom' -> triples are represented by owlapy axioms.

        Returns: Iterable of tuples or owlapy axiom, depending on the mode.
        """
        sparql_query = f"SELECT DISTINCT ?p ?o WHERE {{ <{str_iri}> ?p ?o }}"
        # CD: Although subject_ is not required. Arguably, it is more in to return also the subject_
        subject_ = OWLNamedIndividual(IRI.create(str_iri))

        predicate_and_object_pairs: rdflib.query.ResultRow
        for predicate_and_object_pairs in self.query(sparql_query):
            p, o = predicate_and_object_pairs
            assert isinstance(p, rdflib.term.URIRef) and isinstance(o,
                                                                    rdflib.term.URIRef), f"Currently we only process URIs. Hence, literals, data properties  are ignored. p:{p},o:{o}"
            str_p = p.n3()
            str_o = o.n3()
            if str_p == self.type_predicate:
                # Remove the brackets <>,<>
                yield subject_, IRI.create(str_p[1:-1]), OWLClass(IRI.create(str_o[1:-1]))
            else:
                yield subject_, OWLObjectProperty(IRI.create(str_p[1:-1])), OWLNamedIndividual(IRI.create(str_o[1:-1]))

    def query(self, sparql_query: str) -> rdflib.plugins.sparql.processor.SPARQLResult:
        return self.g.query(sparql_query)

    def classes_in_signature(self) -> Iterable[OWLClass]:
        query = owl_prefix + """SELECT DISTINCT ?x WHERE { ?x a owl:Class }"""
        for str_iri in rdflib_to_str(sparql_result=self.query(query)):
            assert str_iri[0] == "<" and str_iri[-1] == ">"
            yield OWLClass(IRI.create(str_iri[1:-1]))

    def subconcepts(self, named_concept: OWLClass, direct=True):
        assert isinstance(named_concept, OWLClass)
        str_named_concept = f"<{named_concept.get_iri().as_str()}>"
        if direct:
            query = f"""{rdfs_prefix} SELECT ?x WHERE {{ ?x rdfs:subClassOf* {str_named_concept}. }} """
        else:
            query = f"""{rdf_prefix} SELECT ?x WHERE {{ ?x rdf:subClassOf {str_named_concept}. }} """
        for str_iri in rdflib_to_str(sparql_result=self.query(query)):
            assert str_iri[0] == "<" and str_iri[-1] == ">"
            yield OWLClass(IRI.create(str_iri[1:-1]))

    def get_type_individuals(self, individual: str):
        query = f"""SELECT DISTINCT ?x WHERE {{ <{individual}> a ?x }}"""
        for str_iri in rdflib_to_str(sparql_result=self.query(query)):
            assert str_iri[0] == "<" and str_iri[-1] == ">"
            yield OWLClass(IRI.create(str_iri[1:-1]))

    def instances(self, expression: OWLClassExpression):
        assert isinstance(expression, OWLClassExpression)
        # convert to SPARQL query
        # (1)
        try:
            query = self.converter.as_query("?x", expression)
        except Exception as exc:
            # @TODO creating a SPARQL query from OWLObjectMinCardinality causes a problem.
            print(f"Error at converting {expression} into sparql")
            traceback.print_exception(exc)
            print(f"Error at converting {expression} into sparql")
            query = None
        if query:
            for str_iri in rdflib_to_str(sparql_result=self.query(query)):
                assert str_iri[0] == "<" and str_iri[-1] == ">"
                yield OWLNamedIndividual(IRI.create(str_iri[1:-1]))
        else:
            yield

    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        # owl:OWLNamedIndividual is often missing: Perhaps we should add union as well
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a ?y. ?y a owl:Class.}"
        for str_iri in rdflib_to_str(sparql_result=self.query(query)):
            assert str_iri[0] == "<" and str_iri[-1] == ">"
            yield OWLNamedIndividual(IRI.create(str_iri[1:-1]))

    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:DatatypeProperty.}"
        for str_iri in rdflib_to_str(sparql_result=self.query(query)):
            assert str_iri[0] == "<" and str_iri[-1] == ">"
            yield OWLDataProperty(IRI.create(str_iri[1:-1]))

    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:ObjectProperty.}"
        for str_iri in rdflib_to_str(sparql_result=self.query(query)):
            assert str_iri[0] == "<" and str_iri[-1] == ">"
            yield OWLObjectProperty(IRI.create(str_iri[1:-1]))

    def boolean_data_properties(self):
        # @TODO: Double check the SPARQL query to return all boolean data properties
        query = rdf_prefix + xsd_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x rdf:type rdf:Property; rdfs:range xsd:boolean}"
        for str_iri in rdflib_to_str(sparql_result=self.query(query)):
            assert str_iri[0] == "<" and str_iri[-1] == ">"
            raise NotImplementedError("Unsure how to represent a boolean data proerty with owlapy")
            # yield OWLObjectProperty(IRI.create(str_iri[1:-1]))

        yield


class TripleStore:
    """ triple store """
    path: str
    url: str

    def __init__(self, path: str = None, url: str = None):

        # Single object to replace the
        if path:
            self.g = TripleStoreReasonerOntology(rdflib.Graph().parse(path))
        else:
            self.g = TripleStoreReasonerOntology(rdflib.Graph(), url=url)

        self.ontology = self.g
        self.reasoner = self.g
        # CD: We may want to remove it later. This is required at base_concept_learner.py
        self.generator = ConceptGenerator()
        self.length_metric = OWLClassExpressionLengthMetric.get_default()

    def concise_bounded_description(self, individual: OWLNamedIndividual, mode: str = "native") -> Generator[
        Tuple[OWLNamedIndividual, Union[IRI, OWLObjectProperty], Union[OWLClass, OWLNamedIndividual]], None, None]:
        """

        Get the CBD (https://www.w3.org/submissions/CBD/) of a named individual.

        Args:
            individual (OWLNamedIndividual): Individual to get the abox axioms from.
            mode (str): The return format.
             1) 'native' -> returns triples as tuples of owlapy objects,
             2) 'iri' -> returns triples as tuples of IRIs as string,
             3) 'axiom' -> triples are represented by owlapy axioms.

        Returns: Iterable of tuples or owlapy axiom, depending on the mode.
        """
        assert mode in ['native', 'iri', 'axiom'], "Valid modes are: 'native', 'iri' or 'axiom'"
        if mode == "native":
            yield from self.g.concise_bounded_description(str_iri=individual.get_iri().as_str())

        elif mode == "iri":
            raise NotImplementedError("Mode==iri has not been implemented yet.")
            yield from ((i.str, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                         t.get_iri().as_str()) for t in self.get_types(ind=i, direct=True))
            for dp in self.get_data_properties_for_ind(ind=i):
                yield from ((i.str, dp.get_iri().as_str(), literal.get_literal()) for literal in
                            self.get_data_property_values(i, dp))
            for op in self.get_object_properties_for_ind(ind=i):
                yield from ((i.str, op.get_iri().as_str(), ind.get_iri().as_str()) for ind in
                            self.get_object_property_values(i, op))
        elif mode == "axiom":
            raise NotImplementedError("Mode==axiom has not been implemented yet.")

            yield from (OWLClassAssertionAxiom(i, t) for t in self.get_types(ind=i, direct=True))
            for dp in self.get_data_properties_for_ind(ind=i):
                yield from (OWLDataPropertyAssertionAxiom(i, dp, literal) for literal in
                            self.get_data_property_values(i, dp))
            for op in self.get_object_properties_for_ind(ind=i):
                yield from (OWLObjectPropertyAssertionAxiom(i, op, ind) for ind in
                            self.get_object_property_values(i, op))

    def abox(self, individual: OWLNamedIndividual, mode: str = "native") -> Generator[
        Tuple[OWLNamedIndividual, Union[IRI, OWLObjectProperty], Union[OWLClass, OWLNamedIndividual]], None, None]:
        """

        Get all axioms of a given individual being a subject entity

        Args:
            individual (OWLNamedIndividual): An individual
            mode (str): The return format.
             1) 'native' -> returns triples as tuples of owlapy objects,
             2) 'iri' -> returns triples as tuples of IRIs as string,
             3) 'axiom' -> triples are represented by owlapy axioms.
             4) 'expression' -> unique owl class expressions based on (1).

        Returns: Iterable of tuples or owlapy axiom, depending on the mode.
        """
        assert mode in ['native', 'iri', 'axiom',
                        "expression"], "Valid modes are: 'native', 'iri' or 'axiom', 'expression'"
        if mode == "native":
            yield from self.g.abox(str_iri=individual.get_iri().as_str())

        elif mode == "iri":
            raise NotImplementedError("Mode==iri has not been implemented yet.")
            yield from ((i.str, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                         t.get_iri().as_str()) for t in self.get_types(ind=i, direct=True))
            for dp in self.get_data_properties_for_ind(ind=i):
                yield from ((i.str, dp.get_iri().as_str(), literal.get_literal()) for literal in
                            self.get_data_property_values(i, dp))
            for op in self.get_object_properties_for_ind(ind=i):
                yield from ((i.str, op.get_iri().as_str(), ind.get_iri().as_str()) for ind in
                            self.get_object_property_values(i, op))
        elif mode == "expression":
            mapping = dict()
            # To no return duplicate objects.
            quantifier_gate = set()
            # (1) Iterate over triples where individual is in the subject position.
            for s, p, o in self.g.abox(str_iri=individual.get_iri().as_str()):
                if isinstance(p, IRI) and isinstance(o, OWLClass):
                    # RETURN MEMBERSHIP/Type INFORMATION: C(s)
                    yield o
                elif isinstance(p, OWLObjectProperty) and isinstance(o, OWLNamedIndividual):
                    mapping.setdefault(p, []).append(o)
                else:
                    raise RuntimeError("Unrecognized triples to expression mappings")

            for k, iter_inds in mapping.items():
                # RETURN Existential Quantifiers over Nominals: \exists r. {x....y}
                for x in iter_inds:
                    yield OWLObjectSomeValuesFrom(property=k, filler=OWLObjectOneOf(x))
                type_: OWLClass
                count: int
                for type_, count in Counter(
                        [type_i for i in iter_inds for type_i in self.get_types(ind=i, direct=True)]).items():
                    min_cardinality_item = OWLObjectMinCardinality(cardinality=count, property=k, filler=type_)
                    if min_cardinality_item in quantifier_gate:
                        continue
                    else:
                        quantifier_gate.add(min_cardinality_item)
                        # RETURN \ge number r. C
                        yield min_cardinality_item
                    existential_quantifier = OWLObjectSomeValuesFrom(property=k, filler=type_)
                    if existential_quantifier in quantifier_gate:
                        continue
                    else:
                        # RETURN Existential Quantifiers over Concepts: \exists r. C
                        quantifier_gate.add(existential_quantifier)
                        yield existential_quantifier
        elif mode == "axiom":
            raise NotImplementedError("Axioms should be checked.")
            yield from (OWLClassAssertionAxiom(i, t) for t in self.get_types(ind=i, direct=True))
            for dp in self.get_data_properties_for_ind(ind=i):
                yield from (OWLDataPropertyAssertionAxiom(i, dp, literal) for literal in
                            self.get_data_property_values(i, dp))
            for op in self.get_object_properties_for_ind(ind=i):
                yield from (OWLObjectPropertyAssertionAxiom(i, op, ind) for ind in
                            self.get_object_property_values(i, op))

    def get_object_properties(self):
        yield from self.reasoner.object_properties_in_signature()

    def get_boolean_data_properties(self):
        yield from self.reasoner.boolean_data_properties()

    def individuals(self, concept: Optional[OWLClassExpression] = None) -> Iterable[OWLNamedIndividual]:
        """Given an OWL class expression, retrieve all individuals belonging to it.


        Args:
            concept: Class expression of which to list individuals.
        Returns:
            Individuals belonging to the given class.
        """

        if concept is None or concept.is_owl_thing():
            yield from self.reasoner.individuals_in_signature()
        else:
            yield from self.reasoner.instances(concept)

    def get_types(self, ind: OWLNamedIndividual, direct: True) -> Generator[OWLClass, None, None]:
        if not direct:
            raise NotImplementedError("Inferring indirect types not available")
        return self.reasoner.get_type_individuals(ind.str)

    def get_all_sub_concepts(self, concept: OWLClass, direct=True):
        yield from self.reasoner.subconcepts(concept, direct)

    def named_concepts(self):
        yield from self.reasoner.classes_in_signature()

    def quality_retrieval(self, expression: OWLClass, pos: set[OWLNamedIndividual], neg: set[OWLNamedIndividual]):
        assert isinstance(expression,
                          OWLClass), "Currently we can only compute the F1 score of a named concepts given pos and neg"

        sparql_str = f"{self.dbo_prefix}{self.rdf_prefix}"
        num_pos = len(pos)
        str_concept_reminder = expression.get_iri().get_remainder()

        str_concept = expression.get_iri().as_str()
        str_pos = " ".join(("<" + i.str + ">" for i in pos))
        str_neg = " ".join(("<" + i.str + ">" for i in neg))

        # TODO
        sparql_str += f"""
        SELECT ?tp ?fp ?fn
        WHERE {{ 

        {{SELECT DISTINCT (COUNT(?var) as ?tp) ( {num_pos}-COUNT(?var) as ?fn) 
        WHERE {{ VALUES ?var {{ {str_pos} }} ?var rdf:type dbo:{str_concept_reminder} .}} }}

        {{SELECT DISTINCT (COUNT(?var) as ?fp)
        WHERE {{ VALUES ?var  {{ {str_neg} }} ?var rdf:type dbo:{str_concept_reminder} .}} }}

        }}
        """

        response = requests.post('http://dice-dbpedia.cs.upb.de:9080/sparql', auth=("", ""),
                                 data=sparql_str,
                                 headers={"Content-Type": "application/sparql-query"})
        bindings = response.json()["results"]["bindings"]
        assert len(bindings) == 1
        results = bindings.pop()
        assert len(results) == 3
        tp = int(results["tp"]["value"])
        fp = int(results["fp"]["value"])
        fn = int(results["fn"]["value"])
        # Compute recall (Sensitivity): Relevant retrieved instances / all relevant instances.
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
        # Compute recall (Sensitivity): Relevant retrieved instances / all retrieved instances.
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        f1 = 0 if precision == 0 or recall == 0 else 2 * ((precision * recall) / (precision + recall))

        return f1

    def query(self, sparql: str) -> rdflib.plugins.sparql.processor.SPARQLResult:
        yield from self.g.query(sparql_query=sparql)

    def concept_len(self, ce: OWLClassExpression) -> int:
        """Calculates the length of a concept and is used by some concept learning algorithms to
        find the best results considering also the length of the concepts.

        Args:
            ce: The concept to be measured.
        Returns:
            Length of the concept.
        """

        return self.length_metric.length(ce)

    def individuals_set(self,
                        arg: Union[Iterable[OWLNamedIndividual], OWLNamedIndividual, OWLClassExpression]) -> FrozenSet:
        """Retrieve the individuals specified in the arg as a frozenset. If `arg` is an OWLClassExpression then this
        method behaves as the method "individuals" but will return the final result as a frozenset.

        Args:
            arg: more than one individual/ single individual/ class expression of which to list individuals.
        Returns:
            Frozenset of the individuals depending on the arg type.

        UPDATE: CD: This function should be deprecated it does not introduce any new functionality but coves a rewriting
        ,e .g. if args needs to be a frozen set, doing frozenset(arg) solves this need without introducing this function
        """

        if isinstance(arg, OWLClassExpression):
            return frozenset(self.individuals(arg))
        elif isinstance(arg, OWLNamedIndividual):
            return frozenset({arg})
        else:
            return frozenset(arg)
