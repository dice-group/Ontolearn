"""Triple store representations."""
import logging
import re
from itertools import chain
from typing import Iterable, Set, Optional, Generator, Union, FrozenSet, Tuple
import requests
from owlapy.class_expression import OWLClassExpression, OWLThing, OWLClass, OWLObjectSomeValuesFrom, OWLObjectOneOf, \
    OWLObjectMinCardinality
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLObjectPropertyRangeAxiom, OWLObjectPropertyDomainAxiom, OWLDataPropertyRangeAxiom, \
    OWLDataPropertyDomainAxiom, OWLClassAxiom, OWLEquivalentClassesAxiom
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_ontology import OWLOntologyID, OWLOntology
from owlapy.owl_property import OWLDataProperty, OWLObjectPropertyExpression, OWLObjectInverseOf, OWLObjectProperty, \
    OWLProperty
from requests import Response
from requests.exceptions import RequestException, JSONDecodeError
from owlapy.converter import Owl2SparqlConverter
from ontolearn.base.ext import OWLReasonerEx
from ontolearn.knowledge_base import KnowledgeBase
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

from owlapy import owl_expression_to_sparql
from owlapy.class_expression import OWLObjectHasValue, OWLDataHasValue, OWLDataSomeValuesFrom, OWLDataOneOf
from typing import List
from owlapy.owl_property import OWLProperty


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
                "WHERE { ?x owl:equivalentClass " + f"<{c.str}>." + \
                "FILTER(?x != " + f"<{c.str}>)}}"
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
        query = rdfs_prefix + "SELECT ?x WHERE { " + f"<{pe.str}>" + " rdfs:range ?x. }"
        ranges = set(get_results_from_ts(self.url, query, OWLClass))
        if len(ranges) == 0:
            yield OWLObjectPropertyRangeAxiom(pe, OWLThing)
        else:
            for rng in ranges:
                yield OWLObjectPropertyRangeAxiom(pe, rng)

    def _get_property_domains(self, pe: OWLProperty):
        if isinstance(pe, OWLObjectProperty) or isinstance(pe, OWLDataProperty):
            query = rdfs_prefix + "SELECT ?x WHERE { " + f"<{pe.str}>" + " rdfs:domain ?x. }"
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
                        "WHERE { {?x owl:equivalentClass " + f"<{ce.str}>.}}" + \
                        "UNION {" + f"<{ce.str}>" + " owl:equivalentClass ?x.}" + \
                        "FILTER(?x != " + f"<{ce.str}>)}}"
                yield from get_results_from_ts(self.url, query, OWLClass)
            else:
                raise NotImplementedError("Equivalent classes for complex class expressions is not implemented")
        else:
            raise NotImplementedError("Finding equivalent complex classes is not implemented")

    def disjoint_classes(self, ce: OWLClassExpression, only_named: bool = True) -> Iterable[OWLClassExpression]:
        if only_named:
            if isinstance(ce, OWLClass):
                query = owl_prefix + " SELECT DISTINCT ?x " + \
                        "WHERE { " + f"<{ce.str}>" + " owl:disjointWith ?x .}"
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
                    "WHERE { {?x owl:equivalentProperty " + f"<{op.str}>.}}" + \
                    "UNION {" + f"<{op.str}>" + " owl:equivalentProperty ?x.}" + \
                    "FILTER(?x != " + f"<{op.str}>)}}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = owl_prefix + "SELECT DISTINCT ?x " + \
                    "WHERE {  ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ." + \
                    " {?x owl:equivalentProperty ?inverseProperty .}" + \
                    "UNION { ?inverseProperty owl:equivalentClass ?x.}" + \
                    "FILTER(?x != ?inverseProperty }>)}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def equivalent_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x" + \
                "WHERE { {?x owl:equivalentProperty " + f"<{dp.str}>.}}" + \
                "UNION {" + f"<{dp.str}>" + " owl:equivalentProperty ?x.}" + \
                "FILTER(?x != " + f"<{dp.str}>)}}"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty, direct: bool = True) \
            -> Iterable[OWLLiteral]:
        query = "SELECT ?x WHERE { " + f"<{ind.str}>" + f"<{pe.str}>" + " ?x . }"
        yield from get_results_from_ts(self.url, query, OWLLiteral)
        if not direct:
            for prop in self.sub_data_properties(pe):
                yield from self.data_property_values(ind, prop, True)

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression, direct: bool = True) \
            -> Iterable[OWLNamedIndividual]:
        if isinstance(pe, OWLObjectProperty):
            query = "SELECT ?x WHERE { " + f"<{ind.str}> " + f"<{pe.str}>" + " ?x . }"
            yield from get_results_from_ts(self.url, query, OWLNamedIndividual)
        elif isinstance(pe, OWLObjectInverseOf):
            query = (owl_prefix + "SELECT ?x WHERE { ?inverseProperty owl:inverseOf " +
                     f"<{pe.get_inverse().str}>." +
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
                    "SELECT ?x WHERE { ?x rdfs:subClassOf" + suf(direct) + f"<{ce.str}>" + ". }"
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
                    "SELECT ?x WHERE { " + f"<{ce.str}>" + " rdfs:subClassOf" + suf(direct) + "?x. }"
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
                    "?AllDisjointProperties owl:members/rdf:rest*/rdf:first" + f"<{op.str}>" + ".\n" + \
                    "FILTER(?x != " + f"<{op.str}>" + ")}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = owl_prefix + " SELECT DISTINCT ?x " + \
                    "WHERE {  ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ." + \
                    " ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n" + \
                    " ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?inverseProperty.\n" + \
                    " FILTER(?x != ?inverseProperty)}"
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def disjoint_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        query = owl_prefix + rdf_prefix + "SELECT DISTINCT ?x \n" + \
                "WHERE{ ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n" + \
                "?AllDisjointProperties owl:members/rdf:rest*/rdf:first" + f"<{dp.str}>" + ".\n" + \
                "FILTER(?x != " + f"<{dp.str}>" + ")}"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def all_data_property_values(self, pe: OWLDataProperty, direct: bool = True) -> Iterable[OWLLiteral]:
        query = "SELECT DISTINCT ?x WHERE { ?y" + f"<{pe.str}>" + " ?x . }"
        yield from get_results_from_ts(self.url, query, OWLLiteral)
        if not direct:
            for prop in self.sub_data_properties(pe):
                yield from self.all_data_property_values(prop, True)

    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        query = rdfs_prefix + \
                "SELECT ?x WHERE { ?x rdfs:subPropertyOf" + suf(direct) + f"<{dp.str}>" + ". }"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def super_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        query = rdfs_prefix + \
                "SELECT ?x WHERE {" + f"<{dp.str}>" + " rdfs:subPropertyOf" + suf(direct) + " ?x. }"
        yield from get_results_from_ts(self.url, query, OWLDataProperty)

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (rdfs_prefix + "SELECT ?x WHERE { ?x rdfs:subPropertyOf"
                     + suf(direct) + f"<{op.str}> . FILTER(?x != " + f"<{op.str}>) }}")
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (rdfs_prefix + "SELECT ?x " +
                     "WHERE { ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ." +
                     " ?x rdfs:subPropertyOf" + suf(direct) + " ?inverseProperty . }")
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)

    def super_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (rdfs_prefix + "SELECT ?x WHERE {" + f"<{op.str}>" + " rdfs:subPropertyOf"
                     + suf(direct) + " ?x. FILTER(?x != " + f"<{op.str}>) }}")
            yield from get_results_from_ts(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (rdfs_prefix + "SELECT ?x " +
                     "WHERE { ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ." +
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


#######################################################################################################################


class TripleStoreReasonerOntology:

    def __init__(self, url: str = None):
        assert url is not None, "URL cannot be None"
        self.url = url

    def query(self, sparql_query: str):
        return requests.Session().post(self.url, data={'query': sparql_query})  #.json()["results"]["bindings"]

    def are_owl_concept_disjoint(self, c: OWLClass, cc: OWLClass) -> bool:
        query = f"""{owl_prefix}ASK WHERE {{<{c.str}> owl:disjointWith <{cc.str}> .}}"""
        # Workaround self.query doesn't work for ASK at the moment
        return requests.Session().post(self.url, data={'query': query}).json()["boolean"]

    def abox(self, str_iri: str) -> Generator[Tuple[
        Tuple[OWLNamedIndividual, OWLProperty, OWLClass],
        Tuple[OWLObjectProperty, OWLObjectProperty, OWLNamedIndividual],
        Tuple[OWLObjectProperty, OWLDataProperty, OWLLiteral]], None, None]:
        """@TODO:"""
        sparql_query = f"SELECT DISTINCT ?p ?o WHERE {{ <{str_iri}> ?p ?o }}"
        subject_ = OWLNamedIndividual(str_iri)
        for binding in self.query(sparql_query).json()["results"]["bindings"]:
            p, o = binding["p"], binding["o"]
            # ORDER MATTERS
            if p["value"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                yield subject_, OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), OWLClass(o["value"])
            elif o["type"] == "uri":
                #################################################################
                # IMPORTANT
                # Can we assume that if o has URI and is not owl class, then o can be considered as an individual ?
                #################################################################
                yield subject_, OWLObjectProperty(p["value"]), OWLNamedIndividual(o["value"])
            elif o["type"] == "literal":
                if o["datatype"] == "http://www.w3.org/2001/XMLSchema#boolean":
                    yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=bool(o["value"]))
                elif o["datatype"] == "http://www.w3.org/2001/XMLSchema#double":
                    yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=float(o["value"]))
                else:
                    raise NotImplementedError(f"Currently this type of literal is not supported:{o} "
                                              f"but can done easily let us know :)")
            else:
                raise RuntimeError(f"Unrecognized type {subject_} ({p}) ({o})")

    def classes_in_signature(self) -> Iterable[OWLClass]:
        query = owl_prefix + """SELECT DISTINCT ?x WHERE { ?x a owl:Class }"""
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLClass(binding["x"]["value"])

    def most_general_classes(self) -> Iterable[OWLClass]:
        """ At least it has single subclass and there is no superclass """
        query = f"""{rdf_prefix}{rdfs_prefix}{owl_prefix} SELECT ?x WHERE {{
        ?concept rdf:type owl:Class .
        FILTER EXISTS {{ ?x rdfs:subClassOf ?z . }}
        FILTER NOT EXISTS {{ ?y rdfs:subClassOf ?x . }}
        }}
        """
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLClass(binding["x"]["value"])

    def least_general_named_concepts(self) -> Generator[OWLClass, None, None]:
        """ At least it has single superclass and there is no subclass """
        query = f"""{rdf_prefix}{rdfs_prefix}{owl_prefix} SELECT ?concept WHERE {{
        ?concept rdf:type owl:Class .
        FILTER EXISTS {{ ?concept rdfs:subClassOf ?x . }}
        FILTER NOT EXISTS {{ ?y rdfs:subClassOf ?concept . }}
        }}"""
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLClass(binding["concept"]["value"])

    def get_direct_parents(self, named_concept: OWLClass):
        """ Father rdf:subClassOf Person"""
        assert isinstance(named_concept, OWLClass)
        str_named_concept = f"<{named_concept.str}>"
        query = f"""{rdfs_prefix} SELECT ?x WHERE {{ {str_named_concept} rdfs:subClassOf ?x . }} """
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLClass(binding["x"]["value"])

    def subconcepts(self, named_concept: OWLClass, direct=True):
        assert isinstance(named_concept, OWLClass)
        str_named_concept = f"<{named_concept.str}>"
        if direct:
            query = f"""{rdfs_prefix} SELECT ?x WHERE {{ ?x rdfs:subClassOf* {str_named_concept}. }} """
        else:
            query = f"""{rdf_prefix} SELECT ?x WHERE {{ ?x rdf:subClassOf {str_named_concept}. }} """
        for str_iri in self.query(query):
            yield OWLClass(str_iri)

    def get_type_individuals(self, individual: str):
        query = f"""SELECT DISTINCT ?x WHERE {{ <{individual}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x }}"""
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLClass(binding["x"]["value"])

    def instances(self, expression: OWLClassExpression) -> Generator[OWLNamedIndividual, None, None]:
        assert isinstance(expression, OWLClassExpression)
        try:
            sparql_query = owl_expression_to_sparql(expression=expression)
        except Exception as exc:
            print(f"Error at converting {expression} into sparql")
            traceback.print_exception(exc)
            print(f"Error at converting {expression} into sparql")
            raise RuntimeError("Couldn't convert")
        try:
            for binding in self.query(sparql_query).json()["results"]["bindings"]:
                yield OWLNamedIndividual(binding["x"]["value"])
        except:
            print(self.query(sparql_query).text)
            raise RuntimeError

    def individuals_in_signature(self) -> Generator[OWLNamedIndividual, None, None]:
        # owl:OWLNamedIndividual is often missing: Perhaps we should add union as well
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a ?y. ?y a owl:Class.}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLNamedIndividual(binding["x"]["value"])

    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x " + "WHERE {?x a owl:DatatypeProperty.}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLDataProperty(binding["x"]["value"])

    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x " + "WHERE {?x a owl:ObjectProperty.}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLObjectProperty(binding["x"]["value"])

    def boolean_data_properties(self):
        query = f"{rdf_prefix}\n{rdfs_prefix}\n{xsd_prefix}SELECT DISTINCT ?x WHERE {{?x rdfs:range xsd:boolean}}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLDataProperty(binding["x"]["value"])

    def double_data_properties(self):
        query = f"{rdf_prefix}\n{rdfs_prefix}\n{xsd_prefix}SELECT DISTINCT ?x WHERE {{?x rdfs:range xsd:double}}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLDataProperty(binding["x"]["value"])

    def range_of_double_data_properties(self, prop: OWLDataProperty):
        query = f"{rdf_prefix}\n{rdfs_prefix}\n{xsd_prefix}SELECT DISTINCT ?x WHERE {{?z <{prop.str}> ?x}}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLLiteral(value=float(binding["x"]["value"]))

    def domain_of_double_data_properties(self, prop: OWLDataProperty):
        query = f"{rdf_prefix}\n{rdfs_prefix}\n{xsd_prefix}SELECT DISTINCT ?x WHERE {{?x <{prop.str}> ?z}}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLNamedIndividual(binding["x"]["value"])


class TripleStore:
    """ Connecting a triple store"""
    url: str

    def __init__(self, reasoner=None, url: str = None):

        if reasoner is None:
            assert url is not None, f"Reasoner:{reasoner} and url of a triplestore {url} cannot be both None."
            self.g = TripleStoreReasonerOntology(url=url)
        else:
            self.g = reasoner
        # This assigment is done as many CEL models are implemented to use both attributes seperately.
        # CEL models will be refactored.
        self.ontology = self.g
        self.reasoner = self.g

    def __abox_expression(self, individual: OWLNamedIndividual) -> Generator[
        Union[OWLClass, OWLObjectSomeValuesFrom, OWLObjectMinCardinality, OWLDataSomeValuesFrom], None, None]:
        """
        Return OWL Class Expressions obtained from all set of triples where an input OWLNamedIndividual is subject.

        Retrieve all triples (i,p,o) where p \in Resources, and o \in [Resources, Literals] and return the followings
        1- Owl Named Classes: C(i)=1.
        2- ObjectSomeValuesFrom Nominals: \exists r. {a, b, ..., d}, e.g. (i r, a) exists.
        3- OWLObjectSomeValuesFrom over named classes: \exists r. C  s.t. x \in {a, b, ..., d} C(x)=1.
        4- OWLObjectMinCardinality over named classes: ≥ c  r. C
        5- OWLDataSomeValuesFrom over literals: \exists r. {literal_a, ..., literal_b}
        """

        object_property_to_individuals = dict()
        data_property_to_individuals = dict()
        # To no return duplicate objects.
        quantifier_gate = set()
        # (1) Iterate over triples where individual is in the subject position.
        for s, p, o in self.g.abox(str_iri=individual.str):
            if isinstance(p, OWLProperty) and isinstance(o, OWLClass):
                ##############################################################
                # RETURN OWLClass
                ##############################################################
                yield o
            elif isinstance(p, OWLObjectProperty) and isinstance(o, OWLNamedIndividual):
                ##############################################################
                # Store for \exist r. {i, ..., j} and OWLObjectMinCardinality over type counts
                ##############################################################
                object_property_to_individuals.setdefault(p, []).append(o)
            elif isinstance(p, OWLDataProperty) and isinstance(o, OWLLiteral):
                ##############################################################
                # Store for  \exist r. {literal, ..., another literal}
                ##############################################################
                data_property_to_individuals.setdefault(p, []).append(o)
            else:
                raise RuntimeError(f"Unrecognized triples to expression mappings {p}{o}")
        # Iterating over the mappings of object properties to individuals.
        for object_property, list_owl_individuals in object_property_to_individuals.items():
            # RETURN: \exists r. {x1,x33, .., x8} => Existential restriction over nominals
            yield OWLObjectSomeValuesFrom(property=object_property, filler=OWLObjectOneOf(list_owl_individuals))
            owl_class: OWLClass
            count: int
            for owl_class, count in Counter(
                    [type_i for i in list_owl_individuals for type_i in
                     self.get_types(ind=i, direct=True)]).items():
                existential_quantifier = OWLObjectSomeValuesFrom(property=object_property, filler=owl_class)

                if existential_quantifier in quantifier_gate:
                    "Do nothing"
                else:
                    ##############################################################
                    # RETURN: \exists r. C => Existential quantifiers over Named OWL Class
                    ##############################################################
                    quantifier_gate.add(existential_quantifier)
                    yield existential_quantifier

                object_min_cardinality = OWLObjectMinCardinality(cardinality=count,
                                                                 property=object_property,
                                                                 filler=owl_class)

                if object_min_cardinality in quantifier_gate:
                    "Do nothing"
                else:
                    ##############################################################
                    # RETURN: ≥ c  r. C => OWLObjectMinCardinality over Named OWL Class
                    ##############################################################
                    quantifier_gate.add(object_min_cardinality)
                    yield object_min_cardinality
        # Iterating over the mappings of data properties to individuals.
        for data_property, list_owl_literal in data_property_to_individuals.items():
            ##############################################################
            # RETURN: \exists r. {literal, ..., another literal} => Existential quantifiers over Named OWL Class
            ##############################################################
            # if list_owl_literal is {True, False) doesn't really make sense OWLDataSomeValuesFrom
            # Perhaps, if
            yield OWLDataSomeValuesFrom(property=data_property, filler=OWLDataOneOf(list_owl_literal))

    def abox(self, individual: OWLNamedIndividual, mode: str = "native"):
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
            yield from self.g.abox(str_iri=individual.str)
        elif mode == "expression":
            yield from self.__abox_expression(individual)
        elif mode == "axiom":
            raise NotImplementedError("Axioms should be checked.")

    def are_owl_concept_disjoint(self, c: OWLClass, cc: OWLClass) -> bool:
        assert isinstance(c, OWLClass) and isinstance(cc, OWLClass)
        return self.reasoner.are_owl_concept_disjoint(c, cc)

    def get_object_properties(self):
        yield from self.reasoner.object_properties_in_signature()

    def get_data_properties(self):
        yield from self.reasoner.data_properties_in_signature()

    def get_concepts(self) -> OWLClass:
        yield from self.reasoner.classes_in_signature()

    def get_classes_in_signature(self) -> OWLClass:
        yield from self.reasoner.classes_in_signature()

    def get_most_general_classes(self):
        yield from self.reasoner.most_general_classes()

    def get_boolean_data_properties(self):
        yield from self.reasoner.boolean_data_properties()

    def get_double_data_properties(self):
        yield from self.reasoner.double_data_properties()

    def get_range_of_double_data_properties(self, prop: OWLDataProperty):
        yield from self.reasoner.range_of_double_data_properties(prop)

    def individuals(self, concept: Optional[OWLClassExpression] = None) -> Generator[OWLNamedIndividual, None, None]:
        """Given an OWL class expression, retrieve all individuals belonging to it.
        Args:
            concept: Class expression of which to list individuals.
        Returns:
            Generator of individuals belonging to the given class.
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

    def classes_in_signature(self):
        yield from self.reasoner.classes_in_signature()

    def get_direct_parents(self, c: OWLClass):
        yield from self.reasoner.get_direct_parents(c)

    def most_general_named_concepts(self):
        yield from self.reasoner.most_general_named_concepts()

    def least_general_named_concepts(self):
        yield from self.reasoner.least_general_named_concepts()

    def query(self, sparql: str) -> rdflib.plugins.sparql.processor.SPARQLResult:
        yield from self.g.query(sparql_query=sparql)
