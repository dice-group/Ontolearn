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

"""Triple store representations."""

import logging
import re
from itertools import chain
from typing import Iterable, Set, Optional, Generator, Union, Tuple, Callable, FrozenSet
import requests

from owlapy.class_expression import *
from owlapy.class_expression import OWLThing
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLObjectPropertyRangeAxiom,
    OWLObjectPropertyDomainAxiom,
    OWLDataPropertyRangeAxiom,
    OWLDataPropertyDomainAxiom,
    OWLClassAxiom,
    OWLEquivalentClassesAxiom, OWLAxiom,
)
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral, BooleanOWLDatatype, DoubleOWLDatatype, NUMERIC_DATATYPES, TIME_DATATYPES
from owlapy.owl_ontology import OWLOntologyID
from owlapy.abstracts import AbstractOWLOntology, AbstractOWLReasoner
from owlapy.owl_property import (
    OWLDataProperty,
    OWLObjectPropertyExpression,
    OWLObjectInverseOf,
    OWLObjectProperty,
    OWLProperty, OWLDataPropertyExpression,
)
from requests import Response
from requests.exceptions import RequestException, JSONDecodeError
from owlapy.converter import Owl2SparqlConverter

from ontolearn.abstracts import AbstractKnowledgeBase
# import traceback
from collections import Counter

logger = logging.getLogger(__name__)

rdfs_prefix = "PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n "
owl_prefix = "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n "
rdf_prefix = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n "
xsd_prefix = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"


def is_valid_url(url) -> bool:
    """
    Check the validity of a URL.

    Args:
        url (str): The url to validate.

    Returns:
        True if url is not None, and it passes the regex check.

    """
    regex = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return url is not None and regex.search(url)


def peek(generator):
    """Peeks the generator and returns the first element and the generator. Used to check whether the generator is
    empty by checking if the first element is None.

    Note: This is more efficiently than converting the generator to set and checking the len()"""
    try:
        first = next(generator)
    except StopIteration:
        return None
    return first, chain([first], generator)


def send_http_request_to_ts_and_fetch_results(triplestore_address: str, query: str, return_type: Callable):
    """
    Execute the SPARQL query in the given triplestore_address and return the result as the given return_type.

    Args:
        triplestore_address (str): The triplestore address where the query will be executed.
        query (str): SPARQL query where the root variable should be '?x'.
        return_type (Callable): OWLAPY class as type. e.g. OWLClass, OWLNamedIndividual, etc.

    Returns:
        Generator containing the results of the query as the given type.
    """
    try:
        response = requests.post(triplestore_address, data={"query": query})
    except RequestException as e:
        raise RequestException(
            f"Make sure the server is running on the `triplestore_address` = '{triplestore_address}'"
            f". Check the error below:"
            f"\n  -->Error: {e}"
        )
    try:
        if return_type == OWLLiteral:
            yield from unwrap(response)
        else:
            yield from [return_type(i) for i in unwrap(response) if i is not None]
        # return [return_type(IRI.create(i['x']['value'])) for i in
        #         response.json()['results']['bindings']]
    except JSONDecodeError as e:
        raise JSONDecodeError(
            f"Something went wrong with decoding JSON from the response. Check for typos in "
            f"the `triplestore_address` = '{triplestore_address}' otherwise the error is likely "
            f"caused by an internal issue. \n  -->Error: {e}"
        )


def unwrap(result: Response):
    json = result.json()
    vars_ = list(json["head"]["vars"])
    for b in json["results"]["bindings"]:
        val = []
        for v in vars_:
            if b[v]["type"] == "uri":
                val.append(IRI.create(b[v]["value"]))
            elif b[v]["type"] == "bnode":
                continue
            elif b[v]["type"] == "literal" and "datatype" in b[v]:
                val.append(OWLLiteral(b[v]["value"], OWLDatatype(IRI.create(b[v]["datatype"]))))
            elif b[v]["type"] == "literal" and "datatype" not in b[v]:
                continue

            elif b[v]["type"] == "literal" and "datatype" in b[v]:
                val.append(OWLLiteral(b[v]["value"], OWLDatatype(IRI.create(b[v]["datatype"]))))
            elif b[v]["type"] == "literal" and "datatype" not in b[v]:
                continue
            else:
                raise NotImplementedError(f"Seems like this kind of data is not handled: {b[v]}")

        if len(val) == 1:
            yield val.pop()
        else:
            yield None


def suf(direct: bool):
    """Put the star for rdfs properties depending on direct param"""
    return " " if direct else "* "


class TripleStoreOntology(AbstractOWLOntology):

    def __init__(self, triplestore_address: str):
        assert is_valid_url(triplestore_address), (
            "You should specify a valid URL in the following argument: "
            "'triplestore_address' of class `TripleStore`")
        self.url = triplestore_address

    def classes_in_signature(self) -> Iterable[OWLClass]:
        query = owl_prefix + "SELECT DISTINCT ?x WHERE {?x a owl:Class.}"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass)

    def data_properties_in_signature(self) -> Iterable[OWLDataProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:DatatypeProperty.}"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLDataProperty)

    def object_properties_in_signature(self) -> Iterable[OWLObjectProperty]:
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:ObjectProperty.}"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)

    # def individuals_in_signature(self) -> Generator[OWLNamedIndividual, None, None]:
    # TODO AB: <<TO BE DECIDED>> this or the implementation down below
    # TODO AB: owl:Class is not an individual!?
    # TODO AB: Why the return type is Generator[OWLNamedIndividual, None, None]? It does not adhere to
    #          individuals_in_signature method of AbstractOWLOntology.

    #     # owl:OWLNamedIndividual is often missing: Perhaps we should add union as well
    #     query = (
    #             owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a ?y. ?y a owl:Class.}"
    #     )
    #     for binding in self.query(query).json()["results"]["bindings"]:
    #         yield OWLNamedIndividual(binding["x"]["value"])

    def individuals_in_signature(self) -> Iterable[OWLNamedIndividual]:
        # TODO AB: Maybe extend this method to check for implicit individuals (idea: check for ?x a owl:Thing and
        #          exclude everything that is not a class, property, etc.)
        query = owl_prefix + "SELECT DISTINCT ?x\n " + "WHERE {?x a owl:NamedIndividual.}"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLNamedIndividual)

    def equivalent_classes_axioms(self, c: OWLClass) -> Iterable[OWLEquivalentClassesAxiom]:
        query = (owl_prefix + "SELECT DISTINCT ?x" + "WHERE { ?x owl:equivalentClass " + f"<{c.str}>."
                 + "FILTER(?x != " + f"<{c.str}>)}}")
        for cls in send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass):
            yield OWLEquivalentClassesAxiom([c, cls])

    def general_class_axioms(self) -> Iterable[OWLClassAxiom]:
        # doc strings inherited from abstract method in base class
        raise NotImplementedError("Currently, ")

    def data_property_domain_axioms(self, pe: OWLDataProperty) -> Iterable[OWLDataPropertyDomainAxiom]:
        first_element, domains = peek(self.get_property_domains(pe))
        if first_element is None:
            yield OWLDataPropertyDomainAxiom(pe, OWLThing)
        else:
            for dom in domains:
                yield OWLDataPropertyDomainAxiom(pe, dom)

    def data_property_range_axioms(self, pe: OWLDataProperty) -> Iterable[OWLDataPropertyRangeAxiom]:
        query = f"{rdfs_prefix}SELECT DISTINCT ?x WHERE {{ <{pe.str}> rdfs:range ?x. }}"
        for rng in send_http_request_to_ts_and_fetch_results(self.url, query, OWLDatatype):
            yield OWLDataPropertyRangeAxiom(pe, rng)

    def object_property_domain_axioms(
            self, pe: OWLObjectProperty
    ) -> Iterable[OWLObjectPropertyDomainAxiom]:
        first_element, domains = peek(self.get_property_domains(pe))
        if first_element is None:
            yield OWLObjectPropertyDomainAxiom(pe, OWLThing)
        else:
            for dom in domains:
                yield OWLObjectPropertyDomainAxiom(pe, dom)

    def object_property_range_axioms(self, pe: OWLObjectProperty) -> Iterable[OWLObjectPropertyRangeAxiom]:
        query = rdfs_prefix + "SELECT ?x WHERE { " + f"<{pe.str}>" + " rdfs:range ?x. }"
        first_element, ranges = peek(send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass))
        if first_element is None:
            yield OWLObjectPropertyRangeAxiom(pe, OWLThing)
        else:
            for rng in ranges:
                yield OWLObjectPropertyRangeAxiom(pe, rng)

    def get_property_domains(self, pe: OWLProperty) -> Set:
        if isinstance(pe, OWLObjectProperty) or isinstance(pe, OWLDataProperty):
            query = rdfs_prefix + "SELECT ?x WHERE { " + f"<{pe.str}>" + " rdfs:domain ?x. }"
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass)
        else:
            raise NotImplementedError

    def get_ontology_id(self) -> OWLOntologyID:
        # doc strings inherited from abstract method in base class

        # query = (rdf_prefix + owl_prefix +
        #          "SELECT ?ontologyIRI WHERE { ?ontology rdf:type owl:Ontology . ?ontology rdf:about ?ontologyIRI .}")
        # return list(get_results_from_ts(self.url, query, OWLOntologyID)).pop()
        raise NotImplementedError

    def add_axiom(self, axiom: Union[OWLAxiom, Iterable[OWLAxiom]]):
        """Cant modify a triplestore ontology. Implemented because of the base class."""
        pass

    def remove_axiom(self, axiom: Union[OWLAxiom, Iterable[OWLAxiom]]):
        """Cant modify a triplestore ontology. Implemented because of the base class."""
        pass

    def save(self, document_iri: Optional[IRI] = None):
        """Cant save a triplestore ontology. Implemented because of the base class."""
        pass

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.url == other.url
        return NotImplemented

    def __hash__(self):
        return hash(self.url)

    def __repr__(self):
        return f"TripleStoreOntology({self.url})"



class TripleStoreReasoner(AbstractOWLReasoner):

    def __init__(self, ontology: TripleStoreOntology):
        self.ontology = ontology
        self.url = self.ontology.url
        self._owl2sparql_converter = Owl2SparqlConverter()

    def query(self, sparql_query: str):
        return requests.Session().post(self.url, data={"query": sparql_query})

    def data_property_domains(self, pe: OWLDataProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        domains = {d.get_domain() for d in self.ontology.data_property_domain_axioms(pe)}
        sub_domains = set(chain.from_iterable([self.sub_classes(d) for d in domains]))
        yield from domains - sub_domains
        if not direct:
            yield from sub_domains

    def object_property_domains(self, pe: OWLObjectProperty, direct: bool = False) -> Iterable[OWLClassExpression]:
        domains = {
            d.get_domain() for d in self.ontology.object_property_domain_axioms(pe)
        }
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

    def data_property_ranges(self, pe: OWLDataProperty, direct: bool = True) -> Iterable[OWLClassExpression]:
        if direct:
            yield from [r.get_range() for r in self.ontology.data_property_range_axioms(pe)]
        else:
            # hierarchy of data types is not considered.
            return NotImplemented()

    def equivalent_classes(self, ce: OWLClassExpression, only_named: bool = True) -> Iterable[OWLClassExpression]:
        if only_named:
            if isinstance(ce, OWLClass):
                query = (owl_prefix + "SELECT DISTINCT ?x " + "WHERE { {?x owl:equivalentClass " + f"<{ce.str}>.}}"
                         + "UNION {" + f"<{ce.str}>" + " owl:equivalentClass ?x.}" + "FILTER(?x != " + f"<{ce.str}>)}}")
                yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass)
            else:
                logger.info(msg=f"Equivalent classes for complex class expressions is not implemented\t{ce}")
                # raise NotImplementedError(f"Equivalent classes for complex class expressions is not implemented\t{ce}")
                yield from {}
        else:
            raise NotImplementedError("Finding equivalent complex classes is not implemented")

    def disjoint_classes(self, ce: OWLClassExpression, only_named: bool = True) -> Iterable[OWLClassExpression]:
        if only_named:
            if isinstance(ce, OWLClass):
                query = owl_prefix + " SELECT DISTINCT ?x " + "WHERE { " + f"<{ce.str}>" + " owl:disjointWith ?x .}"
                yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass)
            else:
                raise NotImplementedError(
                    "Disjoint classes for complex class expressions is not implemented"
                )
        else:
            raise NotImplementedError(
                "Finding disjoint complex classes is not implemented"
            )

    def different_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        query = (owl_prefix + rdf_prefix + "SELECT DISTINCT ?x \n"
                 + "WHERE{ ?allDifferent owl:distinctMembers/rdf:rest*/rdf:first ?x.\n"
                 + "?allDifferent owl:distinctMembers/rdf:rest*/rdf:first" + f"<{ind.str}>" + ".\n"
                 + "FILTER(?x != " + f"<{ind.str}>" + ")}")
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLNamedIndividual)

    def same_individuals(self, ind: OWLNamedIndividual) -> Iterable[OWLNamedIndividual]:
        query = (owl_prefix + "SELECT DISTINCT ?x WHERE {{ ?x owl:sameAs " + f"<{ind.str}>" + " .}"
                 + "UNION { " + f"<{ind.str}>" + " owl:sameAs ?x.}}")
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLNamedIndividual)

    def equivalent_object_properties(self, op: OWLObjectPropertyExpression) -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (owl_prefix + "SELECT DISTINCT ?x " + "WHERE { {?x owl:equivalentProperty " + f"<{op.str}>.}}"
                     + "UNION {" + f"<{op.str}>" + " owl:equivalentProperty ?x.}" + "FILTER(?x != " + f"<{op.str}>)}}")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (owl_prefix + "SELECT DISTINCT ?x "
                     + "WHERE {  ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ."
                     + " {?x owl:equivalentProperty ?inverseProperty .}"
                     + "UNION { ?inverseProperty owl:equivalentClass ?x.} FILTER(?x != ?inverseProperty }>)}")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)

    def equivalent_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        query = (owl_prefix + "SELECT DISTINCT ?x" + "WHERE { {?x owl:equivalentProperty " + f"<{dp.str}>.}}"
                 + "UNION {" + f"<{dp.str}>" + " owl:equivalentProperty ?x.}" + "FILTER(?x != " + f"<{dp.str}>)}}")
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLDataProperty)

    def data_property_values(self, ind: OWLNamedIndividual, pe: OWLDataProperty, direct: bool = True) \
            -> Iterable[OWLLiteral]:
        query = "SELECT ?x WHERE { " + f"<{ind.str}> " + f"<{pe.str}>" + " ?x . }"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLLiteral)
        if not direct:
            for prop in self.sub_data_properties(pe):
                yield from self.data_property_values(ind, prop, True)

    def object_property_values(self, ind: OWLNamedIndividual, pe: OWLObjectPropertyExpression, direct: bool = True) \
            -> Iterable[OWLNamedIndividual]:
        if isinstance(pe, OWLObjectProperty):
            query = "SELECT ?x WHERE { " + f"<{ind.str}> " + f"<{pe.str}>" + " ?x . }"
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLNamedIndividual)
        elif isinstance(pe, OWLObjectInverseOf):
            query = (owl_prefix + "SELECT ?x WHERE { ?inverseProperty owl:inverseOf "
                     + f"<{pe.get_inverse().str}>." + f"<{ind.str}> ?inverseProperty ?x . }}")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLNamedIndividual)
        if not direct:
            for prop in self.sub_object_properties(pe):
                yield from self.object_property_values(ind, prop, True)

    def flush(self) -> None:
        pass

    # def instances(self, expression: OWLClassExpression, named_individuals: bool = False) \
    #         -> Generator[OWLNamedIndividual, None, None]:
    # TODO AB: <<TO BE DECIDED>> this or the implementation down below
    # TODO AB: Why the return type is Generator[OWLNamedIndividual, None, None]?
    #  It does not adhere to the return type of `instances` of AbstractOWLReasoner.

    #     assert isinstance(expression, OWLClassExpression)
    #     try:
    #         sparql_query = owl_expression_to_sparql(expression=expression,
    #                                                 named_individuals=named_individuals)
    #
    #     except Exception as exc:
    #         print(f"Error at converting {expression} into sparql")
    #         traceback.print_exception(exc)
    #         print(f"Error at converting {expression} into sparql")
    #         raise RuntimeError("Couldn't convert")
    #     try:
    #         # TODO:Be aware of the implicit inference of x being OWLNamedIndividual!
    #         for binding in self.query(sparql_query).json()["results"]["bindings"]:
    #             yield OWLNamedIndividual(binding["x"]["value"])
    #     except:
    #         print(self.query(sparql_query).text)
    #         raise RuntimeError

    def instances(self, ce: OWLClassExpression, direct: bool = False, seen_set: Set = None) \
            -> Iterable[OWLNamedIndividual]:
        if not seen_set:
            seen_set = set()
            seen_set.add(ce)
        ce_to_sparql = self._owl2sparql_converter.as_query("?x", ce)
        if not direct:
            ce_to_sparql = ce_to_sparql.replace(
                "?x a ",
                "?x a ?some_cls. \n ?some_cls <http://www.w3.org/2000/01/rdf-schema#subClassOf>* ",
            )
        yield from send_http_request_to_ts_and_fetch_results(self.url, ce_to_sparql, OWLNamedIndividual)
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
            query = rdfs_prefix + "SELECT ?x WHERE { ?x rdfs:subClassOf" + suf(direct) + f"<{ce.str}>" + ". }"
            results = list(send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass))
            if ce in results:
                # TODO AB: Should we remove ce?
                results.remove(ce)
            yield from results
        else:
            raise NotImplementedError(
                "Subclasses of complex classes retrieved via triple store is not implemented"
            )
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
            query = rdfs_prefix + "SELECT ?x WHERE { " + f"<{ce.str}>" + " rdfs:subClassOf" + suf(direct) + "?x. }"
            results = list(send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass))
            if ce in results:
                results.remove(ce)
            if (not direct and OWLThing not in results) or len(results) == 0:
                results.append(OWLThing)
            yield from results
        else:
            raise NotImplementedError(
                "Superclasses of complex classes retrieved via triple store is not "
                "implemented"
            )

    def disjoint_object_properties(self, op: OWLObjectPropertyExpression) -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (owl_prefix + rdf_prefix + "SELECT DISTINCT ?x \n"
                     + "WHERE{ ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n"
                     + "?AllDisjointProperties owl:members/rdf:rest*/rdf:first" + f"<{op.str}>" + ".\n"
                     + "FILTER(?x != " + f"<{op.str}>" + ")}")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (owl_prefix + " SELECT DISTINCT ?x "
                     + "WHERE {  ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ."
                     + " ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n"
                     + " ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?inverseProperty.\n"
                     + " FILTER(?x != ?inverseProperty)}")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)

    def disjoint_data_properties(self, dp: OWLDataProperty) -> Iterable[OWLDataProperty]:
        query = (owl_prefix + rdf_prefix + "SELECT DISTINCT ?x \n"
                 + "WHERE{ ?AllDisjointProperties owl:members/rdf:rest*/rdf:first ?x.\n"
                 + "?AllDisjointProperties owl:members/rdf:rest*/rdf:first" + f"<{dp.str}>" + ".\n"
                 + "FILTER(?x != " + f"<{dp.str}>" + ")}")
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLDataProperty)

    def all_data_property_values(self, pe: OWLDataProperty, direct: bool = True) -> Iterable[OWLLiteral]:
        query = "SELECT DISTINCT ?x WHERE { ?y" + f"<{pe.str}>" + " ?x . }"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLLiteral)
        if not direct:
            for prop in self.sub_data_properties(pe):
                yield from self.all_data_property_values(prop, True)

    def sub_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        query = rdfs_prefix + "SELECT ?x WHERE { ?x rdfs:subPropertyOf" + suf(direct) + f"<{dp.str}>" + ". }"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLDataProperty)

    def super_data_properties(self, dp: OWLDataProperty, direct: bool = False) -> Iterable[OWLDataProperty]:
        query = rdfs_prefix + "SELECT ?x WHERE {" + f"<{dp.str}>" + " rdfs:subPropertyOf" + suf(direct) + " ?x. }"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLDataProperty)

    def sub_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (rdfs_prefix + "SELECT ?x WHERE { ?x rdfs:subPropertyOf" + suf(direct) + f"<{op.str}> ." +
                     " FILTER(?x != " + f"<{op.str}>) }}")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (rdfs_prefix + "SELECT ?x "
                     + "WHERE { ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ."
                     + " ?x rdfs:subPropertyOf" + suf(direct) + " ?inverseProperty . }")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)

    def super_object_properties(self, op: OWLObjectPropertyExpression, direct: bool = False) \
            -> Iterable[OWLObjectPropertyExpression]:
        if isinstance(op, OWLObjectProperty):
            query = (rdfs_prefix + "SELECT ?x WHERE {" + f"<{op.str}>" + " rdfs:subPropertyOf" + suf(direct) + " ?x. "
                     + "FILTER(?x != " + f"<{op.str}>) }}")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)
        elif isinstance(op, OWLObjectInverseOf):
            query = (rdfs_prefix + "SELECT ?x "
                     + "WHERE { ?inverseProperty owl:inverseOf " + f"<{op.get_inverse().str}> ."
                     + " ?inverseProperty rdfs:subPropertyOf" + suf(direct) + "?x  . }")
            yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLObjectProperty)

    def types(self, ind: OWLNamedIndividual, direct: bool = False) -> Iterable[OWLClass]:
        if direct:
            query = "SELECT ?x WHERE {" + f"<{ind.str}> a" + " ?x. }"
        else:
            query = rdfs_prefix + "SELECT DISTINCT ?x WHERE {" + f"<{ind.str}> a ?cls. " " ?cls rdfs:subClassOf* ?x}"
        yield from send_http_request_to_ts_and_fetch_results(self.url, query, OWLClass)

    def get_root_ontology(self) -> AbstractOWLOntology:
        return self.ontology

    def is_isolated(self):
        # not needed here
        pass


class TripleStore(AbstractKnowledgeBase):

    def __init__(self, ontology=None, reasoner=None, url: str = None):

        self.url = url
        self.ontology = ontology
        self.reasoner = reasoner
        if url is None:
            if ontology is not None:
                self.url = ontology.url
            else:
                assert (reasoner is not None), "ontology or reasoner or url must be provided"
                self.url = reasoner.url

        if ontology is None:
            if reasoner is not None:
                self.ontology = reasoner.ontology
            else:
                self.ontology = TripleStoreOntology(url)

        if reasoner is None:
            self.reasoner = TripleStoreReasoner(self.ontology)

        assert self.url == self.ontology.url == self.reasoner.url, "URLs do not match"

    def __str__(self):
        return f"TripleStore:{self.ontology, self.reasoner, self.url}"

    def query(self, sparql_query: str):
        return requests.Session().post(self.url, data={"query": sparql_query})

    def _abox(self, str_iri: str) -> Generator[
        Tuple[
            Tuple[OWLNamedIndividual, OWLProperty, OWLClass],
            Tuple[OWLObjectProperty, OWLObjectProperty, OWLNamedIndividual],
            Tuple[OWLObjectProperty, OWLDataProperty, OWLLiteral],
        ],
        None,
        None,
    ]:
        """@TODO:"""
        sparql_query = f"SELECT DISTINCT ?p ?o WHERE {{ <{str_iri}> ?p ?o }}"
        subject_ = OWLNamedIndividual(str_iri)
        for binding in self.query(sparql_query).json()["results"]["bindings"]:
            p, o = binding["p"], binding["o"]
            # ORDER MATTERS
            if p["value"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                yield subject_, OWLProperty(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                ), OWLClass(o["value"])
            elif o["type"] == "uri":
                #################################################################
                # IMPORTANT
                # Can we assume that if o has URI and is not owl class, then o can be considered as an individual ?
                # RE AB: No, it can be everything identified by an IRI, for example a property.
                #################################################################
                yield subject_, OWLObjectProperty(p["value"]), OWLNamedIndividual(
                    o["value"]
                )
            elif o["type"] == "literal":
                if data_type := o.get("datatype", None):
                    if data_type == "http://www.w3.org/2001/XMLSchema#boolean":
                        yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=bool(o["value"]))
                    elif data_type == "http://www.w3.org/2001/XMLSchema#integer":
                        yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=int(o["value"]))
                    elif data_type == "http://www.w3.org/2001/XMLSchema#nonNegativeInteger":
                        # TODO AB: set type to NonNegativeInteger for OWLLiteral below
                        #       after integrating the new owlapy release (> 1.3.3)
                        yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=int(o["value"]))
                    elif data_type == "http://www.w3.org/2001/XMLSchema#double":
                        yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=float(o["value"]))
                    else:
                        # TODO: Unclear for the time being.
                        # print(f"Currently this type of literal is not supported:{o} but can done easily let us know :)")
                        continue
                    """
                    # TODO: Converting a SPARQL query becomes an issue with strings.
                    elif data_type == "http://www.w3.org/2001/XMLSchema#string":
                        yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=repr(o["value"]))
                    elif data_type == "http://www.w3.org/2001/XMLSchema#date":
                        yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=repr(o["value"])) 
                    """

                else:
                    # print(f"Currently this type of literal is not supported:{o} but can done easily let us know :)")
                    continue
                    # yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=repr(o["value"]))

            else:
                raise RuntimeError(f"Unrecognized type {subject_} ({p}) ({o})")

    def __abox_expression(self, individual: OWLNamedIndividual) -> Generator[
        Union[
            OWLClass,
            OWLObjectSomeValuesFrom,
            OWLObjectMinCardinality,
            OWLDataSomeValuesFrom,
        ],
        None,
        None,
    ]:
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
        for s, p, o in self._abox(str_iri=individual.str):
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
                raise RuntimeError(
                    f"Unrecognized triples to expression mappings {p}{o}"
                )
        # Iterating over the mappings of object properties to individuals.
        for (
                object_property,
                list_owl_individuals,
        ) in object_property_to_individuals.items():
            # RETURN: \exists r. {x1,x33, .., x8} => Existential restriction over nominals
            yield OWLObjectSomeValuesFrom(
                property=object_property, filler=OWLObjectOneOf(list_owl_individuals)
            )
            owl_class: OWLClass
            count: int
            for owl_class, count in Counter(
                    [
                        type_i
                        for i in list_owl_individuals
                        for type_i in self.get_types(ind=i, direct=True)
                    ]
            ).items():
                existential_quantifier = OWLObjectSomeValuesFrom(
                    property=object_property, filler=owl_class
                )

                if existential_quantifier in quantifier_gate:
                    "Do nothing"
                else:
                    ##############################################################
                    # RETURN: \exists r. C => Existential quantifiers over Named OWL Class
                    ##############################################################
                    quantifier_gate.add(existential_quantifier)
                    yield existential_quantifier

                object_min_cardinality = OWLObjectMinCardinality(
                    cardinality=count, property=object_property, filler=owl_class
                )

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
            yield OWLDataSomeValuesFrom(
                property=data_property, filler=OWLDataOneOf(list_owl_literal)
            )

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
        assert mode in [
            "native",
            "iri",
            "axiom",
            "expression",
        ], "Valid modes are: 'native', 'iri' or 'axiom', 'expression'"
        # TODO: AB: We should probably remove "native" mode because it does not make sense since abox method is supposed
        #           to return abox axioms and axioms in owlapy are represented by an object of type "OWLAxiom", in
        #           other words we should keep only the "axiom" mode. The user can get the entities from the axiom
        #           object if he wants to do any other operations with them.
        if mode == "native":
            yield from self._abox(str_iri=individual.str)
        elif mode == "expression":
            yield from self.__abox_expression(individual)
        elif mode == "axiom":
            raise NotImplementedError("Axioms should be checked.")

    def tbox(self, entities: Union[Iterable[OWLClass], Iterable[OWLDataProperty], Iterable[OWLObjectProperty], OWLClass,
             OWLDataProperty, OWLObjectProperty, None] = None, mode='native'):
        raise NotImplementedError()

    def triples(self, mode=None):
        raise NotImplementedError()

    def are_owl_concept_disjoint(self, c: OWLClass, cc: OWLClass) -> bool:
        if cc in self.reasoner.disjoint_classes(c):
            return True
        return False

    def get_object_properties(self) -> Iterable[OWLObjectProperty]:
        yield from self.ontology.object_properties_in_signature()

    def get_data_properties(self, ranges: Union[OWLDatatype, Iterable[OWLDatatype]] = None) \
            -> Iterable[OWLDataProperty]:
        if ranges is None:
            yield from self.ontology.data_properties_in_signature()
        else:
            def get_properties_from_xsd_range(r: OWLDatatype):
                query = (f"{rdf_prefix}\n{rdfs_prefix}\n{xsd_prefix}SELECT DISTINCT ?x " +
                         f"WHERE {{?x rdfs:range xsd:{r.iri.reminder}}}")
                for binding in self.query(query).json()["results"]["bindings"]:
                    yield OWLDataProperty(binding["x"]["value"])
            if isinstance(ranges, OWLDatatype):
                yield from get_properties_from_xsd_range(ranges)
            else:
                for rng in ranges:
                    yield from get_properties_from_xsd_range(rng)

    def get_concepts(self) -> Iterable[OWLClass]:
        yield from self.ontology.classes_in_signature()

    def get_boolean_data_properties(self) -> Iterable[OWLDataProperty]:
        yield from self.get_data_properties(BooleanOWLDatatype)

    def get_double_data_properties(self):
        yield from self.get_data_properties(DoubleOWLDatatype)

    def get_values_of_double_data_property(self, prop: OWLDataProperty):
        query = f"{rdf_prefix}\n{rdfs_prefix}\n{xsd_prefix}SELECT DISTINCT ?x WHERE {{?z <{prop.str}> ?x}}"
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLLiteral(value=float(binding["x"]["value"]))

    def individuals(self, concept: Optional[OWLClassExpression] = None, named_individuals: bool = False) \
            -> Iterable[OWLNamedIndividual]:
        """Given an OWL class expression, retrieve all individuals belonging to it.
        Args:
            concept: Class expression of which to list individuals.
            named_individuals: flag for returning only owl named individuals in the SPARQL mapping
        Returns:
            Generator of individuals belonging to the given class.
        """

        if concept is None or concept.is_owl_thing():
            yield from self.ontology.individuals_in_signature()
        else:
            # yield from self.reasoner.instances(concept, named_individuals=named_individuals)
            yield from self.reasoner.instances(concept)

    # def get_types(self, ind: OWLNamedIndividual, direct: True) -> Generator[OWLClass, None, None]:
    # TODO AB: <<TO BE DECIDED>> this or the implementation down below
    # TODO AB: Why the return type is Generator[OWLClass, None, None]? It does not adhere to get_types of KnowledgeBase

    #     if not direct:
    #         raise NotImplementedError("Inferring indirect types not available")
    #     query = f"""SELECT DISTINCT ?x WHERE {{ <{ind.str}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x }}"""
    #     for binding in self.query(query).json()["results"]["bindings"]:
    #         yield OWLClass(binding["x"]["value"])

    def get_types(self, ind: OWLNamedIndividual, direct: True) -> Iterable[OWLClass]:
        return self.reasoner.types(ind, direct)

    # def get_all_sub_concepts(self, concept: OWLClass, direct=True):
    # TODO AB: <<TO BE DECIDED>> this or the implementation down below
    # TODO AB: Why do we use 'rdf' and not 'rdfs' when direct=False?

    #     assert isinstance(concept, OWLClass)
    #     str_named_concept = f"<{concept.str}>"
    #     if direct:
    #         query = f"""{rdfs_prefix} SELECT ?x WHERE {{ ?x rdfs:subClassOf* {str_named_concept}. }} """
    #     else:
    #         query = f"""{rdf_prefix} SELECT ?x WHERE {{ ?x rdf:subClassOf {str_named_concept}. }} """
    #     for str_iri in self.query(query):
    #         yield OWLClass(str_iri)

    def get_all_sub_concepts(self, concept: OWLClassExpression, direct=False) -> Iterable[OWLClassExpression]:
        assert isinstance(concept, OWLClass)
        yield from self.reasoner.sub_classes(concept, direct)

    def classes_in_signature(self):
        yield from self.ontology.classes_in_signature()

    def get_direct_parents(self, concept: OWLClassExpression) -> Iterable[OWLClass]:
        assert isinstance(concept, OWLClass)
        yield from self.reasoner.super_classes(concept, direct=True)

    def get_direct_sub_concepts(self, concept: OWLClass) -> Iterable[OWLClass]:
        assert isinstance(concept, OWLClass)
        yield from self.reasoner.sub_classes(concept, direct=True)

    def get_all_direct_sub_concepts(self, concept: OWLClassExpression) -> Iterable[OWLClassExpression]:
        assert isinstance(concept, OWLClass)
        yield from self.reasoner.sub_classes(concept, direct=True)

    @property
    def concepts(self) -> Iterable[OWLClass]:
        yield from self.ontology.classes_in_signature()

    def contains_class(self, concept: OWLClassExpression) -> bool:
        assert isinstance(concept, OWLClass)
        return concept in self.ontology.classes_in_signature()

    @property
    def object_properties(self) -> Iterable[OWLObjectProperty]:
        yield from self.ontology.object_properties_in_signature()

    @property
    def data_properties(self) -> Iterable[OWLDataProperty]:
        yield from self.ontology.data_properties_in_signature()

    def individuals_count(self, concept: Optional[OWLClassExpression] = None) -> int:
        return len(set(self.individuals(concept)))

    def individuals_set(self,
                        arg: Union[Iterable[OWLNamedIndividual], OWLNamedIndividual, OWLClassExpression]) -> FrozenSet:
        if isinstance(arg, OWLClassExpression):
            return frozenset(self.individuals(arg))
        elif isinstance(arg, OWLNamedIndividual):
            return frozenset({arg})
        else:
            return frozenset(arg)

    def most_general_object_properties(
            self, *, domain: OWLClassExpression, inverse: bool = False) -> Iterable[OWLObjectProperty]:
        assert isinstance(domain, OWLClassExpression)
        # TODO AB: Implementation copied from KnowledgeBase but is unclear what this method actually does.
        func: Callable
        func = (self.get_object_property_ranges if inverse else self.get_object_property_domains)

        # TODO AB: <<REVIEW>> There is a contradiction in the implementation below because if domain is owl:thing then,
        #          the property is returned, meaning that the domain of the property is a subclass of the 'domain'
        #          argument. On the other side if set of individuals covered by the 'domain' argument is a subset
        #          of the set of individuals covered by the property's domain then the property is returned. That means
        #          that the 'domain' argument is a subclass of the property's domain, which contradict the first
        #          condition.
        inds_domain = self.individuals_set(domain)
        for prop in self.ontology.object_properties_in_signature():
            if domain.is_owl_thing() or inds_domain <= self.individuals_set(func(prop)):
                yield prop

    def data_properties_for_domain(self, domain: OWLClassExpression, data_properties: Iterable[OWLDataProperty]) \
            -> Iterable[OWLDataProperty]:
        # TODO AB: <<REVIEW>> Its unclear what this method is supposed to do but by the name I can say that it is
        #          supposed to return the data properties from the given collection of data properties that have the
        #          specified 'domain'. However old implementation is commented below and is similar to the one in
        #          method 'most_general_object_properties' which is contradicting.
        assert isinstance(domain, OWLClassExpression)
        sub_domains = self.reasoner.sub_classes(domain)
        for dp in data_properties:
            dp_domains = self.get_data_property_domains(dp)
            for d in dp_domains:
                if d == domain or d in sub_domains:
                    yield dp

        # inds_domain = self.individuals_set(domain)
        # for prop in data_properties:
        #     if domain.is_owl_thing() or inds_domain <= self.individuals_set(next(self.get_data_property_domains(prop))):
        #         yield prop

    def most_general_classes(self) -> Iterable[OWLClass]:
        """At least it has single subclass and there is no superclass"""
        query = f"""{rdf_prefix}{rdfs_prefix}{owl_prefix} SELECT ?x WHERE {{
        ?concept rdf:type owl:Class .
        FILTER EXISTS {{ ?x rdfs:subClassOf ?z . }}
        FILTER NOT EXISTS {{ ?y rdfs:subClassOf ?x . }}
        }}
        """
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLClass(binding["x"]["value"])

    def least_general_named_concepts(self) -> Generator[OWLClass, None, None]:
        """At least it has single superclass and there is no subclass"""
        query = f"""{rdf_prefix}{rdfs_prefix}{owl_prefix} SELECT ?concept WHERE {{
        ?concept rdf:type owl:Class .
        FILTER EXISTS {{ ?concept rdfs:subClassOf ?x . }}
        FILTER NOT EXISTS {{ ?y rdfs:subClassOf ?concept . }}
        }}"""
        for binding in self.query(query).json()["results"]["bindings"]:
            yield OWLClass(binding["concept"]["value"])

    def get_object_property_domains(self, prop: OWLObjectProperty, direct=True) -> Iterable[OWLClassExpression]:
        yield from self.reasoner.object_property_domains(prop, direct)

    def get_object_property_ranges(self, prop: OWLObjectProperty, direct=True) -> Iterable[OWLClassExpression]:
        yield from self.reasoner.object_property_ranges(prop, direct)

    def get_data_property_domains(self, prop: OWLDataProperty, direct=True) -> Iterable[OWLClassExpression]:
        yield from self.reasoner.data_property_domains(prop, direct)

    def get_data_property_ranges(self, prop: OWLDataProperty, direct=True) -> Iterable[OWLClassExpression]:
        yield from self.reasoner.data_property_ranges(prop, direct)

    def most_general_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        yield from self.data_properties_for_domain(domain, self.get_data_properties())

    def most_general_boolean_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        yield from self.data_properties_for_domain(domain, self.get_boolean_data_properties())

    def most_general_numeric_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        yield from self.data_properties_for_domain(domain, self.get_numeric_data_properties())

    def most_general_time_data_properties(self, *, domain: OWLClassExpression) -> Iterable[OWLDataProperty]:
        yield from self.data_properties_for_domain(domain, self.get_time_data_properties())

    def most_general_existential_restrictions(self, *,
                                              domain: OWLClassExpression, filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectSomeValuesFrom]:
        if filler is None:
            filler = OWLThing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain):
            yield OWLObjectSomeValuesFrom(property=prop, filler=filler)

    def most_general_universal_restrictions(self, *,
                                            domain: OWLClassExpression, filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectAllValuesFrom]:
        if filler is None:
            filler = OWLThing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain):
            yield OWLObjectAllValuesFrom(property=prop, filler=filler)

    def most_general_existential_restrictions_inverse(self, *,
                                                      domain: OWLClassExpression,
                                                      filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectSomeValuesFrom]:
        if filler is None:
            filler = OWLThing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain, inverse=True):
            yield OWLObjectSomeValuesFrom(property=prop.get_inverse_property(), filler=filler)

    def most_general_universal_restrictions_inverse(self, *,
                                                    domain: OWLClassExpression,
                                                    filler: Optional[OWLClassExpression] = None) \
            -> Iterable[OWLObjectAllValuesFrom]:
        if filler is None:
            filler = OWLThing
        assert isinstance(filler, OWLClassExpression)

        for prop in self.most_general_object_properties(domain=domain, inverse=True):
            yield OWLObjectAllValuesFrom(property=prop.get_inverse_property(), filler=filler)

    def get_numeric_data_properties(self) -> Iterable[OWLDataProperty]:
        yield from self.get_data_properties(NUMERIC_DATATYPES)

    def get_time_data_properties(self) -> Iterable[OWLDataProperty]:
        """Get all time data properties of this concept generator.

        Returns:
            Time data properties.
        """
        yield from self.get_data_properties(TIME_DATATYPES)

    def get_object_properties_for_ind(self, ind: OWLNamedIndividual, direct: bool = True) \
            -> Iterable[OWLObjectProperty]:
        properties = set(self.get_object_properties())
        yield from (pe for pe in self.reasoner.ind_object_properties(ind, direct) if pe in properties)

    def get_data_properties_for_ind(self, ind: OWLNamedIndividual, direct: bool = True) -> Iterable[OWLDataProperty]:
        properties = set(self.get_data_properties())
        yield from (pe for pe in self.reasoner.ind_data_properties(ind, direct) if pe in properties)

    def get_object_property_values(self, ind: OWLNamedIndividual,
                                   property_: OWLObjectPropertyExpression,
                                   direct: bool = True) -> Iterable[OWLNamedIndividual]:
        yield from self.reasoner.object_property_values(ind, property_, direct)

    def get_data_property_values(self, ind: OWLNamedIndividual,
                                 property_: OWLDataPropertyExpression,
                                 direct: bool = True) -> Iterable[OWLLiteral]:
        yield from self.reasoner.data_property_values(ind, property_, direct)
