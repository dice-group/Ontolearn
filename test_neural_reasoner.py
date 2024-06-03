from ontolearn.triple_store import TripleStore, TripleStoreReasonerOntology
from dicee import KGE
from owlapy.class_expression import OWLClass, OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty, OWLProperty
from owlapy.owl_literal import OWLLiteral
from typing import Generator, Tuple
import re

from owlapy.class_expression import (
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectIntersectionOf,
    OWLClassExpression,
    OWLNothing,
    OWLThing,
    OWLNaryBooleanClassExpression,
    OWLObjectUnionOf,
    OWLClass,
    OWLObjectComplementOf,
    OWLObjectMaxCardinality,
    OWLObjectMinCardinality,
    OWLDataSomeValuesFrom,
    OWLDatatypeRestriction,
    OWLDataHasValue,
    OWLObjectExactCardinality,
    OWLObjectHasValue,
    OWLObjectOneOf,
)

# from concept learner test
from ontolearn.learners import TDL
from ontolearn.triple_store import TripleStore
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
import sys


sys.setrecursionlimit(10000)


class DemoNeuralReasoner:
    model: KGE
    default_confidence_threshold: float

    def __init__(self, KGE_path: str, default_confidence_threshold: float = 0.8):
        self.model = KGE(path=KGE_path)
        self.default_confidence_threshold = default_confidence_threshold

    def get_predictions(
        self,
        h: str = None,
        r: str = None,
        t: str = None,
        topk: int = 10,
        confidence_threshold: float = None,
    ):
        if h is not None:
            h = [h]
        if r is not None:
            r = [r]
        if t is not None:
            t = [t]

        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(h=h, r=r, t=t, topk=topk)
            for prediction in predictions:
                confidence = prediction[1]
                predicted_iri_str = prediction[0]
                if confidence >= confidence_threshold:
                    yield (predicted_iri_str, confidence)
        except Exception as e:
            print(f"Error: {e}")
            return

    def abox(self, str_iri: str) -> Generator[
        Tuple[
            Tuple[OWLNamedIndividual, OWLProperty, OWLClass],
            Tuple[OWLObjectProperty, OWLObjectProperty, OWLNamedIndividual],
            Tuple[OWLObjectProperty, OWLDataProperty, OWLLiteral],
        ],
        None,
        None,
    ]:
        subject_ = OWLNamedIndividual(str_iri)
        # for p == type
        for cl in self.get_type_individuals(str_iri):
            yield (
                subject_,
                OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                cl,
            )

        # for p == object property
        for op in self.object_properties_in_signature():
            for o in self.get_object_property_values(str_iri, op):
                yield (subject_, op, o)

        # for p == data property
        for dp in self.data_properties_in_signature():
            print("these data properties are in the signature: ", dp.str)
            for l in self.get_data_property_values(str_iri, dp):
                yield (subject_, dp, l)

    def classes_in_signature(
        self, confidence_threshold: float = None
    ) -> Generator[OWLClass, None, None]:
        for prediction in self.get_predictions(
            h=None,
            r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            t="http://www.w3.org/2002/07/owl#Class",
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_class = OWLClass(prediction[0])
                yield owl_class
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def get_type_individuals(
        self, individual: str, confidence_threshold: float = None
    ) -> Generator[OWLClass, None, None]:
        for prediction in self.get_predictions(
            h=individual,
            r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            t=None,
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_class = OWLClass(prediction[0])
                yield owl_class
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def instances(
        self, expression: OWLClassExpression, confidence_threshold: float = None
    ) -> Generator[OWLNamedIndividual, None, None]:
        if expression.is_owl_thing():
            yield from self.individuals_in_signature()
        if isinstance(expression, OWLClass):
            yield from self.get_individuals_of_class(
                owl_class=expression, confidence_threshold=confidence_threshold
            )

        # Handling intersection of class expressions
        elif isinstance(expression, OWLObjectIntersectionOf):
            # Get the class expressions
            operands = list(expression.operands())
            sets_of_individuals = [
                list(
                    self.instances(
                        expression=operand, confidence_threshold=confidence_threshold
                    )
                )
                for operand in operands
            ]
            if sets_of_individuals:
                # Get the intersection of the sets
                common_individuals = set(sets_of_individuals[0])
                for individuals in sets_of_individuals[1:]:
                    common_individuals = common_individuals.intersection_update(
                        individuals
                    )
                for individual in common_individuals:
                    yield individual

        # Handling union of class expressions
        elif isinstance(expression, OWLObjectUnionOf):
            # Get the class expressions
            operands = list(expression.operands())
            seen = set()
            for operand in operands:
                for individual in self.instances(operand, confidence_threshold):
                    if individual not in seen:
                        seen.add(individual)
                        yield individual

        # Handling complement of class expressions
        elif isinstance(expression, OWLObjectComplementOf):
            # This case is tricky because it needs the complement within a specific domain
            # It's generally non-trivial to implement without knowing the domain of discourse
            all_individuals = list(
                self.individuals_in_signature()
            )  # Assume this retrieves all individuals
            excluded_individuals = set(
                self.instances(expression.get_operand(), confidence_threshold)
            )
            for individual in all_individuals:
                if individual not in excluded_individuals:
                    yield individual

    def individuals_in_signature(self) -> Generator[OWLNamedIndividual, None, None]:
        for cl in self.classes_in_signature():
            for prediction in self.get_predictions(
                h=None,
                r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                t=cl.str,
                confidence_threshold=self.default_confidence_threshold,
            ):
                try:
                    owl_named_individual = OWLNamedIndividual(prediction[0])
                    yield owl_named_individual
                except Exception as e:
                    # Log the invalid IRI
                    print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                    continue

    def data_properties_in_signature(
        self, confidence_threshold: float = None
    ) -> Generator[OWLDataProperty, None, None]:
        for prediction in self.get_predictions(
            h=None,
            r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            t="http://www.w3.org/2002/07/owl#DatatypeProperty",
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_data_property = OWLDataProperty(prediction[0])
                yield owl_data_property
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def object_properties_in_signature(
        self, confidence_threshold: float = None
    ) -> Generator[OWLObjectProperty, None, None]:
        for prediction in self.get_predictions(
            h=None,
            r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            t="http://www.w3.org/2002/07/owl#ObjectProperty",
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_object_property = OWLObjectProperty(prediction[0])
                yield owl_object_property
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def boolean_data_properties(
        self, confidence_threshold: float = None
    ) -> Generator[OWLDataProperty, None, None]:
        for prediction in self.get_predictions(
            h=None,
            r="http://www.w3.org/2000/01/rdf-schema#range",
            t="http://www.w3.org/2001/XMLSchema#boolean",
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_data_property = OWLDataProperty(prediction[0])
                yield owl_data_property
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def double_data_properties(
        self, confidence_threshold: float = None
    ) -> Generator[OWLDataProperty, None, None]:
        for prediction in self.get_predictions(
            h=None,
            r="http://www.w3.org/2000/01/rdf-schema#range",
            t="http://www.w3.org/2001/XMLSchema#double",
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_data_property = OWLDataProperty(prediction[0])
                yield owl_data_property
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    ### additional functions for neural reasoner

    def get_object_property_values(
        self,
        subject: str,
        object_property: OWLObjectProperty,
        confidence_threshold: float = None,
    ) -> Generator[OWLNamedIndividual, None, None]:
        for prediction in self.get_predictions(
            h=subject,
            r=object_property.str,
            t=None,
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_named_individual = OWLNamedIndividual(prediction[0])
                yield owl_named_individual
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def get_data_property_values(
        self,
        subject: str,
        data_property: OWLDataProperty,
        confidence_threshold: float = None,
    ) -> Generator[OWLLiteral, None, None]:
        for prediction in self.get_predictions(
            h=subject,
            r=data_property.str,
            t=None,
            confidence_threshold=confidence_threshold,
        ):
            try:
                # TODO: check the datatype and convert it to the correct type
                # like in abox triplestore line 773ff

                # Extract the value from the IRI
                value = re.search(r"\"(.+?)\"", prediction[0]).group(1)
                owl_literal = OWLLiteral(value)
                yield owl_literal
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def get_individuals_of_class(
        self, owl_class: OWLClass, confidence_threshold: float = None
    ) -> Generator[OWLNamedIndividual, None, None]:
        for prediction in self.get_predictions(
            h=None,
            r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            t=owl_class.str,
            confidence_threshold=confidence_threshold,
        ):
            try:
                owl_named_individual = OWLNamedIndividual(prediction[0])
                yield owl_named_individual
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue


###testing


# classic reasoner
kb = TripleStore(url="http://localhost:3030/nctrer")

for dp in kb.g.data_properties_in_signature():
    print("data properties in signature: ", dp)

for booldp in kb.g.boolean_data_properties():
    print("boolean data properties: ", booldp)

for doubledp in kb.g.double_data_properties():
    print("double data properties: ", doubledp.str)


# neural reasoner
print("--- Neural Reasoner ---")
kb_neural = TripleStore(DemoNeuralReasoner("embeddings/Pykeen_QuatENctrerRun/"))
for tripel in kb_neural.g.abox("http://dl-learner.org/nctrer/bond887"):
    print(tripel)
