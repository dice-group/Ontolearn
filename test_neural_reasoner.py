from ontolearn.triple_store import TripleStore, TripleStoreReasonerOntology
from dicee import KGE
from owlapy.class_expression import OWLClass, OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty, OWLProperty
from owlapy.owl_literal import OWLLiteral
from typing import Generator, Tuple
import re

# from concept learner test
from ontolearn.learners import TDL
from ontolearn.triple_store import TripleStore
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl


class DemoNeuralReasoner:
    model: KGE
    default_confidence_threshold: float

    def __init__(self, KGE_path: str, default_confidence_threshold: float = 0.8):
        self.model = KGE(path=KGE_path)
        self.default_confidence_threshold = default_confidence_threshold

    def abox(self, str_iri: str) -> Generator[
        Tuple[
            Tuple[OWLNamedIndividual, OWLProperty, OWLClass],
            Tuple[OWLObjectProperty, OWLObjectProperty, OWLNamedIndividual],
            Tuple[OWLObjectProperty, OWLDataProperty, OWLLiteral],
        ],
        None,
        None,
    ]:
        # for p == type
        for cl in self.get_type_individuals(str_iri):
            yield (
                OWLNamedIndividual(str_iri),
                OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                cl,
            )

        # for p == object property
        for op in self.object_properties_in_signature():
            for o in self.get_object_property_values(str_iri, op):
                yield (OWLNamedIndividual(str_iri), op, o)

        # for p == data property
        for dp in self.data_properties_in_signature():
            for l in self.get_data_property_values(str_iri, dp):
                yield (OWLNamedIndividual(str_iri), dp, l)

    def classes_in_signature(
        self, confidence_threshold: float = None
    ) -> Generator[OWLClass, None, None]:
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(
                h=None,
                r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                t=["http://www.w3.org/2002/07/owl#Class"],
                topk=10,
            )
        except Exception as e:
            return
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_class = OWLClass(prediction[0])
                if confidence >= confidence_threshold:
                    yield owl_class
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    def get_type_individuals(
        self, individual: str, confidence_threshold: float = None
    ) -> Generator[OWLClass, None, None]:
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(
                h=[individual],
                r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                t=None,
                topk=10,
            )
        except Exception as e:
            return
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_class = OWLClass(prediction[0])
                if confidence >= confidence_threshold:
                    yield owl_class
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    def instances(
        self, owl_class: OWLClassExpression, confidence_threshold: float = None
    ) -> Generator[OWLNamedIndividual, None, None]:
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(
                h=None,
                r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                t=[owl_class.str],
                topk=10,
            )
        except Exception as e:
            return
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_named_individual = OWLNamedIndividual(predicted_iri_str)
                if confidence >= confidence_threshold:
                    print("confidence: ", confidence)
                    yield owl_named_individual
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    def individuals_in_signature(self) -> Generator[OWLNamedIndividual, None, None]:
        pass

    def data_properties_in_signature(
        self, confidence_threshold: float = None
    ) -> Generator[OWLDataProperty, None, None]:
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(
                h=None,
                r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                t=["http://www.w3.org/2002/07/owl#DatatypeProperty"],
                topk=10,
            )
        except Exception as e:
            return
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_data_property = OWLDataProperty(predicted_iri_str)

                if confidence >= confidence_threshold:
                    yield owl_data_property
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    def object_properties_in_signature(
        self, confidence_threshold: float = None
    ) -> Generator[OWLObjectProperty, None, None]:
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(
                h=None,
                r=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                t=["http://www.w3.org/2002/07/owl#ObjectProperty"],
                topk=10,
            )
        except Exception as e:
            return
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_object_property = OWLObjectProperty(predicted_iri_str)
                if confidence >= confidence_threshold:
                    yield owl_object_property
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    ### additional functions for neural reasoner

    def get_object_property_values(
        self,
        subject: str,
        object_property: OWLObjectProperty,
        confidence_threshold: float = None,
    ) -> Generator[OWLNamedIndividual, None, None]:
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(
                h=[subject],
                r=[object_property.str],
                t=None,
                topk=10,
            )
        except Exception as e:
            return
        for prediction in predictions:
            confidence = prediction[1]
            predicted_iri_str = prediction[0]
            try:
                owl_named_individual = OWLNamedIndividual(predicted_iri_str)
                if confidence >= confidence_threshold:
                    yield owl_named_individual
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {predicted_iri_str}, error: {e}")
                continue

    def get_data_property_values(
        self,
        subject: str,
        data_property: OWLDataProperty,
        confidence_threshold: float = None,
    ) -> Generator[OWLLiteral, None, None]:
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        try:
            predictions = self.model.predict_topk(
                h=[subject],
                r=[data_property.str],
                t=None,
                topk=10,
            )
        except Exception as e:
            return
        for prediction in predictions:
            confidence = prediction[1]
            predicted_literal_str = prediction[0]
            try:
                # TODO: check the datatype and convert it to the correct type
                # like in abox triplestore line 773ff
                owl_literal = OWLLiteral(predicted_literal_str)
                if confidence_threshold >= confidence_threshold:
                    yield owl_literal
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid literal detected: {predicted_literal_str}, error: {e}")
                continue


###testing

# (1) Initialize Triplestore
# sudo docker run -p 3030:3030 -e ADMIN_PASSWORD=pw123 stain/jena-fuseki
# Login http://localhost:3030/#/ with admin and pw123
# Create a new dataset called family and upload KGs/Family/family.owl
# kb = TripleStore(DemoNeuralReasoner(KGE_path="./Pykeen_QuatEFamilyRun"))
kb = TripleStore(url="http://localhost:3030/family")
# (2) Initialize a learner.
model = TDL(knowledge_base=kb)
# (3) Define a description logic concept learning problem.
lp = PosNegLPStandard(
    pos={
        OWLNamedIndividual("http://www.benchmark.org/family#F5M64"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6M92"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9M157"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6M71"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2M11"),
        OWLNamedIndividual("http://www.benchmark.org/family#F3M45"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7M123"),
    },
    neg={
        OWLNamedIndividual("http://www.benchmark.org/family#F5M66"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10M188"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10M173"),
        OWLNamedIndividual("http://www.benchmark.org/family#F1F3"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2M37"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7M115"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F84"),
    },
)
# (4) Learn description logic concepts best fitting (3).
h = model.fit(learning_problem=lp).best_hypotheses()
print(h)
print(owl_expression_to_dl(h))
print(owl_expression_to_sparql(expression=h))


# (1) Initialize Triplestore
# sudo docker run -p 3030:3030 -e ADMIN_PASSWORD=pw123 stain/jena-fuseki
# Login http://localhost:3030/#/ with admin and pw123
# Create a new dataset called family and upload KGs/Family/family.owl
kb = TripleStore(DemoNeuralReasoner(KGE_path="./Pykeen_QuatEFatherRun"))
# (2) Initialize a learner.
model = TDL(knowledge_base=kb)
# (3) Define a description logic concept learning problem.
lp = PosNegLPStandard(
    pos={OWLNamedIndividual("http://example.com/father#stefan")},
    neg={
        OWLNamedIndividual("http://example.com/father#heinz"),
        OWLNamedIndividual("http://example.com/father#anna"),
        OWLNamedIndividual("http://example.com/father#michelle"),
    },
)
# (4) Learn description logic concepts best fitting (3).
h = model.fit(learning_problem=lp).best_hypotheses()
print(h)
print(owl_expression_to_dl(h))
print(owl_expression_to_sparql(expression=h))

print("------------------- Classic Reasoner -------------------")

kb_classic = TripleStore(url="http://localhost:3030/father")

for t in kb_classic.g.get_type_individuals("http://example.com/father#anna"):
    print(t)

print("------------------- instances function -------------------")
for i in kb_classic.g.instances(OWLClass("http://example.com/father#female")):
    print(i)


print("+++++++++++++++++++ Neural Reasoner +++++++++++++++++++")

kb_neural = TripleStore(DemoNeuralReasoner(KGE_path="./Pykeen_QuatEFatherRun"))
for t in kb_neural.g.get_type_individuals("http://example.com/father#anna"):
    print(t)

print("+++++++++++++++++++ instances function +++++++++++++++++++")
for i in kb_neural.g.instances(OWLClass("http://example.com/father#female")):
    print(i)

print("+++++++++++++++++++ classes_in_signature function +++++++++++++++++++")
for c in kb_neural.g.classes_in_signature():
    print(c)

print("+++++++++++++++++++ data_properties_in_signature function +++++++++++++++++++")
for dp in kb_neural.g.data_properties_in_signature():
    print(dp)

print("+++++++++++++++++++ abox function +++++++++++++++++++")
for a in kb_neural.g.abox("http://example.com/father#anna"):
    print(a)


print("+++++++++++++++++++ object_properties_in_signature function +++++++++++++++++++")
for op in kb_neural.g.object_properties_in_signature():
    print(op)
