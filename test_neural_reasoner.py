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
        self, owl_class: OWLClassExpression, confidence_threshold: float = None
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

    def individuals_in_signature(self) -> Generator[OWLNamedIndividual, None, None]:
        pass

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


###testing

# (1) Initialize Triplestore
# sudo docker run -p 3030:3030 -e ADMIN_PASSWORD=pw123 stain/jena-fuseki
# Login http://localhost:3030/#/ with admin and pw123
# Create a new dataset called family and upload KGs/Family/family.owl

# kb = TripleStore(url="http://localhost:3030/family")
kb = TripleStore(DemoNeuralReasoner(KGE_path="./embeddings/Pykeen_QuatEFamilyRun"))
for dp in kb.g.data_properties_in_signature():
    print(dp)

assert True == False
# (2) Initialize a learner.
model = TDL(knowledge_base=kb)
# (3) Define a description logic concept learning problem.
lp = PosNegLPStandard(
    pos={
        OWLNamedIndividual("http://www.benchmark.org/family#F2F14"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F12"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F19"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F26"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F28"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F36"),
        OWLNamedIndividual("http://www.benchmark.org/family#F3F52"),
        OWLNamedIndividual("http://www.benchmark.org/family#F3F53"),
        OWLNamedIndividual("http://www.benchmark.org/family#F5F62"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F72"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F79"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F77"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F86"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F91"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F84"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F96"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F101"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F93"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F114"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F106"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F116"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F119"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F126"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F121"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F148"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F150"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F143"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F152"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F154"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F141"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F160"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F163"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F158"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F168"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F174"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F179"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F181"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F192"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F193"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F186"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F195"),
    },
    neg={
        OWLNamedIndividual("http://www.benchmark.org/family#F6M99"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F200"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F156"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6M69"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F15"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6M100"),
        OWLNamedIndividual("http://www.benchmark.org/family#F8F133"),
        OWLNamedIndividual("http://www.benchmark.org/family#F3F48"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F30"),
        OWLNamedIndividual("http://www.benchmark.org/family#F4F55"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F74"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10M199"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7M104"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9M146"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6M71"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F22"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2M13"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F169"),
        OWLNamedIndividual("http://www.benchmark.org/family#F5F65"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6M81"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7M131"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F129"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7M107"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10F189"),
        OWLNamedIndividual("http://www.benchmark.org/family#F8F135"),
        OWLNamedIndividual("http://www.benchmark.org/family#F8M136"),
        OWLNamedIndividual("http://www.benchmark.org/family#F10M188"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F164"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F118"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2F10"),
        OWLNamedIndividual("http://www.benchmark.org/family#F6F97"),
        OWLNamedIndividual("http://www.benchmark.org/family#F7F111"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9M151"),
        OWLNamedIndividual("http://www.benchmark.org/family#F4M59"),
        OWLNamedIndividual("http://www.benchmark.org/family#F2M37"),
        OWLNamedIndividual("http://www.benchmark.org/family#F1M1"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9M142"),
        OWLNamedIndividual("http://www.benchmark.org/family#F4M57"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9M170"),
        OWLNamedIndividual("http://www.benchmark.org/family#F5M66"),
        OWLNamedIndividual("http://www.benchmark.org/family#F9F145"),
    },
)
# (4) Learn description logic concepts best fitting (3).
h = model.fit(learning_problem=lp).best_hypotheses()
print(h)
print(owl_expression_to_dl(h))
print(owl_expression_to_sparql(expression=h))
