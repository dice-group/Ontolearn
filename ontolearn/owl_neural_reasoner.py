from owlapy.owl_property import (
    OWLDataProperty,
    OWLObjectInverseOf,
    OWLObjectProperty,
    OWLProperty,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.class_expression import *
from typing import Generator, Tuple
from dicee.knowledge_graph_embeddings import KGE
import os
import re
from collections import Counter


# Neural Reasoner
class TripleStoreNeuralReasoner:
    """ OWL Neural Reasoner uses a neural link predictor to retrieve instances of an OWL Class Expression"""
    model: KGE
    gamma: float

    def __init__(self, path_of_kb: str = None,
                 path_neural_embedding: str = None, gamma: float = 0.25):

        if path_neural_embedding:  # pragma: no cover
            assert os.path.isdir(
                path_neural_embedding), f"The given path ({path_neural_embedding}) does not lead to a directory"
            self.model = KGE(path=path_neural_embedding)
        elif path_of_kb:
            assert os.path.isfile(path_of_kb), f"The given path ({path_of_kb}) does not lead to an RDF Knowledge Graph."
            # Check we have already a trained model for a given path of a knowledge base
            dir_of_potential_neural_embedding_model = path_of_kb.replace("/", "_").replace(".", "_")
            if os.path.isdir(dir_of_potential_neural_embedding_model):
                self.model = KGE(path=dir_of_potential_neural_embedding_model)
            else:  # pragma: no cover
                # Train a KGE on the fly
                from dicee.executer import Execute
                from dicee.config import Namespace
                args = Namespace()
                args.model = 'Keci'
                args.scoring_technique = "AllvsAll"
                args.path_single_kg = path_of_kb
                path_of_kb = path_of_kb.replace("/", "_")
                path_of_kb = path_of_kb.replace(".", "_")
                args.path_to_store_single_run = path_of_kb
                args.num_epochs = 100
                args.embedding_dim = 512
                args.batch_size = 1024
                args.backend = "rdflib"
                args.trainer = "PL"
                reports = Execute(args).start()
                path_neural_embedding = reports["path_experiment_folder"]
                self.model = KGE(path=path_neural_embedding)
        else:
            raise RuntimeError(
                f"path_neural_embedding {path_neural_embedding} and path_of_kb {path_of_kb} cannot be both None")

        self.gamma = gamma
        # Caching for the sake of memory usage.
        # TODO:CD: We may want to use  @functools.lru_cache(maxsize=?, typed=?)
        #  https://docs.python.org/3/library/functools.html
        self.inferred_owl_individuals = None
        self.inferred_object_properties = None
        self.inferred_named_owl_classes = None

    @property
    def set_inferred_individuals(self):
        if self.inferred_owl_individuals is None:
            # self.inferred_owl_individuals is filled in here
            return {i for i in self.individuals_in_signature()}
        else:
            return self.inferred_owl_individuals

    @property
    def set_inferred_object_properties(self):  # pragma: no cover
        if self.inferred_object_properties is None:
            # self.inferred_owl_individuals is filled in here
            return {i for i in self.object_properties_in_signature()}
        else:
            return self.inferred_object_properties

    @property
    def set_inferred_owl_classes(self):  # pragma: no cover
        if self.inferred_named_owl_classes is None:
            # self.inferred_owl_individuals is filled in here
            return {i for i in self.classes_in_signature()}
        else:
            return self.inferred_named_owl_classes

    def __str__(self):
        return f"TripleStoreNeuralReasoner:{self.model} with likelihood threshold gamma : {self.gamma}"

    def get_predictions(self, h: str = None, r: str = None, t: str = None, confidence_threshold: float = None,
                        ) -> Generator[Tuple[str, float], None, None]:
        """
        Generate predictions for a given head entity (h), relation (r), or tail entity (t) with an optional confidence threshold. The method yields predictions that exceed the specified confidence threshold. If no threshold is provided, it defaults to the model's gamma value.

        Parameters:
        - h (str, optional): The identifier for the head entity.
        - r (str, optional): The identifier for the relation.
        - t (str, optional): The identifier for the tail entity.
        - confidence_threshold (float, optional): The minimum confidence level required for a prediction to be returned.

        Returns:
        - Generator[Tuple[str, float], None, None]: A generator of tuples, where each tuple contains a predicted entity or relation identifier (IRI) as a string and its corresponding confidence level as a float.

        Raises:
        - Exception: If an error occurs during prediction.
        """
        # sanity check
        assert h is not None or r is not None or t is not None, "At least one of h, r, or t must be provided."
        assert confidence_threshold is None or 0 <= confidence_threshold <= 1, "Confidence threshold must be in the range [0, 1]."
        assert h is None or isinstance(h, str), "Head entity must be a string."
        assert r is None or isinstance(r, str), "Relation must be a string."
        assert t is None or isinstance(t, str), "Tail entity must be a string."

        if h is not None:
            if (self.model.entity_to_idx.get(h, None)) is None:
                return
            h = [h]

        if r is not None:
            if (self.model.relation_to_idx.get(r, None)) is None:
                return
            r = [r]
        if t is not None:
            if (self.model.entity_to_idx.get(t, None)) is None:
                return
            t = [t]

        if confidence_threshold is None:
            confidence_threshold = self.gamma

        if r is None:
            topk = len(self.model.relation_to_idx)
        else:
            topk = len(self.model.entity_to_idx)
        try:
            predictions = self.model.predict_topk(h=h, r=r, t=t, topk=topk)
            for prediction in predictions:
                confidence = prediction[1]
                predicted_iri_str = prediction[0]
                if confidence >= confidence_threshold:
                    yield predicted_iri_str, confidence
                else:
                    #todo: replace with return or break?
                    continue
        except Exception as e:  # pragma: no cover
            print(f"Error at getting predictions: {e}")

    def abox(self, str_iri: str) -> Generator[
        Tuple[
            Tuple[OWLNamedIndividual, OWLProperty, OWLClass],
            Tuple[OWLObjectProperty, OWLObjectProperty, OWLNamedIndividual],
            Tuple[OWLObjectProperty, OWLDataProperty, OWLLiteral]], None,None ]:
        # Initialize an owl named individual object.
        subject_ = OWLNamedIndividual(str_iri)
        # Return a triple indicating the type.
        for cl in self.get_type_individuals(str_iri):
            yield subject_,OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), cl

        # Return a triple based on an object property.
        for op in self.object_properties_in_signature():
            for o in self.get_object_property_values(str_iri, op):
                yield subject_, op, o

        # Return a triple based on a data property.
        for dp in self.data_properties_in_signature():  # pragma: no cover
            print("these data properties are in the signature: ", dp.str)
            for l in self.get_data_property_values(str_iri, dp):
                yield subject_, dp, l

    def classes_in_signature(
            self, confidence_threshold: float = None) -> Generator[OWLClass, None, None]:
        if self.inferred_named_owl_classes is None:
            self.inferred_named_owl_classes = set()
            for prediction in self.get_predictions(
                    h=None,
                    r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    t="http://www.w3.org/2002/07/owl#Class",
                    confidence_threshold=confidence_threshold):
                try:
                    owl_class = OWLClass(prediction[0])
                    self.inferred_named_owl_classes.add(owl_class)
                    yield owl_class
                except Exception as e:
                    # Log the invalid IRI
                    print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                    continue
        else:
            yield from self.inferred_named_owl_classes

    def most_general_classes(
            self, confidence_threshold: float = None
    ) -> Generator[OWLClass, None, None]:  # pragma: no cover
        """At least it has single subclass and there is no superclass"""

        for c in self.classes_in_signature(confidence_threshold):
            for x in self.get_direct_parents(c, confidence_threshold):
                # Ignore c if (c subclass x) \in KG.
                break
            else:
                # checks if subconcepts is not empty -> there is at least one subclass
                # c should have at least a single subclass.
                if subconcepts := list(
                        self.subconcepts(
                            named_concept=c, confidence_threshold=confidence_threshold
                        )
                ):
                    yield c

    def least_general_named_concepts(
            self, confidence_threshold: float = None
    ) -> Generator[OWLClass, None, None]:  # pragma: no cover
        """At least it has single superclass and there is no subclass"""
        for _class in self.classes_in_signature(confidence_threshold):
            for concept in self.subconcepts(
                    named_concept=_class, confidence_threshold=confidence_threshold
            ):
                break
            else:
                # checks if superclasses is not empty -> there is at least one superclass
                if superclasses := list(
                        self.get_direct_parents(_class, confidence_threshold)
                ):
                    yield _class

    def subconcepts(
            self, named_concept: OWLClass, confidence_threshold: float = None
    ) -> Generator[OWLClass, None, None]:
        for prediction in self.get_predictions(
                h=None,
                r="http://www.w3.org/2000/01/rdf-schema#subClassOf",
                t=named_concept.str,
                confidence_threshold=confidence_threshold,
        ):
            try:
                yield OWLClass(prediction[0])
            except Exception as e:  # pragma: no cover
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def get_direct_parents(
            self, named_concept: OWLClass, direct=True, confidence_threshold: float = None
    ):  # pragma: no cover
        for prediction in self.get_predictions(
                h=named_concept.str,
                r="http://www.w3.org/2000/01/rdf-schema#subClassOf",
                t=None,
                confidence_threshold=confidence_threshold,
        ):
            try:
                yield OWLClass(prediction[0])
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
                yield OWLClass(prediction[0])
            except Exception as e:  # pragma: no cover
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def individuals(self, expression: OWLClassExpression = None, named_individuals: bool = False,
                    confidence_threshold: float = None, ) -> Generator[OWLNamedIndividual, None, None]:
        if expression is None or expression.is_owl_thing():
            yield from self.individuals_in_signature()
        else:
            yield from self.instances(expression, confidence_threshold=confidence_threshold)

    def instances(self, expression: OWLClassExpression, named_individuals=False,
                  confidence_threshold: float = None, ) -> Generator[OWLNamedIndividual, None, None]:
        """
        if expression.is_owl_thing():
            yield from self.individuals_in_signature()
        """

        """
        if isinstance(expression, OWLNamedIndividual):
            # TODO: CD: expression should not be an instance of  OWLNamedIndividual
            # TODO: CD: Perhaps, we need to ensure that is never the case :).
            yield expression
        """

        if isinstance(expression, OWLClass):
            """ Given an OWLClass A, retrieve its instances Retrieval(A)={ x | phi(x, type, A) ≥ γ } """
            yield from self.get_individuals_of_class(
                owl_class=expression, confidence_threshold=confidence_threshold
            )
        # Handling complement of class expressions
        elif isinstance(expression, OWLObjectComplementOf):
            """ Given an OWLObjectComplementOf ¬A, hence (A is an OWLClass),
            retrieve its instances => Retrieval(¬A)= All Instance Set-DIFF { x | phi(x, type, A) ≥ γ } """
            excluded_individuals = set(self.instances(expression.get_operand(), confidence_threshold))
            yield from self.set_inferred_individuals - excluded_individuals
        # Handling intersection of class expressions
        elif isinstance(expression, OWLObjectIntersectionOf):
            """ Given an OWLObjectIntersectionOf (C ⊓ D),  
            retrieve its instances by intersecting the instance of each operands.
            {x | phi(x, type, C) ≥ γ} ∩ {x | phi(x, type, D) ≥ γ}
            """
            # Get the class expressions
            #
            result = None
            for op in expression.operands():
                retrieval_of_op = {_ for _ in self.instances(expression=op, confidence_threshold=confidence_threshold)}
                if result is None:
                    result = retrieval_of_op
                else:
                    result = result.intersection(retrieval_of_op)
            yield from result
            """
            operands = list(expression.operands())
            sets_of_individuals = [
                set(
                    self.instances(
                        expression=operand, confidence_threshold=confidence_threshold
                    )
                )
                for operand in operands
            ]

            if sets_of_individuals:
                # Start with the set of individuals from the first operand
                common_individuals = sets_of_individuals[0]

                # Update the common individuals set with the intersection of subsequent sets
                for individuals in sets_of_individuals[1:]:
                    common_individuals.intersection_update(individuals)

                # Yield individuals that are common across all operands
                for individual in common_individuals:
                    yield individual
            """
        elif isinstance(expression, OWLObjectAllValuesFrom):
            """
            Given an OWLObjectAllValuesFrom ∀ r.C, retrieve its instances => 
            Retrieval(¬∃ r.¬C) =             
            Entities \setminus {x | ∃ y: \phi(y, type, C) < \gamma AND \phi(x,r,y)  ≥ \gamma } 
            """
            object_property = expression.get_property()
            filler_expression = expression.get_filler()

            filler_individuals = set(self.instances(filler_expression, confidence_threshold))
            to_yield_individuals = set()

            for individual in self.set_inferred_individuals:
                related_individuals = set(self.get_object_property_values(individual.str, object_property))
                if not related_individuals or related_individuals <= filler_individuals:
                    to_yield_individuals.add(individual)

            yield from to_yield_individuals
        elif isinstance(expression, OWLObjectMinCardinality) or isinstance(expression, OWLObjectSomeValuesFrom):
            """
            Given an OWLObjectSomeValuesFrom ∃ r.C, retrieve its instances => 
            Retrieval(∃ r.C) = 
            {x | ∃ y : phi(y, type, C) ≥ \gamma AND phi(x, r, y) ≥ \gamma }  
            """
            object_property = expression.get_property()
            filler_expression = expression.get_filler()
            cardinality = 1
            if isinstance(expression, OWLObjectMinCardinality):
                cardinality = expression.get_cardinality()

            object_individuals = self.instances(filler_expression, confidence_threshold)

            # Initialize counter to keep track of individual occurrences
            result = Counter()

            # Iterate over each object individual to find and count subjects
            for object_individual in object_individuals:
                subjects = self.get_individuals_with_object_property(
                    obj=object_individual,
                    object_property=object_property,
                    confidence_threshold=confidence_threshold
                )
                # Update the counter for all subjects found
                result.update(subjects)

            # Yield only those individuals who meet the cardinality requirement
            for individual, count in result.items():
                if count >= cardinality:
                    yield individual
        elif isinstance(expression, OWLObjectMaxCardinality):
            object_property: OWLObjectProperty
            object_property = expression.get_property()

            filler_expression:OWLClassExpression
            filler_expression = expression.get_filler()

            cardinality:int
            cardinality = expression.get_cardinality()

            # Get all individuals that are instances of the filler expression.
            owl_individual:OWLNamedIndividual
            object_individuals = { owl_individual for owl_individual
                                   in self.instances(filler_expression, confidence_threshold)}

            # Initialize a dictionary to keep track of counts of related individuals for each entity.
            owl_individual:OWLNamedIndividual
            str_subject_individuals_to_count = {owl_individual.str: (owl_individual,0) for owl_individual in self.set_inferred_individuals}

            for object_individual in object_individuals:
                # Get all individuals related to the object individual via the object property.
                subject_individuals = self.get_individuals_with_object_property(obj=object_individual,
                                                              object_property=object_property,
                                                              confidence_threshold=confidence_threshold)

                # Update the count of related individuals for each object individual.
                for subject_individual in subject_individuals:
                    if subject_individual.str in str_subject_individuals_to_count:
                        owl_obj, count = str_subject_individuals_to_count[subject_individual.str]
                        # Increment the count.
                        str_subject_individuals_to_count[subject_individual.str] = (owl_obj, count+1)

            # Filter out individuals who exceed the specified cardinality.
            yield from  {ind for str_ind, (ind, count) in str_subject_individuals_to_count.items() if count <= cardinality}

        # Handling union of class expressions
        elif isinstance(expression, OWLObjectUnionOf):
            # Get the class expressions

            result = None
            for op in expression.operands():
                retrieval_of_op = {_ for _ in self.instances(expression=op, confidence_threshold=confidence_threshold)}
                if result is None:
                    result = retrieval_of_op
                else:
                    result = result.union(retrieval_of_op)
            yield from result

        elif isinstance(expression, OWLObjectOneOf):
            yield from expression.individuals()
        else:
            raise NotImplementedError(
                f"Instances for {type(expression)} are not implemented yet"
            )

    def individuals_in_signature(self) -> Generator[OWLNamedIndividual, None, None]:

        if self.inferred_owl_individuals is None:
            seen_individuals = set()
            try:
                for cl in self.classes_in_signature():
                    predictions = self.get_predictions(
                        h=None, r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t=cl.str,
                        confidence_threshold=self.gamma)
                    for prediction in predictions:
                        try:
                            owl_named_individual = OWLNamedIndividual(prediction[0])
                            if owl_named_individual not in seen_individuals:
                                seen_individuals.add(owl_named_individual)
                                yield owl_named_individual
                        except Exception as e:  # pragma: no cover
                            print(f"Invalid IRI detected: {prediction[0]}, error: {e}")

                    self.inferred_owl_individuals = seen_individuals
            except Exception as e:  # pragma: no cover
                print(f"Error processing classes in signature: {e}")
        else:
            yield from self.inferred_owl_individuals

    def data_properties_in_signature(
            self, confidence_threshold: float = None
    ) -> Generator[OWLDataProperty, None, None]:
        for prediction in self.get_predictions(
                h=None,
                r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                t="http://www.w3.org/2002/07/owl#DatatypeProperty",
                confidence_threshold=confidence_threshold,
        ):
            try:  # pragma: no cover
                yield OWLDataProperty(prediction[0])
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def object_properties_in_signature(
            self, confidence_threshold: float = None
    ) -> Generator[OWLObjectProperty, None, None]:
        if self.inferred_object_properties is None:
            self.inferred_object_properties = set()
            for prediction in self.get_predictions(
                    h=None,
                    r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    t="http://www.w3.org/2002/07/owl#ObjectProperty",
                    confidence_threshold=confidence_threshold,
            ):
                try:
                    owl_obj_property = OWLObjectProperty(prediction[0])
                    self.inferred_object_properties.add(owl_obj_property)
                    yield owl_obj_property
                except Exception as e:  # pragma: no cover
                    # Log the invalid IRI
                    print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                    continue

    def boolean_data_properties(
            self, confidence_threshold: float = None
    ) -> Generator[OWLDataProperty, None, None]:  # pragma: no cover
        for prediction in self.get_predictions(
                h=None,
                r="http://www.w3.org/2000/01/rdf-schema#range",
                t="http://www.w3.org/2001/XMLSchema#boolean",
                confidence_threshold=confidence_threshold,
        ):
            try:
                yield OWLDataProperty(prediction[0])
            except Exception as e:
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def double_data_properties(
            self, confidence_threshold: float = None
    ) -> Generator[OWLDataProperty, None, None]:  # pragma: no cover
        for prediction in self.get_predictions(
                h=None,
                r="http://www.w3.org/2000/01/rdf-schema#range",
                t="http://www.w3.org/2001/XMLSchema#double",
                confidence_threshold=confidence_threshold,
        ):
            try:
                yield OWLDataProperty(prediction[0])
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
        if is_inverse := isinstance(object_property, OWLObjectInverseOf):
            object_property = object_property.get_inverse()
        for prediction in self.get_predictions(
                h=None if is_inverse else subject,
                r=object_property.str,
                t=subject if is_inverse else None,
                confidence_threshold=confidence_threshold,
        ):
            try:
                yield OWLNamedIndividual(prediction[0])
            except Exception as e:  # pragma: no cover
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

    def get_data_property_values(
            self,
            subject: str,
            data_property: OWLDataProperty,
            confidence_threshold: float = None,
    ) -> Generator[OWLLiteral, None, None]:  # pragma: no cover
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
        predictions = self.get_predictions(
            h=None,
            r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            t=owl_class.str,
            confidence_threshold=confidence_threshold,
        )
        seen = set()
        for prediction in predictions:
            try:
                owl_named_individual = OWLNamedIndividual(prediction[0])
                if owl_named_individual not in seen:
                    seen.add(owl_named_individual)
                    yield owl_named_individual
            except Exception as e:  # pragma: no cover
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue

        if len(list(predictions)) == 0:
            # abstract class / class that does not have any instances -> get all child classes and make predictions
            for child_class in self.subconcepts(owl_class, confidence_threshold=confidence_threshold):
                for individual in self.get_individuals_of_class(child_class, confidence_threshold=confidence_threshold):
                    if individual not in seen:
                        seen.add(individual)
                        yield individual

    def get_individuals_with_object_property(
            self,
            object_property: OWLObjectProperty, obj: OWLClass, confidence_threshold: float = None ) \
            -> Generator[OWLNamedIndividual, None, None]:

        is_inverse = isinstance(object_property, OWLObjectInverseOf)

        if is_inverse:
            object_property = object_property.get_inverse()

        for prediction in self.get_predictions(
                h=obj.str if is_inverse else None,
                r=object_property.str,
                t=None if is_inverse else obj.str,
                confidence_threshold=confidence_threshold):

            try:
                yield OWLNamedIndividual(prediction[0])
            except Exception as e:  # pragma: no cover
                # Log the invalid IRI
                print(f"Invalid IRI detected: {prediction[0]}, error: {e}")
                continue
