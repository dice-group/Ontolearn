from ontolearn.learners import TDL
from ontolearn.triple_store import TripleStore
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.utils.static_funcs import save_owl_class_expressions
from owlapy.converter import Owl2SparqlConverter
import json


class TestTriplestore:
    def test_local_triplestore_family_tdl(self):
        # @TODO: CD: Removed because rdflib does not produce correct results
        """


        # (1) Load a knowledge graph.
        kb = TripleStore(path='KGs/Family/family-benchmark_rich_background.owl')
        # (2) Get learning problems.
        with open("LPs/Family/lps.json") as json_file:
            settings = json.load(json_file)
        # (3) Initialize learner.
        model = TDL(knowledge_base=kb, kwargs_classifier={"max_depth": 2})
        # (4) Fitting.
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            h = model.fit(learning_problem=lp).best_hypotheses()
            print(h)
            predicted_expression = frozenset({i for i in kb.individuals(h)})
            print("Number of individuals:", len(predicted_expression))
            q = compute_f1_score(individuals=predicted_expression, pos=lp.pos, neg=lp.neg)
            print(q)
            assert q>=0.80
            break
        """
    def test_remote_triplestore_dbpedia_tdl(self):
        """
        url = "http://dice-dbpedia.cs.upb.de:9080/sparql"
        kb = TripleStore(url=url)
        # Check whether there is a connection
        num_object_properties = len([i for i in kb.get_object_properties()])
        if num_object_properties > 0:
            examples = {"positive_examples": ["http://dbpedia.org/resource/Angela_Merkel"],
                        "negative_examples": ["http://dbpedia.org/resource/Barack_Obama"]}
            model = TDL(knowledge_base=kb, report_classification=True, kwargs_classifier={"random_state": 1})
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, examples["positive_examples"])))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, examples["negative_examples"])))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            h = model.fit(learning_problem=lp).best_hypotheses()
            assert h
            assert DLSyntaxObjectRenderer().render(h)
            save_owl_class_expressions(h)
            sparql = Owl2SparqlConverter().as_query("?x", h)
            assert sparql
        else:
            pass
        """

