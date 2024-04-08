from ontolearn.learners import Drill, TDL
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.verbalizer import LLMVerbalizer
from owlapy.model import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import compute_f1_score
import json

# (1) Load a knowledge graph.
kb = TripleStore(path='KGs/Family/family-benchmark_rich_background.owl')
render = DLSyntaxObjectRenderer()
# (2) Get learning problems.
with open("LPs/Family/lps.json") as json_file:
    settings = json.load(json_file)
# (3) Initialize learner
model = Drill(knowledge_base=kb, use_nominals=False)
# (4)
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    h = model.fit(learning_problem=lp).best_hypotheses(1)
    str_concept = render.render(h)
    f1_score = compute_f1_score(individuals=frozenset({i for i in kb.individuals(h)}), pos=lp.pos, neg=lp.neg)
    # CD: We need to specify ranges for the regression tests.
    assert f1_score >= 0.5
