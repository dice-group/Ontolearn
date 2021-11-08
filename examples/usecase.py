import random

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import OWLClass, OWLObjectSomeValuesFrom, OWLObjectProperty, IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2, OWLReasoner_Owlready2
from owlapy.render import DLSyntaxObjectRenderer

if __name__ == '__main__':
    # In[45]:

    mgr = OWLOntologyManager_Owlready2()
    onto = mgr.load_ontology(IRI.create("file://ai4bd-sml1.owl"))
    base_reasoner = OWLReasoner_Owlready2(onto)
    reasoner = OWLReasoner_FastInstanceChecker(onto, base_reasoner,
                                               negation_default=True)

    kb = KnowledgeBase(ontology=onto, reasoner=reasoner)

    # In[46]:

    NS = 'http://example.com/daikiri#'

    # In[22]:

    list(onto.classes_in_signature())

    # In[47]:

    pos = set(reasoner.instances(OWLObjectSomeValuesFrom(filler=OWLClass(IRI.create(NS, 'anomaly1_True')),
                                                         property=OWLObjectProperty(IRI.create(NS, 'anomaly1')))))

    # In[48]:

    nan_list = list(reasoner.instances(OWLObjectSomeValuesFrom(filler=OWLClass(IRI.create(NS, 'anomaly1_nan')),
                                                               property=OWLObjectProperty(IRI.create(NS, 'anomaly1')))))
    sample = random.sample(nan_list, len(pos) * 10)
    tneg = set(reasoner.instances(OWLObjectSomeValuesFrom(filler=OWLClass(IRI.create(NS, 'anomaly1_False')),
                                                          property=OWLObjectProperty(IRI.create(NS, 'anomaly1')))))
    neg = tneg | set(sample)
    random.sample(neg, 10)

    # In[49]:

    kb = kb.ignore_and_copy(ignored_classes=(OWLClass(IRI.create(NS, 'anomaly1_True')),
                                             OWLClass(IRI.create(NS, 'anomaly1_False')),
                                             OWLClass(IRI.create(NS, 'anomaly1_nan'))))

    # In[26]:

    list(kb.ontology().object_properties_in_signature())

    # In[50]:

    lp = PosNegLPStandard(knowledge_base=kb, pos=pos, neg=neg)
    pred_acc = Accuracy(lp)
    f1 = F1(lp)
    alg = CELOE(knowledge_base=kb,
                learning_problem=lp,
                max_runtime=60,
                iter_bound=1_000_000,
                max_num_of_concepts_tested=1_000_000,
                )

    # In[ ]:

    alg.fit()

    # In[29]:

    render = DLSyntaxObjectRenderer()

    # In[40]:

    print("solutions:")
    i = 1
    for h in alg.best_hypotheses(3):
        individuals_set = kb.individuals_set(h.concept)
        print(f'{i}: {render.render(h.concept)} ('
              f'pred. acc.: {pred_acc.score(individuals_set)[1]}, '
              f'F-Measure: {f1.score(individuals_set)[1]}'
              f') [Node '
              f'quality: {h.quality}, h-exp: {h.h_exp}, RC: {h.refinement_count}'
              f']')
        i += 1
