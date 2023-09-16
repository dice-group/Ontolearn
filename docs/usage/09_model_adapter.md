# Model Adaptor

To simplify the connection between all the
[Components](08_architecture.md#component-architecture), there is a
model adaptor available that automatically constructs and connects them.
Here is how to implement the previous example using the Model Adaptor:

```python
from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.metrics import Accuracy
from ontolearn.model_adapter import ModelAdapter
from ontolearn.owlapy.model import OWLNamedIndividual, IRI
from ontolearn.owlapy.namespaces import Namespaces
from ontolearn.owlapy.owlready2 import OWLOntologyManager_Owlready2
from ontolearn.owlapy.owlready2.complex_ce_instances import OWLReasoner_Owlready2_ComplexCEInstances
from ontolearn.owlapy.render import DLSyntaxObjectRenderer

manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create("KGs/father.owl"))
complex_ce_reasoner = OWLReasoner_Owlready2_ComplexCEInstances(onto)

NS = Namespaces('ex', 'http://example.com/father#')

positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}
negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}

# Only the class of the learning algorithm is specified
model = ModelAdapter(learner_type=CELOE,
                     reasoner=complex_ce_reasoner,  # (*)
                     path="KGs/father.owl",
                     quality_type=Accuracy,
                     heuristic_type=CELOEHeuristic,  # (*)
                     expansionPenaltyFactor=0.05,
                     startNodeBonus=1.0,
                     nodeRefinementPenalty=0.01,
                     )

# no need to construct the IRI here ourselves
model.fit(pos=positive_examples,
          neg=negative_examples,
          )

dlsr = DLSyntaxObjectRenderer()

for desc in model.best_hypotheses(1):
    print('The result:', dlsr.render(desc.concept), 'has quality', desc.quality)
```

Lines marked with `(*)` are not strictly required as they happen to be
the default choices. For now, you can use ModelAdaptor only for EvoLearner, CELOE and OCEL.
