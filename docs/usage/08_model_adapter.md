# Model Adaptor

To simplify the connection between all the
components, there is a
model adaptor available that automatically constructs and connects them.
Here is how to implement the previous example using the [ModelAdapter](ontolearn.mode_adapter.ModelAdapter):

```python
from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.metrics import Accuracy
from ontolearn.model_adapter import ModelAdapter
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from ontolearn.base import OWLOntologyManager_Owlready2
from ontolearn.base import OWLReasoner_Owlready2_ComplexCEInstances
from owlapy.render import DLSyntaxObjectRenderer

manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create("KGs/Family/father.owl"))
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
                     path="KGs/Family/father.owl",
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
the default choices. For now, you can use ModelAdapter only for EvoLearner, CELOE and OCEL.
