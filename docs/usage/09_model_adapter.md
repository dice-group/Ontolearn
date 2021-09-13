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
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from owlapy.model import OWLOntology, OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.owlready2 import OWLOntology_Owlready2, OWLOntologyManager_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.render import DLSyntaxObjectRenderer


def my_reasoner_factory(onto: OWLOntology):
    assert isinstance(onto, OWLOntology_Owlready2)
    temp_classes_reasoner = OWLReasoner_Owlready2_TempClasses(onto)
    fast_instance_checker = OWLReasoner_FastInstanceChecker(
        onto,
        temp_classes_reasoner)
    return fast_instance_checker


NS = Namespaces('ex', 'http://example.com/father#')

positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}
negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}

# Only the class of the learning algorithm is specified
model = ModelAdapter(learner_type=CELOE,
                     ontologymanager_factory=OWLOntologyManager_Owlready2,  # (*)
                     reasoner_factory=my_reasoner_factory,  # (*)
                     path="KGs/father.owl",
                     quality_type=Accuracy,
                     heuristic_type=CELOEHeuristic,
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
the default choices.
