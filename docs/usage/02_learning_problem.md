<!--
```python
from owlapy.model import IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2
from owlapy.owlready2 import OWLReasoner_Owlready2
from owlapy.owlready2.complex_ce_instances import OWLReasoner_Owlready2_ComplexCEInstances
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker

mgr = OWLOntologyManager_Owlready2()
onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))

structural_reasoner = OWLReasoner_Owlready2(onto)
complex_ce_reasoner = OWLReasoner_Owlready2_ComplexCEInstances(onto)
fast_instance_checker = OWLReasoner_FastInstanceChecker(onto, complex_ce_reasoner)
```
-->

# Learning Problem

The Structured Machine Learning implemented in our Ontolearn library
is working with a type of [supervised
learning](https://en.wikipedia.org/wiki/Supervised_learning).  One of
the first things to do after loading the Ontology is thus to define
the positive and negative examples whose description the learning
algorithm should attempt to find.

### Referencing Named Individuals

Let's assume we are working with the Ontology `father.owl` that was
loaded in the previous chapter. Our positive examples (individuals to
describe) are stefan, markus, and martin. And our negative examples
(individuals to not describe) are heinz, anna, and michelle. Then we
could write the following Python code:

<!--pytest-codeblocks:cont-->
```python
from owlapy.namespaces import Namespaces
from owlapy.model import OWLNamedIndividual, IRI

NS = Namespaces('ex', 'http://example.com/father#')

positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}
negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}
```

Note that the namespace has to match the Namespace/IRI that is defined
in the Ontology document.


### Creating the Learning Problem

Now the learning problem can be captured in its respective object, the
[positive-negative standard learning
problem](ontolearn.learning_problem.PosNegLPStandard):

<!--pytest-codeblocks:cont-->
```python
from ontolearn.learning_problem import PosNegLPStandard

lp = PosNegLPStandard(pos=positive_examples, neg=negative_examples)
```

In the next guide, you can learn more about ontologies in Ontolearn and how you can modify them
using axioms.