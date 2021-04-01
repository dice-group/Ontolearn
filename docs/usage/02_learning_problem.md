# Defining a Learning Problem

The Structured Machine Learning implemented in our Ontolearn library
is working with a type of [supervised
learning](https://en.wikipedia.org/wiki/Supervised_learning).  One of
the first things to do after loading the Ontology is thus to define
the positive and negative examples whose description the learning
algorithm should attempt to find.

## Referencing Named Individuals

Let's assume we are working with the Ontology `father.owl` that was
loaded in the previous chapter. Our positive examples (individuals to
describe) are stefan, markus, and martin. And our negative examples
(individuals to not describe) are heinz, anna, and michelle. Then we
could write the following Python code:

```py
from owlapy import IRI
from owlapy.namespaces import Namespaces
from owlapy.model import OWLNamedIndividual

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

## Knowledge Base over the Ontology

The [knowledge base](ontolearn.knowledge_base.KnowledgeBase) is the Ontolearn
representation of an Ontology and a Reasoner. It also contains a
Hierarchy generator as well as other Ontology-related state required
for the Structured Machine Learning library. It is required to run a
learning algorithm. Creation is done like follows:

```py
from ontolearn import KnowledgeBase

kb = KnowledgeBase(ontology=onto, reasoner=fast_instance_checker)
```

Further actions are possible on the knowledge base, for example the
ignorance of specific classes. Refer to the documentation for the full
details.

## Creating the Learning Problem

Now the learning problem can be captured in its respective object, the
[positive-negative standard learning
problem](ontolearn.learning_problem.PosNegLPStandard):

```py
from ontolearn.learning_problem import PosNegLPStandard

lp = PosNegLPStandard(kb, pos=positive_examples, neg=negative_examples)
```

