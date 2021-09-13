# Working with Ontologies

To get started with Structured Machine Learning, the first thing
required is an [Ontology](https://www.w3.org/TR/owl2-overview/) with
[Named
Individuals](https://www.w3.org/TR/owl-syntax/#Named_Individuals). We
cannot provide such ontologies for you and depending on the use case,
it may be necessary to first map existing data into an ontology.

However, some sample ontologies are included with Ontolearn so that
you can start using it right away. One such sample ontology is
contained in the `KGs/father.owl` file. It contains six persons
(individuals), of which four are male and two are female. We will use
this tiny ontology as an example.

## Loading an Ontology

To load an ontology, use the following Python code:

```python
from owlapy.model import IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2

mgr = OWLOntologyManager_Owlready2()
onto = mgr.load_ontology(IRI.create("file://KGs/father.owl"))
```

First, we import the IRI class and a suitable OWLOntologyManager. To
load a file from our computer, we have to reference it with an
[IRI](https://tools.ietf.org/html/rfc3987). Secondly, we need an
Ontology Manager, which is a component that can manage ontologies for
us. Currently, Ontolearn contains one such manager: The
[OWLOntologyManager_Owlready2](owlapy.owlready2.OWLOntologyManager_Owlready2).

Now, we can already inspect the contents of the ontology. For example,
to list all individuals:

<!--pytest-codeblocks:cont-->
```python
for ind in onto.individuals_in_signature():
    print(ind)
```

Refer to the [OWLOntology](owlapy.model.OWLOntology) documentation for
more details.

## Attaching a reasoner

In order to validate facts about statements in the ontology (and thus
also for the Structured Machine Learning task), the help of a reasoner
component is required.

In our Ontolearn library, we provide several reasoners to choose
from. Currently, there are the the
[fast instance checker](owlapy.fast_instance_checker.OWLReasoner_FastInstanceChecker),
the
[structural Owlready2 reasoner](owlapy.owlready2.OWLReasoner_Owlready2),
and the [class instantiation Owlready2 reasoner](owlapy.owlready2.temp_classes.OWLReasoner_Owlready2_TempClasses)
available to choose from.

To load any reasoner, follow this Python code:

<!--pytest-codeblocks:cont-->
```python
from owlapy.owlready2 import OWLReasoner_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker

structural_reasoner = OWLReasoner_Owlready2(onto)
temp_classes_reasoner = OWLReasoner_Owlready2_TempClasses(onto)
fast_instance_checker = OWLReasoner_FastInstanceChecker(onto, temp_classes_reasoner)
```

The reasoner takes as its first argument the ontology to load. The
fast instance checker requires a base reasoner to which any reasoning
tasks not covered by the fast instance checking code are deferred to.


