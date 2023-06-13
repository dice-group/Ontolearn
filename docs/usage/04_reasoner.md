# Working with Reasoners

To validate facts about statements in the ontology (and thus
also for the Structured Machine Learning task), the help of a reasoner
component is required.

In our Ontolearn library, we provide several **reasoners** to choose
from. Currently, there are the following reasoners available for you to choose from: 

- Fast instance checker: [OWLReasoner_FastInstanceChecker](owlapy.fast_instance_checker.OWLReasoner_FastInstanceChecker)
- Structural Owlready2 reasoner: [OWLReasoner_Owlready2](owlapy.owlready2.OWLReasoner_Owlready2)
- Class instantiation Owlready2 reasoner:  [OWLReasoner_Owlready2_TempClasses](owlapy.owlready2.temp_classes.OWLReasoner_Owlready2_TempClasses)


To load any reasoner, use the following code:

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

----------------------------------------------------------------------------

## Usage of the Reasoner
All the reasoners available in the Ontolearn library inherit from the
class: [OWLReasonerEx](owlapy.ext.OWLReasonerEx). This class provides some 
extra convenient methods compared to its base class [OWLReasoner](owlapy.model.OWLReasoner).
Further in this guide, we will use 
[OWLReasoner_FastInstanceChecker](owlapy.fast_instance_checker.OWLReasoner_FastInstanceChecker) (FIC).
to show the capabilities of a reasoner implemented in Ontolearn.

To give examples we will consider the _father_ dataset. 
If you are not already familiar with this small dataset,
you can find an overview of it [here](01_knowledge_base.md).


## Class Reasoning

Using an [OWLOntology](owlapy.model.OWLOntology) you can list all the classes in the signature, 
but a reasoner can give you more than that. You can get the subclasses, superclasses or the 
equivalent classes of a class in the ontology:

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLClass
from owlapy.model import IRI
namespace = "http://example.com/father#"
male = OWLClass(IRI(namespace,"male"))

male_super_classes = fast_instance_checker.super_classes(male)
male_sub_classes = fast_instance_checker.sub_classes(male)
male_equivalent_classes = fast_instance_checker.equivalent_classes(male)
```

We define the _male_ class by creating an [OWLClass](owlapy.model.OWLClass) object. The 
methods `super_classes` and `sub_classes` have 2 more boolean arguments: `direct` and `only_named`. 
If `direct=True` then only the direct classes in the 
hierarchy will be returned, else it will return every class in the hierarchy depending 
on the method(sub_classes or super_classes).
By default, its value is _False_. 
The next argument `only_named` specifies whether you want
to show only named classes or complex classes as well. By default, its value is _True_ which 
means that it will return only the named classes.

>**NOTE**: The extra arguments `direct` and `only_named` are also used in other methods that reason
upon the class, object property, or data property hierarchy and not only.

You can get all the types of a certain individual using `types` method:

<!--pytest-codeblocks:cont-->
```python
from owlapy.owlready2 import OWLOntologyManager_Owlready2

manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create("file://KGs/father.owl"))
anna = list(onto.individuals_in_signature()).pop() # get the 1st individual in the list of individuals which is 'anna'

anna_types = fast_instance_checker.types(anna)
```

We first create a manager and use that to open the _father_ ontology. 
We retrieve _anna_ as the first individual on the list of individuals 
of this ontology. You can see more details on ontologies in the [Working with Ontologies](03_ontologies.md)
guide. 


## Object Properties and Data Properties Reasoning
Fast instance checker offers some convenient methods for working with object properties and 
data properties. Below we show some of them, but you can always check all the methods in the 
[OWLReasoner_FastInstanceChecker](owlapy.fast_instance_checker.OWLReasoner_FastInstanceChecker) 
class documentation. 

You can get all the object properties that an individual has by using the 
following method:

<!--pytest-codeblocks:cont-->
```python
anna = individuals[0] 
object_properties = fast_instance_checker.ind_object_properties(anna)
```
In this example, `object_properties` contains all the object properties
that _anna_ has, which in our case would only be _hasChild_.
Now we can get the individuals of this object property for _anna_.

<!--pytest-codeblocks:cont-->
```python
for op in object_properties:
    object_properties_values = fast_instance_checker.object_property_values(anna, op)
    for individual in object_properties_values:
        print(individual)
```

In this example we iterated over the `object_properties`, assuming that there
are more than 1, and we use the reasoner
to get the values for each object property `op` of the individual `anna`. The values 
are individuals which we store in the variable `object_properties_values` and are 
printed in the end. The method `object_property_values` requires as the
first argument, an [OWLNamedIndividual](owlapy.model.OWLNamedIndividual) that is the subject of the object property values and 
the second argument an [OWLObjectProperty](owlapy.model.OWLObjectProperty) whose values are to be retrieved for the 
specified individual.  

> **NOTE:** You can as well get all the data properties of an individual in the same way by using 
`ind_data_properties` instead of `ind_object_properties` and `data_property_values` instead of 
`object_property_values`. Keep in mind that `data_property_values` returns literal values 
(type of [OWLLiteral](owlapy.model.OWLLiteral)) instead of individuals.

In the same way as with classes, you can also get the sub object properties or equivalent object properties.

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLObjectProperty

hasChild = OWLObjectProperty(IRI(namespace,"hasChild"))

equivalent_to_hasChild = fast_instance_checker.equivalent_object_properties(hasChild)
hasChild_sub_properties = fast_instance_checker.sub_object_properties(hasChild)
```

In case you want to get the domains and ranges of an object property use the following:

<!--pytest-codeblocks:cont-->
```python
hasChild_domains = fast_instance_checker.object_property_domains(hasChild)
hasChild_ranges = fast_instance_checker.object_property_ranges(hasChild)
```

> **NOTE:** Again, you can do the same for data properties but instead of the word 'object' in the 
> method name you should put 'data'.


## Find Instances

The method `instances` of fast instance checker is a very convenient method. It takes only 1 argument that is basically
a class expression and returns all the individuals belonging to that class expression. In Ontolearn 
we have implemented a Python class for each type of class expression. Therefore `instances` 
has multiple implementations depending on which type of argument is passed, but you don't have to 
worry about any of that. Below you will find a list of all the supported class expressions for this 
method:

- [OWLClass](owlapy.model.OWLClass)
- [OWLObjectUnionOf](owlapy.model.OWLObjectUnionOf)
- [OWLObjectIntersectionOf](owlapy.model.OWLObjectIntersectionOf)
- [OWLObjectSomeValuesFrom](owlapy.model.OWLObjectSomeValuesFrom)
- [OWLObjectComplementOf](owlapy.model.OWLObjectComplementOf)
- [OWLObjectAllValuesFrom](owlapy.model.OWLObjectAllValuesFrom)
- [OWLObjectOneOf](owlapy.model.OWLObjectOneOf)
- [OWLObjectHasValue](owlapy.model.OWLObjectHasValue)
- [OWLObjectMinCardinality](owlapy.model.OWLObjectMinCardinality)
- [OWLObjectMaxCardinality](owlapy.model.OWLObjectMaxCardinality)
- [OWLObjectExactCardinality](owlapy.model.OWLObjectExactCardinality)
- [OWLDataSomeValuesFrom](owlapy.model.OWLDataSomeValuesFrom)
- [OWLDataAllValuesFrom](owlapy.model.OWLDataAllValuesFrom)
- [OWLDataHasValue](owlapy.model.OWLDataHasValue)

Let us now show a simple example by finding the instances of the class _male_ and printing them:

<!--pytest-codeblocks:cont-->
```python
male_individuals = fast_instance_checker.instances(male)
for ind in male_individuals:
    print(ind)
```



## Sync Reasoner

_sync_reasoner_ is a definition used in owlready2 to call HermiT or Pellet. We made it possible in our
reasoners to use that functionality by implementing a delegator method. This is an 
internal method of the class 
[OWLReasoner_Owlready2](owlapy.owlready2.OWLReasoner_Owlready2) named `_sync_reasoner`
which is used at [OWLReasoner_Owlready2_TempClasses](owlapy.owlready2.temp_classes.OWLReasoner_Owlready2_TempClasses) 
in the `instances` method and only for complex classes.
Soon we will implement the other methods of the base class for _OWLReasoner_Owlready2_TempClasses_ and
update the `instances` method to cover the atomic classes as well.


In the next guide, we show how to use concept learners to learn class expressions in a 
knowledge base for a certain learning problem.