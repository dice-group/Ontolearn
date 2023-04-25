# Working with Ontologies

In this guide, we will explain how to modify or get specific data from an Ontology. There 
are some similar functionalities as the `KnowledgeBase` class which we describe here:
[Working with Knowledge Bases](01_knowledge_base.md). 

## Loading an Ontology
> **Note:**  Although you can load an ontology directly, you will still need a `KnowledgeBase` object to run
concept learning algorithms.


To load an ontology as well as to manage it, you will need an [OWLOntologyManager](owlapy.owlready2.OWLOntologyManager).
To load an ontology, use the following Python code:

```python
from owlapy.model import IRI
from owlapy.owlready2 import OWLOntologyManager_Owlready2

manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create("file://KGs/father.owl"))
```

First, we import the `IRI` class and a suitable OWLOntologyManager. To
load a file from our computer, we have to reference it with an
[IRI](https://tools.ietf.org/html/rfc3987). Secondly, we need the
Ontology Manager. Currently, Ontolearn contains one such manager: The
[OWLOntologyManager_Owlready2](owlapy.owlready2.OWLOntologyManager_Owlready2).

Now, we can already inspect the contents of the ontology. For example,
to list all individuals:

<!--pytest-codeblocks:cont-->
```python
for ind in onto.individuals_in_signature():
    print(ind)
```

You can get the object properties in the signature:

<!--pytest-codeblocks:cont-->
```python
onto.object_properties_in_signature()
```

For more methods, refer to the [OWLOntology](owlapy.model.OWLOntology) documentation.

----------------------------------------------------------------------------

## Modifying an Ontology

Axioms in ontology serve as the basis for defining the vocabulary of a domain and for 
making statements about the relationships between entities and concepts in that domain.
They provide a formal and precise way to represent knowledge and allow for automated 
reasoning and inference. Axioms can be **added**, **modified**, or **removed** from an ontology, 
allowing the ontology to evolve and adapt as new knowledge is gained.

In Ontolearn we also have different axioms represented by different classes. You can check all
the axioms classes [here](owlapy.model). The main axioms you may use more frequently are:

- [OWLDeclarationAxiom](owlapy.model.OWLDeclarationAxiom)
- [OWLObjectPropertyAssertionAxiom](owlapy.model.OWLObjectPropertyAssertionAxiom)
- [OWLDataPropertyAssertionAxiom](owlapy.model.OWLObjectPropertyAssertionAxiom)
- [OWLClassAssertionAxiom](owlapy.model.OWLObjectPropertyAssertionAxiom)


#### Add a new Class

Let's suppose you want to add a new class in our example ontology `KGs/father.owl` 
(You can find this ontology inside the project or click [here](01_knowledge_base.md) 
to see a description of it). It can be done as follows:

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLClass
from owlapy.model import OWLDeclarationAxiom

iri = IRI('http://example.com/father#', 'child')
child_class = OWLClass(iri)
child_class_declaration_axiom = OWLDeclarationAxiom(child_class)

manager.add_axiom(onto, child_class_declaration_axiom)
```
In this example, we added the class 'child' to the father.owl ontology.
Firstly we create an instance of [OWLClass](owlapy.model.OWLClass) to represent the concept 
of 'child' by using an [IRI](owlapy.model.IRI). 
On the other side, an instance of `IRI` is created by passing two arguments which are
the namespace of the ontology and the remainder 'child'. To declare this new class we need
an axiom of type `OWLDeclarationAxiom`. We simply pass the `child_class` to create an 
instance of this axiom. The final step is to add this axiom to the ontology using the 
[OWLOntologyManager](owlapy.owlready2.OWLOntologyManager). We use the `add_axiom` method
of the `manager` to add into the ontology
`onto` the axiom `child_class_declaration_axiom`.

#### Add a new Object Property / Data Property

The idea is the same as adding a new class. Instead of `OWLClass`, for object properties,
you can use the class [OWLObjectProperty](owlapy.model.OWLObjectProperty) and for data
properties you can use the class [OWLDataProperty](owlapy.model.OWLDataProperty).

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLObjectProperty
from owlapy.model import OWLDataProperty

# adding the object property 'hasParent'
hasParent_op = OWLObjectProperty(IRI('http://example.com/father#', 'hasParent'))
hasParent_op_declaration_axiom = OWLDeclarationAxiom(hasParent_op)
manager.add_axiom(onto, hasParent_op_declaration_axiom)

#adding the data property 'hasAge' 
hasAge_dp = OWLDataProperty(IRI('http://example.com/father#', 'hasAge'))
hasAge_dp_declaration_axiom = OWLDeclarationAxiom(hasAge_dp)
manager.add_axiom(onto, hasAge_dp_declaration_axiom)
```

See the [API documentation](owlapy.model) for more OWL entities that you can add as a declaration axiom.

#### Add an Assertion Axiom

To assign a class to a specific individual use the following code:

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLClassAssertionAxiom

individuals = list(onto.individuals_in_signature())
heinz = individuals[1] # get the 2nd individual in the list which is 'heinz'

class_assertion_axiom = OWLClassAssertionAxiom(heinz, child_class)

manager.add_axiom(onto, class_assertion_axiom)
```
We have used the previous method `individuals_in_signature()` to get all the individuals 
and converted them to a list, so we can access them by using indexes. In this example, we
want to assert a class axiom for the individual `heinz`. 
We have used the class `OWLClassAssertionAxiom`
where the first argument is the 'individual' `heinz` and the second argument is 
the 'class_expression'. As the class expression, we used the previously defined class 
`child_Class`. Finally, add the axiom by using `add_axiom` method of the [OWLOntologyManager](owlapy.owlready2.OWLOntologyManager).

Let's show one more example using a `OWLDataPropertyAssertionAxiom` to assign the age of 17 to
heinz. 

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLLiteral
from owlapy.model import OWLDataPropertyAssertionAxiom

literal_17 = OWLLiteral(17)
dp_assertion_axiom = OWLDataPropertyAssertionAxiom(heinz, hasAge_dp, literal_17)

manager.add_axiom(onto, dp_assertion_axiom)
```

[OWLLiteral](owlapy.model.OWLLiteral) is a class that represents the literal values in
Ontolearn. We have stored the integer literal value of '18' in the variable `literal_17`.
Then we construct the `OWLDataPropertyAssertionAxiom` by passing as the first argument, the 
individual `heinz`, as the second argument the data property `hasAge_dp`, and the third 
argument the literal value `literal_17`. Finally, add it to the ontology by using `add_axiom` 
method.

Check the [API documentation](owlapy.model) to see all the OWL 
assertion axioms that you can use.


#### Remove an Axiom

To remove an axiom you can use the `remove_axiom` method of the ontology manager as follows:

<!--pytest-codeblocks:cont-->
```python
manager.remove_axiom(onto,dp_assertion_axiom)
```
The first argument is the ontology you want to remove the axiom from and the second 
argument is the axiom you want to remove.

----------------------------------------------------------------------------

## Save an Ontology

If you modified an ontology, you may want to save it as a new file. To do this
you can use the `save_ontology` method of the [OWLOntologyManager](owlapy.owlready2.OWLOntologyManager).
It requires two arguments, the first is the ontology you want to save and The second
is the IRI of the new ontology.

<!--pytest-codeblocks:cont-->
```python
manager.save_ontology(onto, IRI.create('file:/' + 'test' + '.owl'))
```
 The above line of code will save the ontology `onto` in the file *test.owl* which will be
created in the same directory as the file you are running this code.

----------------------------------------------------------------------------

## Attaching a Reasoner

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
Besides the method for each reasoner which you can find in the API documentation, there 
are some extra convenience methods implemented in the class: [OWLReasonerEx](owlapy.ext.OWLReasonerEx).
Let us show some examples of these methods.

You can get all the object properties that an individual has by using the 
following method:

<!--pytest-codeblocks:cont-->
```python
anna = individuals[0] # get the 1st individual in the list of individuals which is 'anna'
object_properties = structural_reasoner.ind_object_properties(anna)
```

Now we can get the individuals of these object properties for 'anna'.

<!--pytest-codeblocks:cont-->
```python
for op in object_properties:
    object_properties_values = structural_reasoner.object_property_values(anna, op)
    for individual in object_properties_values:
        print(individual)
```

In this example we iterated over the `object_properties` and we use the reasoner
to get the values for each object property `op` of the individual `anna`. The values 
are individuals which we store in the variable `object_properties_values` and are 
printed in the end. The method `object_property_values` requires as the
first argument, an [OWLNamedIndividual](owlapy.model.OWLNamedIndividual) that is the subject of the object property values and 
the second argument an [OWLObjectProperty](owlapy.model.OWLObjectProperty) whose values are to be retrieved for the 
specified individual.  



