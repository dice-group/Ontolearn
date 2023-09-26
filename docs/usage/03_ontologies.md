# Ontologies
To get started with Structured Machine Learning, the first thing
required is an [Ontology](https://www.w3.org/TR/owl2-overview/) with
[Named Individuals](https://www.w3.org/TR/owl-syntax/#Named_Individuals).
In this guide we show the basics of working with ontologies in Ontolearn.

We will frequently **use a sample ontology** to give examples. You can find it in  
`KGs/father.owl`. Here is a hierarchical diagram that shows the classes and their
relationships:

             Thing
               |
            Person
           /   |   
       Male  Female

It contains only one object property which is _'hasChild'_ and in total there 
are six persons (individuals), of which four are male and two are female.


## Loading an Ontology

To load an ontology as well as to manage it, you will need an [OWLOntologyManager](ontolearn.owlapy.owlready2.OWLOntologyManager).
To load an ontology, use the following Python code:

```python
from ontolearn.owlapy.model import IRI
from ontolearn.owlapy.owlready2 import OWLOntologyManager_Owlready2

manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create("file://KGs/father.owl"))
```

First, we import the `IRI` class and a suitable OWLOntologyManager. To
load a file from our computer, we have to reference it with an
[IRI](https://tools.ietf.org/html/rfc3987). Secondly, we need the
Ontology Manager. Currently, Ontolearn contains one such manager: The
[OWLOntologyManager_Owlready2](ontolearn.owlapy.owlready2.OWLOntologyManager_Owlready2).

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

For more methods, refer to the [OWLOntology](ontolearn.owlapy.model.OWLOntology) class documentation.

## Modifying an Ontology

Axioms in ontology serve as the basis for defining the vocabulary of a domain and for 
making statements about the relationships between individuals and concepts in that domain.
They provide a formal and precise way to represent knowledge and allow for automated 
reasoning and inference. Axioms can be **added**, **modified**, or **removed** from an ontology, 
allowing the ontology to evolve and adapt as new knowledge is gained.

In Ontolearn we also have different axioms represented by different classes. You can check all
the axioms classes [here](ontolearn.owlapy.model). A few important axioms are:

- [OWLDeclarationAxiom](ontolearn.owlapy.model.OWLDeclarationAxiom)
- [OWLObjectPropertyAssertionAxiom](ontolearn.owlapy.model.OWLObjectPropertyAssertionAxiom)
- [OWLDataPropertyAssertionAxiom](ontolearn.owlapy.model.OWLObjectPropertyAssertionAxiom)
- [OWLClassAssertionAxiom](ontolearn.owlapy.model.OWLObjectPropertyAssertionAxiom)
- [OWLSubClassOfAxiom](ontolearn.owlapy.model.OWLSubClassOfAxiom)
- [OWLEquivalentClassesAxiom](ontolearn.owlapy.model.OWLEquivalentClassesAxiom)


#### Add a new Class

Let's suppose you want to add a new class in our example ontology `KGs/father.owl` 
It can be done as follows:

<!--pytest-codeblocks:cont-->

```python
from ontolearn.owlapy.model import OWLClass
from ontolearn.owlapy.model import OWLDeclarationAxiom

iri = IRI('http://example.com/father#', 'child')
child_class = OWLClass(iri)
child_class_declaration_axiom = OWLDeclarationAxiom(child_class)

manager.add_axiom(onto, child_class_declaration_axiom)
```
In this example, we added the class 'child' to the father.owl ontology.
Firstly we create an instance of [OWLClass](ontolearn.owlapy.model.OWLClass) to represent the concept 
of 'child' by using an [IRI](ontolearn.owlapy.model.IRI). 
On the other side, an instance of `IRI` is created by passing two arguments which are
the namespace of the ontology and the remainder 'child'. To declare this new class we need
an axiom of type `OWLDeclarationAxiom`. We simply pass the `child_class` to create an 
instance of this axiom. The final step is to add this axiom to the ontology using the 
[OWLOntologyManager](ontolearn.owlapy.owlready2.OWLOntologyManager). We use the `add_axiom` method
of the `manager` to add into the ontology
`onto` the axiom `child_class_declaration_axiom`.

#### Add a new Object Property / Data Property

The idea is the same as adding a new class. Instead of `OWLClass`, for object properties,
you can use the class [OWLObjectProperty](ontolearn.owlapy.model.OWLObjectProperty) and for data
properties you can use the class [OWLDataProperty](ontolearn.owlapy.model.OWLDataProperty).

<!--pytest-codeblocks:cont-->

```python
from ontolearn.owlapy.model import OWLObjectProperty
from ontolearn.owlapy.model import OWLDataProperty

# adding the object property 'hasParent'
hasParent_op = OWLObjectProperty(IRI('http://example.com/father#', 'hasParent'))
hasParent_op_declaration_axiom = OWLDeclarationAxiom(hasParent_op)
manager.add_axiom(onto, hasParent_op_declaration_axiom)

# adding the data property 'hasAge' 
hasAge_dp = OWLDataProperty(IRI('http://example.com/father#', 'hasAge'))
hasAge_dp_declaration_axiom = OWLDeclarationAxiom(hasAge_dp)
manager.add_axiom(onto, hasAge_dp_declaration_axiom)
```

See the [API documentation](ontolearn.owlapy.model) for more OWL entities that you can add as a declaration axiom.

#### Add an Assertion Axiom

To assign a class to a specific individual use the following code:

<!--pytest-codeblocks:cont-->

```python
from ontolearn.owlapy.model import OWLClassAssertionAxiom

individuals = list(onto.individuals_in_signature())
heinz = individuals[1]  # get the 2nd individual in the list which is 'heinz'

class_assertion_axiom = OWLClassAssertionAxiom(heinz, child_class)

manager.add_axiom(onto, class_assertion_axiom)
```
We have used the previous method `individuals_in_signature()` to get all the individuals 
and converted them to a list, so we can access them by using indexes. In this example, we
want to assert a class axiom for the individual `heinz`. 
We have used the class `OWLClassAssertionAxiom`
where the first argument is the 'individual' `heinz` and the second argument is 
the 'class_expression'. As the class expression, we used the previously defined class 
`child_Class`. Finally, add the axiom by using `add_axiom` method of the [OWLOntologyManager](ontolearn.owlapy.owlready2.OWLOntologyManager).

Let's show one more example using a `OWLDataPropertyAssertionAxiom` to assign the age of 17 to
heinz. 

<!--pytest-codeblocks:cont-->

```python
from ontolearn.owlapy.model import OWLLiteral
from ontolearn.owlapy.model import OWLDataPropertyAssertionAxiom

literal_17 = OWLLiteral(17)
dp_assertion_axiom = OWLDataPropertyAssertionAxiom(heinz, hasAge_dp, literal_17)

manager.add_axiom(onto, dp_assertion_axiom)
```

[OWLLiteral](ontolearn.owlapy.model.OWLLiteral) is a class that represents the literal values in
Ontolearn. We have stored the integer literal value of '18' in the variable `literal_17`.
Then we construct the `OWLDataPropertyAssertionAxiom` by passing as the first argument, the 
individual `heinz`, as the second argument the data property `hasAge_dp`, and the third 
argument the literal value `literal_17`. Finally, add it to the ontology by using `add_axiom` 
method.

Check the [API documentation](ontolearn.owlapy.model) to see all the OWL 
assertion axioms that you can use.


#### Remove an Axiom

To remove an axiom you can use the `remove_axiom` method of the ontology manager as follows:

<!--pytest-codeblocks:cont-->
```python
manager.remove_axiom(onto,dp_assertion_axiom)
```
The first argument is the ontology you want to remove the axiom from and the second 
argument is the axiom you want to remove.


## Save an Ontology

If you modified an ontology, you may want to save it as a new file. To do this
you can use the `save_ontology` method of the [OWLOntologyManager](ontolearn.owlapy.owlready2.OWLOntologyManager).
It requires two arguments, the first is the ontology you want to save and The second
is the IRI of the new ontology.

<!--pytest-codeblocks:cont-->
```python
manager.save_ontology(onto, IRI.create('file:/' + 'test' + '.owl'))
```
 The above line of code will save the ontology `onto` in the file *test.owl* which will be
created in the same directory as the file you are running this code.


## Worlds

Owlready2 stores every triples in a ‘World’ object, and it can handle several Worlds in parallel.
Owlready2 uses an optimized quadstore to store the world. Each world object is stored in a separate quadstore and 
by default the quadstore is stored in memory,
but it can also be stored in an SQLite3 file. The method `save_world()` of the ontology manager does the latter.
When an _OWLOntologyManager_ object is created, a new world is also created as an attribute of the manager.
By calling the method `load_ontology(iri)` the ontology is loaded to this world. 

It possible to create several isolated “worlds”, sometimes
called “universe of speech”. This makes it possible in particular to load
the same ontology several times, independently, that is to say, without
the modifications made on one copy affecting the other copy. Sometimes the need to [isolate an ontology](07_reasoning_details.md#isolated-world) 
arise. What that means is that you can have multiple reference of the same ontology in different
worlds.

-------------------------------------------------------------------------------------

In the next guide we will explore the [KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase) class that is needed to 
run a concept learner.




