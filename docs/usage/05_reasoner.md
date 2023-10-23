# Reasoners

To validate facts about statements in the ontology (and thus
also for the Structured Machine Learning task), the help of a reasoner
component is required.

For this guide we will also consider the 'Father' ontology:

```python
from ontolearn.owlapy.owlready2 import OWLOntologyManager_Owlready2

manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create("KGs/father.owl"))
```

In our Ontolearn library, we provide several **reasoners** to choose
from. Currently, there are the following reasoners available: 

-  [**OWLReasoner_Owlready2**](ontolearn.owlapy.owlready2.OWLReasoner_Owlready2)

    Or differently Structural Owlready2 Reasoner, is the base reasoner in Ontolearn. The functionalities
    of this reasoner are limited. It does not provide full reasoning in _ALCH_. Furthermore,
    it has no support for instances of complex class expressions, which is covered by the
    other reasoners (CCEI and FIC). We recommend to use the other reasoners for any reasoning tasks.

    **Initialization:**

    ```python
    from ontolearn.owlapy.owlready2 import OWLReasoner_Owlready2
    
    structural_reasoner = OWLReasoner_Owlready2(onto)
    ```

    The structural reasoner requires an ontology ([OWLOntology](ontolearn.owlapy.model.OWLOntology)).
    The second argument is `isolate` argument which isolates the world (therefore the ontology) where the reasoner is 
    performing the reasoning. More on that on _[Reasoning Details](07_reasoning_details.md#isolated-world)_. 
    The rest of the arguments `use_triplestore` and `triplestore_address` are used in case you want to
    retrieve instances from a triplestore (go to 
    [#use-triplestore-to-retrieve-instances](05_reasoner.md#use-triplestore-to-retrieve-instances) for details).
    


- [**OWLReasoner_Owlready2_ComplexCEInstances**](ontolearn.owlapy.owlready2.complex_ce_instances.OWLReasoner_Owlready2_ComplexCEInstances) **(CCEI)**

    Can perform full reasoning in _ALCH_ due to the use of HermiT/Pellet and provides support for
    complex class expression instances (when using the method `instances`). CCEI is more useful when 
    your main goal is reasoning over the ontology.

    **Initialization:**

    ```python
    from ontolearn.owlapy.owlready2.complex_ce_instances import OWLReasoner_Owlready2_ComplexCEInstances
    from ontolearn.owlapy.owlready2 import BaseReasoner_Owlready2
    
    ccei_reasoner = OWLReasoner_Owlready2_ComplexCEInstances(onto, BaseReasoner_Owlready2.HERMIT,
                                                             infer_property_values = True)
    ```
    
    CCEI requires an ontology and a base reasoner of type [BaseReasoner_Owlready2](ontolearn.owlapy.owlready2.BaseReasoner_Owlready2)
    which is just an enumeration with two possible values: `BaseReasoner_Owlready2.HERMIT` and `BaseReasoner_Owlready2.PELLET`.
    You can set the `infer_property_values` argument to `True` if you want the reasoner to infer
    property values. `infer_data_property_values` is an additional argument when the base reasoner is set to 
    `BaseReasoner_Owlready2.PELLET`. The rest of the arguments `isolated`, `use_triplestore` and `triplestore_address` 
    are inherited from the base class.


- [**OWLReasoner_FastInstanceChecker**](ontolearn.owlapy.fast_instance_checker.OWLReasoner_FastInstanceChecker) **(FIC)**

    FIC also provides support for complex class expression but the rest of the methods are the same as in 
    the base reasoner.
    It has a cache storing system that allows for faster execution of some reasoning functionalities. Due to this 
    feature, FIC is more appropriate to be used in concept learning.

    **Initialization:**

    ```python
    from ontolearn.owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
    
    fic_reasoner = OWLReasoner_FastInstanceChecker(onto, structural_reasoner, property_cache = True,
                                                   negation_default = True, sub_properties = False)
    ```
    Besides the ontology, FIC requires a base reasoner to delegate any reasoning tasks not covered by it.
    This base reasoner 
    can be any other reasoner in Ontolearn. `property_cache` specifies whether to cache property values. This 
    requires more memory, but it speeds up the reasoning processes. If `negation_default` argument is set 
    to `True` the missing facts in the ontology means false. The argument
    `sub_properties` is another boolean argument to specify whether you want to take sub properties in consideration
    for `instances()` method.

## Usage of the Reasoner
All the reasoners available in the Ontolearn library inherit from the
class: [OWLReasonerEx](ontolearn.owlapy.ext.OWLReasonerEx). This class provides some 
extra convenient methods compared to its base class [OWLReasoner](ontolearn.owlapy.model.OWLReasoner), which is an 
abstract class.
Further in this guide, we use 
[OWLReasoner_Owlready2_ComplexCEInstances](ontolearn.owlapy.owlready2.complex_ce_instances).
to show the capabilities of a reasoner implemented in Ontolearn.

To give examples we consider the _father_ dataset. 
If you are not already familiar with this small dataset,
you can find an overview of it [here](03_ontologies.md).


## Class Reasoning

Using an [OWLOntology](ontolearn.owlapy.model.OWLOntology) you can list all the classes in the signature, 
but a reasoner can give you more than that. You can get the subclasses, superclasses or the 
equivalent classes of a class in the ontology:

<!--pytest-codeblocks:cont-->

```python
from ontolearn.owlapy.model import OWLClass
from ontolearn.owlapy.model import IRI

namespace = "http://example.com/father#"
male = OWLClass(IRI(namespace, "male"))

male_super_classes = ccei_reasoner.super_classes(male)
male_sub_classes = ccei_reasoner.sub_classes(male)
male_equivalent_classes = ccei_reasoner.equivalent_classes(male)
```

We define the _male_ class by creating an [OWLClass](ontolearn.owlapy.model.OWLClass) object. The 
methods `super_classes` and `sub_classes` have 2 more boolean arguments: `direct` and `only_named`. 
If `direct=True` then only the direct classes in the 
hierarchy will be returned, else it will return every class in the hierarchy depending 
on the method(sub_classes or super_classes).
By default, its value is _False_. 
The next argument `only_named` specifies whether you want
to show only named classes or complex classes as well. By default, its value is _True_ which 
means that it will return only the named classes.

>**NOTE**: The extra arguments `direct` and `only_named` are also used in other methods that reason
upon the class, object property, or data property hierarchy.

You can get all the types of a certain individual using `types` method:

<!--pytest-codeblocks:cont-->

```python
anna = list( onto.individuals_in_signature()).pop()

anna_types = ccei_reasoner.types(anna)
```

We retrieve _anna_ as the first individual on the list of individuals 
of the 'Father' ontology. The `type` method only returns named classes.


## Object Properties and Data Properties Reasoning
Ontolearn reasoners offers some convenient methods for working with object properties and 
data properties. Below we show some of them, but you can always check all the methods in the 
[OWLReasoner_Owlready2_ComplexCEInstances](ontolearn.owlapy.owlready2.complex_ce_instances)
class documentation. 

You can get all the object properties that an individual has by using the 
following method:

<!--pytest-codeblocks:cont-->
```python
anna = individuals[0] 
object_properties = ccei_reasoner.ind_object_properties(anna)
```
In this example, `object_properties` contains all the object properties
that _anna_ has, which in our case would only be _hasChild_.
Now we can get the individuals of this object property for _anna_.

<!--pytest-codeblocks:cont-->
```python
for op in object_properties:
    object_properties_values = ccei_reasoner.object_property_values(anna, op)
    for individual in object_properties_values:
        print(individual)
```

In this example we iterated over the `object_properties`, assuming that there
are more than 1, and we use the reasoner
to get the values for each object property `op` of the individual `anna`. The values 
are individuals which we store in the variable `object_properties_values` and are 
printed in the end. The method `object_property_values` requires as the
first argument, an [OWLNamedIndividual](ontolearn.owlapy.model.OWLNamedIndividual) that is the subject of the object property values and 
the second argument an [OWLObjectProperty](ontolearn.owlapy.model.OWLObjectProperty) whose values are to be retrieved for the 
specified individual.  

> **NOTE:** You can as well get all the data properties of an individual in the same way by using 
`ind_data_properties` instead of `ind_object_properties` and `data_property_values` instead of 
`object_property_values`. Keep in mind that `data_property_values` returns literal values 
(type of [OWLLiteral](ontolearn.owlapy.model.OWLLiteral)).

In the same way as with classes, you can also get the sub object properties or equivalent object properties.

<!--pytest-codeblocks:cont-->

```python
from ontolearn.owlapy.model import OWLObjectProperty

hasChild = OWLObjectProperty(IRI(namespace, "hasChild"))

equivalent_to_hasChild = ccei_reasoner.equivalent_object_properties(hasChild)
hasChild_sub_properties = ccei_reasoner.sub_object_properties(hasChild)
```

In case you want to get the domains and ranges of an object property use the following:

<!--pytest-codeblocks:cont-->
```python
hasChild_domains = ccei_reasoner.object_property_domains(hasChild)
hasChild_ranges = ccei_reasoner.object_property_ranges(hasChild)
```

> **NOTE:** Again, you can do the same for data properties but instead of the word 'object' in the 
> method name you should use 'data'.


## Find Instances

The method `instances` is a very convenient method. It takes only 1 argument that is basically
a class expression and returns all the individuals belonging to that class expression. In Ontolearn 
we have implemented a Python class for each type of class expression.
The argument is of type [OWLClassExpression](ontolearn.owlapy.model.OWLClassExpression).

Let us now show a simple example by finding the instances of the class _male_ and printing them:

<!--pytest-codeblocks:cont-->
```python
male_individuals = ccei_reasoner.instances(male)
for ind in male_individuals:
    print(ind)
```

### Use triplestore to retrieve instances

Instead of going through nodes using expensive computation resources why not just make use of the
efficient approach of querying a triplestore using SPARQL queries. We have brought this 
functionality to Ontolearn, and we take care of the conversion part behind the scene. You can use the `instances` 
method as usual. Let's see what it takes to make use of it.

First of all you need a server which should host the triplestore for your ontology. If you don't
already have one, see [Loading and launching a triplestore](#loading-and-launching-a-triplestore) below.

As we mentioned earlier, OWLReasoner has two arguments for enabling triplestore 
retrieval: 
- `use_triplestore` is a boolean argument. Setting this to True tells `instances` that it
should query the triplestore hosted on the server address specified by the following argument:
- `triplestore_address` is a string that contains the URL of the triplestore host.

For example below we are initializing a reasoner that uses the triplestore in the address
`http://localhost:3030/path/` and retrieving the male instances.
```python
reasoner = OWLReasoner_Owlready2(onto,use_triplestore=True, triplestore_address="http://localhost:3030/path/sparql")
males = reasoner.instances(male, direct=False)
```

> :warning: **You have to make sure** that the ontology that is loaded in the triplestore is
> exactly the same as the ontology that is being used by the reasoner, otherwise you may
> encounter inconsistent results.

> :warning: When using triplestore the `instances` method **will default to the base
> implementation**. This means that no matter which type of reasoner you are using,
> the results will be always the same for a given class expression.

> :warning: **You cannot pass these arguments directly to FIC constructor.** 
> Because of the way it is implemented, if the base reasoner is set to use triplestore,
> then FIC's `instances` method will default to the base reasoners implementation that uses 
> triplestore for instance retrieval.

#### Loading and launching a triplestore

We will provide a simple approach to load and launch a triplestore in a local server. For this,
we will be using _apache-jena_ and _apache-jena-fuseki_. As a prerequisite you need
JDK 11 or higher and if you are on Windows, you need [Cygwin](https://www.cygwin.com/). In case of
issues or any further reference please visit the official page of [Apache Jena](https://jena.apache.org/index.html) 
and check the documentation under "Triple Store".

Having that said, let us now load and launch a triplestore on the "Father" ontology:

Open a terminal window and make sure you are in the root directory. Create a directory to 
store the files for Fuseki server:

```shell
mkdir Fuseki && cd Fuseki
```
Install _apache-jena_ and _apache-jena-fuseki_. We will use version 4.7.0.

```shell
# install Jena
wget https://archive.apache.org/dist/jena/binaries/apache-jena-4.7.0.tar.gz
#install Jena-Fuseki
wget https://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-4.7.0.tar.gz
```

Unzip the files:

```shell
tar -xzf apache-jena-fuseki-4.7.0.tar.gz
tar -xzf apache-jena-4.7.0.tar.gz
```

Make a directory for our 'father' database inside jena-fuseki:

```shell
mkdir -p apache-jena-fuseki-4.7.0/databases/father/
```

Now just load the 'father' ontology using the following command:

```shell
cd .. && Fuseki/apache-jena-4.7.0/bin/tdb2.tdbloader --loader=parallel --loc Fuseki/apache-jena-fuseki-4.7.0/databases/father/ KGs/father.owl
```

Launch the server, and it will be waiting eagerly for your queries.

```shell
cd Fuseki/apache-jena-fuseki-4.7.0 && java -Xmx4G -jar fuseki-server.jar --tdb2 --loc=databases/father /father
```

Notice that we launched the database found in `Fuseki/apache-jena-fuseki-4.7.0/databases/father` to the path `/father`.
By default, jena-fuseki runs on port 3030 so the full URL would be: `http://localhost:3030/father`. When 
you pass this url to `triplestore_address` argument of the reasoner, you have to add the
`/sparql` sub-path indicating to the server that we are querying via SPARQL queries. Full path now should look like:
`http://localhost:3030/father/sparql`.

Create a reasoner that uses this URL and retrieve the desired instances:

```python
father_reasoner = OWLReasoner_Owlready2(onto, use_triplestore=True, triplestore_address="http://localhost:3030/father/sparql")
```

-----------------------------------------------------------------------

In this guide we covered the main functionalities of the reasoners in Ontolearn. More
details are provided in _[Reasoning Details](07_reasoning_details.md)_.

Since we have now covered all the basics, on the next guide
you will see how to use concept learners to learn class expressions in a 
knowledge base for a certain learning problem.

