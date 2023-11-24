# Reasoners

To validate facts about statements in the ontology (and thus
also for the Structured Machine Learning task), the help of a reasoner
component is required.

For this guide we will also consider the 'Father' ontology that we slightly described [here](03_ontologies.md):

```python
from ontolearn.base import OWLOntologyManager_Owlready2

manager = OWLOntologyManager_Owlready2()
onto = manager.load_ontology(IRI.create("KGs/father.owl"))
```

In our Ontolearn library, we provide several **reasoners** to choose
from. Currently, there are the following reasoners available: 

- [**OWLReasoner_Owlready2**](ontolearn.base.OWLReasoner_Owlready2)

    Or differently Structural Owlready2 Reasoner, is the base reasoner in Ontolearn. The functionalities
  of this reasoner are limited. It does not provide full reasoning in _ALCH_. Furthermore,
  it has no support for instances of complex class expressions, which is covered by the
  other reasoners (CCEI and FIC). We recommend to use the other reasoners for any reasoning tasks.

    **Initialization:**

   ```python
   from ontolearn.base import OWLReasoner_Owlready2
   
   structural_reasoner = OWLReasoner_Owlready2(onto)
   ```

    The structural reasoner requires an ontology ([OWLOntology](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L3064)).
  The second argument is `isolate` argument which isolates the world (therefore the ontology) where the reasoner is
  performing the reasoning. More on that on _[Reasoning Details](07_reasoning_details.md#isolated-world)_.
  The remaining argument, `triplestore_address`, is used in case you want to
  retrieve instances from a triplestore (go to 
    [_Using a Triplestore for Reasoning Tasks_](#using-a-triplestore-for-reasoning-tasks) for details).
    


- [**OWLReasoner_Owlready2_ComplexCEInstances**](ontolearn.base.complex_ce_instances.OWLReasoner_Owlready2_ComplexCEInstances) **(CCEI)**

    Can perform full reasoning in _ALCH_ due to the use of HermiT/Pellet and provides support for
  complex class expression instances (when using the method `instances`). CCEI is more useful when
  your main goal is reasoning over the ontology.

    **Initialization:**

    ```python
    from ontolearn.base.complex_ce_instances import OWLReasoner_Owlready2_ComplexCEInstances
    from ontolearn.base import BaseReasoner_Owlready2
    
    ccei_reasoner = OWLReasoner_Owlready2_ComplexCEInstances(onto, BaseReasoner_Owlready2.HERMIT,
                                                             infer_property_values = True)
    ```
    
    CCEI requires an ontology and a base reasoner of type [BaseReasoner_Owlready2](ontolearn.base.BaseReasoner_Owlready2)
    which is just an enumeration with two possible values: `BaseReasoner_Owlready2.HERMIT` and `BaseReasoner_Owlready2.PELLET`.
  You can set the `infer_property_values` argument to `True` if you want the reasoner to infer
  property values. `infer_data_property_values` is an additional argument when the base reasoner is set to 
    `BaseReasoner_Owlready2.PELLET`. The rest of the arguments `isolated` and `triplestore_address` 
    are inherited from the base class.


- [**OWLReasoner_FastInstanceChecker**](ontolearn.base.fast_instance_checker.OWLReasoner_FastInstanceChecker) **(FIC)**

    FIC also provides support for complex class expression but the rest of the methods are the same as in
  the base reasoner.
  It has a cache storing system that allows for faster execution of some reasoning functionalities. Due to this
  feature, FIC is more appropriate to be used in concept learning.

    **Initialization:**

    ```python
    from ontolearn.base.fast_instance_checker import OWLReasoner_FastInstanceChecker
    
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
class: [OWLReasonerEx](ontolearn.base.ext.OWLReasonerEx). This class provides some 
extra convenient methods compared to its base class [OWLReasoner](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L3322), which is an 
abstract class.
Further in this guide, we use 
[OWLReasoner_Owlready2_ComplexCEInstances](ontolearn.base.OWLReasoner_Owlready2_ComplexCEInstances).
to show the capabilities of a reasoner implemented in Ontolearn.

To give examples we consider the _father_ dataset. 
If you are not already familiar with this small dataset,
you can find an overview of it [here](03_ontologies.md).


## Class Reasoning

Using an [OWLOntology](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L3064) you can list all the classes in the signature, 
but a reasoner can give you more than that. You can get the subclasses, superclasses or the 
equivalent classes of a class in the ontology:

<!--pytest-codeblocks:cont-->

```python
from owlapy.model import OWLClass
from owlapy.model import IRI

namespace = "http://example.com/father#"
male = OWLClass(IRI(namespace, "male"))

male_super_classes = ccei_reasoner.super_classes(male)
male_sub_classes = ccei_reasoner.sub_classes(male)
male_equivalent_classes = ccei_reasoner.equivalent_classes(male)
```

We define the _male_ class by creating an [OWLClass](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L209) object. The 
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
[OWLReasoner_Owlready2_ComplexCEInstances](ontolearn.base.complex_ce_instances.OWLReasoner_Owlready2_ComplexCEInstances)
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
first argument, an [OWLNamedIndividual](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L917) that is the subject of the object property values and 
the second argument an [OWLObjectProperty](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L380C7-L380C24) whose values are to be retrieved for the 
specified individual.  

> **NOTE:** You can as well get all the data properties of an individual in the same way by using 
`ind_data_properties` instead of `ind_object_properties` and `data_property_values` instead of 
`object_property_values`. Keep in mind that `data_property_values` returns literal values 
(type of [OWLLiteral](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L1128)).

In the same way as with classes, you can also get the sub object properties or equivalent object properties.

<!--pytest-codeblocks:cont-->

```python
from owlapy.model import OWLObjectProperty

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
a class expression and returns all the individuals belonging to that class expression. In Owlapy 
we have implemented a Python class for each type of class expression.
The argument is of type [OWLClassExpression](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L209Expression).

Let us now show a simple example by finding the instances of the class _male_ and printing them:

<!--pytest-codeblocks:cont-->
```python
male_individuals = ccei_reasoner.instances(male)
for ind in male_individuals:
    print(ind)
```

### Using a Triplestore for Reasoning Tasks

As we mentioned earlier, OWLReasoner has an argument for enabling triplestore querying:
- `triplestore_address` - a string that contains the URL of the triplestore host/server. If specified, it tells
the reasoner that for its operations it should query the triplestore hosted on the given address.

Triplestores are known for their efficiency in retrieving data, and they can be queried using SPARQL.
Making this functionality available for reasoners in Ontolearn makes it possible to use concept learners that
fully operates in datasets hosted on triplestores. Although that is the main goal, the reasoner can be used
independently for reasoning tasks. Therefore, you can initialize a reasoner to use triplestore as follows:

```python
from ontolearn.base import OWLReasoner_Owlready2

reasoner = OWLReasoner_Owlready2(onto, triplestore_address="http://some_domain/some_path/sparql")
```

Now you can use the reasoner methods as you would normally do:

```python
# Retrieving the male instances using `male` variable that we declared earlier
males = reasoner.instances(male, direct=False)
```

**Some important notice are given below:**

> Not all the methods of the reasoner are implemented to use triplestore but the main methods 
> such as 'instance' and those used to get sub/super classes/properties will work just fine.

> **You cannot pass the triplestore argument directly to FIC constructor.** 
> Because of the way it is implemented, if the base reasoner is set to use triplestore,
> then FIC's is considered to using triplestore.

> When using triplestore all methods, including `instances` method **will default to the base
> implementation**. This means that no matter which type of reasoner you are using, the results will be always 
> the same.

-----------------------------------------------------------------------

In this guide we covered the main functionalities of the reasoners in Ontolearn. More
details are provided in _[Reasoning Details](07_reasoning_details.md)_.

Since we have now covered all the basics, on the next guide
you will see how to use concept learners to learn class expressions in a 
knowledge base for a certain learning problem.

