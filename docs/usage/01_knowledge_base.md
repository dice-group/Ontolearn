# Working with Knowledge Bases

To get started with Structured Machine Learning, the first thing
required is an [Ontology](https://www.w3.org/TR/owl2-overview/) with
[Named Individuals](https://www.w3.org/TR/owl-syntax/#Named_Individuals).
In the literature, an Ontology is part of a
Knowledge Base. In Ontolearn we represent a knowledge base 
by the class [KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase) which contains two main class attributes, 
an ontology represented by the class [OWLOntology](owlapy.model.OWLOntology) and a reasoner represented by the class [OWLReasoner](owlapy.model.OWLReasoner).
It also contains a Hierarchy generator as well as other Ontology-related attributes required for the Structured Machine Learning library.

An instance of `KnowledgeBase` is required to run a learning
algorithm.

We will frequently **use a sample ontology** to give examples. You can find in  
the `KGs/father.owl` file. Here is a hierarchical diagram that shows the classes and their
relationships:

             Thing
               |
            Person
           /   |   
       Male  Female

It contains only one object property which is _'hasChild'_ and in total there 
are six persons (individuals), of which four are male and two are female.



----------------------------------------------------------------------------

## Create an instance
There are different ways to initialize an object of type `KnowledgeBase` 
because of the optional arguments. Let us show the most basic ones.

We consider that you have already an OWL ontology (containing *.owl* extension).

The simplest way is to use the path of your _.owl_ file to initialize it as follows:

```python 
from ontolearn.knowledge_base import KnowledgeBase

kb = KnowledgeBase(path="file://KGs/father.owl")
```

Another way would be to use an instance of `OWLOntology` 
and an instance of `OWLReasoner`:
<!--pytest-codeblocks:cont-->
```python

# onto: OWLOntology
# reas: OWLReasoner

kb = KnowledgeBase(ontology= onto, reasoner=reas)
```
You can read more about `OWLOntology` and `OWLReasoner` [here](03_ontologies.md).


----------------------------------------------------------------------------

## Ignore concepts
To avoid trivial solutions sometimes you need to ignore specific concepts.

Suppose that we have the class "Father" for the individuals in 
our example ontology `KGs/father.owl`. If we are trying to learn 
the concept of a 'Father' then we need to ignore this concept
before fitting a model.
It can be done as follows:

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLClass
from owlapy.model import IRI

iri = IRI('http://example.com/father#', 'Father') 
father_concept = OWLClass(iri)
concepts_to_ignore = {father_concept} # you can add more than 1

new_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
```
In this example, we have created an instance of [OWLClass](owlapy.model.OWLClass) by using an [IRI](owlapy.model.IRI). 
On the other side, an instance of `IRI` is created by passing two parameters which are
the namespace of the ontology and the remainder 'Father'.

----------------------------------------------------------------------------

## Accessing individuals
You may need to work with individuals of a knowledge base. 
We cover different ways of accessing them.

Let us give a  simple example of how to get the individuals that
are classified by an [OWLClassExpression](owlapy.model.OWLClassExpression). As a class expression, we will simply use the 
concept 'male'.

<!--pytest-codeblocks:cont-->
```python
NS = 'http://example.com/father#'
male_concept = OWLClass(IRI(NS,'male'))

male_individuals = kb.individuals(male_concept)
```
`male_individuals` will contain all the individuals that are classified as
'male'.
Keep in mind that `OWLClass` inherit from `OWLClassExpression`. 


If you don't specify a class expression than this method returns all the individuals:
<!--pytest-codeblocks:cont-->
```python
all_individuals = kb.individuals()
```

You can as well get all the individuals using:
<!--pytest-codeblocks:cont-->
```python
all_individuals_set  = kb.all_individuals_set()
```
The difference is that `individuals()` return type is `Iterable[OWLNamedIndividual]` 
and `all_individuals_set()` return type is `frozenset(OWLNamedIndividual)`.

In case you need your result as frozenset, `individual_set` method is a better option
then the `individuals` method:
<!--pytest-codeblocks:cont-->
```python
male_individuals_set = kb.individuals_set(male_concept)
```

Or you can even combine both methods:
<!--pytest-codeblocks:cont-->
```python
male_individuals_set = kb.individuals_set(male_individuals)
```

----------------------------------------------------------------------------

## Evaluate a concept

`KnowledgeBase` class offers methods to evaluate a concept, but to do that
you will need a **learning problem** for the concept you want to evaluate.
You may need to check the guidelines for [defining the learning problem](02_learning_problem.md)
before continuing. 

We will consider that you already have constructed an 
instance of [PosNegLPStandard](ontolearn.learning_problem.PosNegLPStandard) class. To evaluate a concept you will 
first need the encoded learning problem.
<!--pytest-codeblocks:cont-->
```python
# lp: PosNegLPStandard
encoded_lp = kb.encode_learning_problem(lp)
```
For more details about this method please refer to the API.

To evaluate a concept you can use the method `evaluate_concept`. To do that 
you will need 3 properties:

1. a concept to evaluate: [OWLClassExpression](owlapy.model.OWLClassExpression)
2. a quality metric: [AbstractScorer](ontolearn.abstracts.AbstractScorer)
3. the encoded learning problem: [EncodedLearningProblem](ontolearn.learning_problem.EncodedPosNegLPStandard)

Let's suppose that the class expression `(¬female) ⊓ (∃ hasChild.⊤)` was generated by [CELOE](https://ontolearn-docs-dice-group.netlify.app/usage/03_algorithm.html#)
for the concept of 'Father' and is stored in the variable `hypothesis`. 
You can evaluate this class expression as follows:
<!--pytest-codeblocks:cont-->
```python
from ontolearn.metrics import F1

evaluated_concept = kb.evaluate_concept(hypothesis, F1(), encoded_lp)
```
In this example we used F1-score to evaluate the hypothesis, but you can also 
use Accuracy by importing it first and then replacing `F1()` with `Accuracy()`.

You can now:

- Print the quality:
    <!--pytest-codeblocks:cont-->
    ```python
    print(evaluated_concept.q) # 1.0
    ```

- Print the set of individuals covered by the hypothesis:
    <!--pytest-codeblocks:cont-->
    ```python
    for ind in evaluated_concept.inds:
        print(ind) 
  
    # OWLNamedIndividual(http://example.com/father#markus)
    # OWLNamedIndividual(http://example.com/father#martin)
    # OWLNamedIndividual(http://example.com/father#stefan)
    ```
- Print the amount of them:
    <!--pytest-codeblocks:cont-->
    ```python
    print(evaluated_concept.ic) # 3
    ```

Check all the methods of the [KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase) class in the documentation.
