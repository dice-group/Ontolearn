# Knowledge Bases

In Ontolearn we represent a knowledge base 
by the class [KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase) which contains two main class attributes, 
an ontology [OWLOntology](owlapy.model.OWLOntology)
and a reasoner [OWLReasoner](owlapy.model.OWLReasoner).
It also contains the class and properties hierarchy as well as other
Ontology-related attributes required for the Structured Machine Learning library.


## Knowledge Base vs Ontology

These terms may be used interchangeably sometimes but in Ontolearn they are not the same thing,
although they share a lot of similarities. An ontology in Ontolearn, as we explained in the
[previous guide](03_ontologies.md) is the object where we load the OWL 2.0 ontologies from
a _.owl_ file containing the ontology in an RDF/XML or OWL/XML format. On the other side
a KnowledgeBase is a class which combines an ontology and a reasoner together. Therefore,
differently from the ontology you can use methods that require reasoning. You can check 
the methods for each in the links below:

- [KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase)
- [OWLOntology](owlapy.model.OWLOntology)

In summary:

- An instance of `KnowledgeBase` contains an ontology and a reasoner and 
is required to run a learning algorithm.

- The ontology object can load an OWL 2.0 ontology,
be modified using the ontology manager and saved.

- Although they have some similar functionalities, there are a lot of other distinct 
functionalities that each of them has.


## Create an Object of KnowledgeBase

Let us show how you can initialize an object of `KnowledgeBase`.
We consider that you have already an OWL 2.0 ontology (containing *.owl* extension).

The simplest way is to use the path of your _.owl_ file as follows:

```python 
from ontolearn.knowledge_base import KnowledgeBase

kb = KnowledgeBase(path="file://KGs/father.owl")
```

What happens in the background is that the ontology located in this path will be loaded
in the `OWLOntology` object of `kb` as done [here](03_ontologies.md#loading-an-ontology).

In our recent version you can also initialize a knowledge base using a dataset hosted in a triplestore.
Since that knowledge base is mainly used for executing a concept learner, we cover that matter more in depth 
in _[Use Triplestore Knowledge Base](06_concept_learners.md#use-triplestore-knowledge-base)_ 
section of _[Concept Learning](06_concept_learners.md)_.

## Ignore Concepts

During concept learning which we describe later, you may need to 
avoid trivial solutions from being learned. So in Ontolearn you
have the opportunity to ignore specific concepts. Since we pass a `KnowledgeBase`
object to the concept learner, we set this ignored concept using the method 
`ignore_and_copy` of the `KnowledgeBase` class.

We don't have such concept in our example ontology `KGs/father.owl` but suppose that 
there is a class(concept) "Father" that we want to ignore, because we are trying
to learn this a meaningful class expression for 'Father' using other classes(e.g. male, female, ∃ hasChild.⊤... ).
So we need to ignore this concept before fitting a model (model fitting is covered in [concept learning](06_concept_learners.md)).
It can be done as follows:

<!--pytest-codeblocks:cont-->

```python
from owlapy.model import OWLClass
from owlapy.model import IRI

iri = IRI('http://example.com/father#', 'Father')
father_concept = OWLClass(iri)
concepts_to_ignore = {father_concept}  # you can add more than 1

new_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
```
In this example, we have created an instance of [OWLClass](owlapy.model.OWLClass) by using an [IRI](owlapy.model.IRI). 
On the other side, an instance of `IRI` is created by passing two parameters which are
the namespace of the ontology and the remainder 'Father'.

## Accessing Individuals

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
Note that the namespace has to match the Namespace/IRI that is defined
in the Ontology document.

`male_individuals` will contain all the individuals of type 'male'.
Keep in mind that `OWLClass` inherit from `OWLClassExpression`. Depending on 
the reasoner that the `kb` object is using the results may differ slightly but in
case of a small dataset like the one we are using for this example, the results do not change.


If you don't give any argument than this method returns all the individuals in the ontology:
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


## Evaluate a Concept

When using a concept learner, the generated concepts (class expressions) for a certain learning problem
need to be evaluated to see the performance. 
To do that you can use the method `evaluate_concept` of `KnowledgeBase`. It requires the following arguments:

1. a concept to evaluate: [OWLClassExpression](owlapy.model.OWLClassExpression)
2. a quality metric: [AbstractScorer](ontolearn.abstracts.AbstractScorer)
3. the encoded learning problem: [EncodedLearningProblem](ontolearn.learning_problem.EncodedPosNegLPStandard)

The evaluation should be done for the learning problem that you used to generate the 
concept. The main result of the evaluation is the quality score describing how well the generated
concept is doing on the job of classifying the positive individuals. The concept learners do this 
process automatically.

### Construct a learning problem

To evaluate a concept you need a learning problem. Firstly, we create two simple sets containing 
the positive and negative examples for the concept of 'Father'. Our positive examples 
(individuals to describe) are stefan, markus, and martin. And our negative examples
(individuals to not describe) are heinz, anna, and michelle.

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLNamedIndividual

positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}

negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}
```

Now the learning problem can be captured in its respective object, the
[positive-negative standard learning problem](ontolearn.learning_problem.PosNegLPStandard) and 
encode it using the method `encode_learning_problem` of `KnowledgeBase`:

<!--pytest-codeblocks:cont-->
```python
from ontolearn.learning_problem import PosNegLPStandard

lp = PosNegLPStandard(pos=positive_examples, neg=negative_examples)

encoded_lp = kb.encode_learning_problem(lp)
```

Now that we have an encoded learning problem, we need a concept to evaluate.

### Construct a concept

Suppose that the class expression `(¬female) ⊓ (∃ hasChild.⊤)` 
was generated by [CELOE](ontolearn.concept_learner.CELOE)
for the concept of 'Father'. We will see how that can happen later
but for now we let's construct this class expression manually:

<!--pytest-codeblocks:cont-->
```python
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom , OWLObjectIntersectionOf

female = OWLClass(IRI(NS,'female'))
not_female = kb.generator.negation(female)
has_child_property = OWLObjectProperty(IRI(NS, "hasChild"))
thing = OWLClass(IRI('http://www.w3.org/2002/07/owl#', 'Thing'))
exist_has_child_T = OWLObjectSomeValuesFrom(property=has_child_property, filler=thing)

concept_to_test = OWLObjectIntersectionOf([not_female, exist_has_child_T])
```

`kb` has an instance of [ConceptGenerator](ontolearn.concept_generator.ConceptGenerator)
which we use in this case to create the negated concept `¬female`. The other classes 
[OWLObjectProperty](owlapy.model.OWLObjectProperty), 
[OWLObjectSomeValuesFrom](owlapy.model.OWLObjectSomeValuesFrom) 
and [OWLObjectIntersectionOf](owlapy.model.OWLObjectIntersectionOf) are classes
that represent different kind of axioms in owlapy and can be found in 
[owlapy model](owlapy.model) module. There are more kind of axioms there which you
can use to construct class expressions like we did in the example above.

### Evaluation and results

You can now evaluate the concept you just constructed as follows:

<!--pytest-codeblocks:cont-->
```python
from ontolearn.metrics import F1

evaluated_concept = kb.evaluate_concept(concept_to_test, F1(), encoded_lp)
```
In this example we use F1-score to evaluate the concept, but there are more [metrics](ontolearn.metrics) 
which you can use including Accuracy, Precision and Recall. 

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

## Obtaining axioms

You can retrieve Tbox and Abox axioms by using `tbox` and `abox` methods respectively.
Let us take them one at a time. The `tbox` method has 2 parameters, `entities` and `mode`.
`entities` specifies the owl entity from which we want to obtain the Tbox axioms. It can be 
a single entity, a `Iterable` of entities, or `None`. 

The allowed types of entities are: 
- OWLClass
- OWLObjectProperty
- OWLDataProperty

Only the Tbox axioms related to the given entit-y/ies will be returned. If no entities are 
passed, then it returns all the Tbox axioms.
The second parameter `mode` _(str)_ sets the return format type. It can have the
following values:
1) `'native'` -> triples are represented as tuples of owlapy objects.
2) `'iri'` -> triples are represented as tuples of IRIs as strings.
3) `'axiom'` -> triples are represented as owlapy axioms.

For the `abox` method the idea is similar. Instead of the parameter `entities`, there is the parameter 
`individuals` which accepts an object of type OWLNamedIndividuals or Iterable[OWLNamedIndividuals].

If you want to obtain all the axioms (Tbox + Abox) of the knowledge base, you can use the method `triples`. It
requires only the `mode` parameter.

> **NOTE**: The results of these methods are limited only to named and direct entities. 
> That means that especially the axioms that contain anonymous owl objects (objects that don't have an IRI)
> will not be part of the result set. For example, if there is a Tbox T={ C ⊑ (A ⊓ B), C ⊑ D }, 
> only the latter subsumption axiom will be returned.


## Sampling the Knowledge Base

Sometimes ontologies and therefore knowledge bases can get very large and our
concept learners become inefficient in terms of runtime. Sampling is an approach
to extract a portion of the whole knowledge base without changing its semantic and
still being expressive enough to yield results with as little loss of quality as 
possible. [OntoSample](https://github.com/alkidbaci/OntoSample/tree/main) is 
a library that we use to perform the sampling process. It offers different sampling 
techniques which fall into the following categories:

- Node-based samplers
   - Random Node
- Edge-based samplers
   - Random Edge
- Exploration-based samplers
    - Random Walker
    - Random Walker with Jumps
    - Random Walker with Prioritization
    - Random Walker with Prioritization & Jumps
    - Forest Fire

and almost each sampler is offered in 3 modes:

- Classic
- Learning problem first (LPF)
- Learning problem centered (LPC)

When used on its own, Ontosample uses a light version of Ontolearn (`ontolearn_light`) 
to reason over ontologies, but when both packages are installed in the same environment 
it will use ontolearn module instead. This is made for compatibility reasons.

Ontosample treats the knowledge base as a graph where nodes are individuals
and edges are object properties. However, Ontosample also offers support for 
data properties sampling, although they are not considered as _"edges"_.

#### Sampling steps:
1. Initialize the sample using a `KnowledgeBase` object. If you are using an LPF or LPC
   sampler than you also need to pass the set of learning problem individuals (`lp_nodes`).
2. To perform the sampling use the `sample` method where you pass the number
   of nodes (`nodes_number`) that you want to sample, the amount of data properties in percentage
   (`data_properties_percentage`) that you want to sample which is represented by float values 
   form 0 to 1 and jump probability (`jump_prob`) for samplers that 
   use "jumping", a technique to avoid infinite loops during sampling.
3. The `sample` method returns the sampled knowledge which you can store to a 
   variable, use directly in the code or save locally by using the static method 
   `save_sample`.

Let's see an example where we use RandomWalkSampler to sample a knowledge base:

```python
from ontosample.classic_samplers import RandomNodeSampler

# 1. Initialize KnowledgeBase object using the path of the ontology
kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")

# 2. Initialize the sampler and generate the sample
sampler = RandomNodeSampler(kb)
sampled_kb = sampler.sample(30) # will generate a sample with 30 nodes

# 3. Save the sampled ontology
sampler.save_sample(kb=sampled_kb, filename="some_name")
```

Here is another example where this time we use an LPC sampler:

```python
from ontosample.lpc_samplers import RandomWalkerJumpsSamplerLPCentralized
from owlapy.model import OWLNamedIndividual,IRI
import json

# 0. Load json that stores the learning problem
with open("examples/uncle_lp2.json") as json_file:
    examples = json.load(json_file)

# 1. Initialize KnowledgeBase object using the path of the ontology
kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")

# 2. Initialize learning problem (required only for LPF and LPC samplers)
pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
lp = pos.union(neg)

# 3. Initialize the sampler and generate the sample
sampler = RandomWalkerJumpsSamplerLPCentralized(graph=kb, lp_nodes=lp)
sampled_kb = sampler.sample(nodes_number=40,jump_prob=0.15)

# 4. Save the sampled ontology
sampler.save_sample(kb=sampled_kb, filename="some_other_name")
```

> WARNING! Random walker and Random Walker with Prioritization are two samplers that suffer 
> from non-termination in case that the ontology contains nodes that point to each other and 
> forms an inescapable loop. In this scenario you can use their "jumping" version to escape 
> these loops and ensure termination.

To see how to use a sampled knowledge base for the task of concept learning check
the `sampling_example.py` in [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) 
folder. You will find descriptive comments in that script that will help you understand it better.

For more details about OntoSample you can see [this paper](https://dl.acm.org/doi/10.1145/3583780.3615158).

-----------------------------------------------------------------------------------------------------

Since we cannot cover everything here in details, see [KnowledgeBase API documentation](ontolearn.knowledge_base.KnowledgeBase)
to check all the methods that this class has to offer. You will find convenient methods to 
access the class/property hierarchy, methods that use the reasoner indirectly and 
a lot more.

Speaking of the reasoner, it is important that an ontology 
is associated with a reasoner which is used to inferring knowledge 
from the ontology, i.e. to perform ontology reasoning.
In the next guide we will see how to use a reasoner in Ontolearn. 