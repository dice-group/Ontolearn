# Knowledge Bases

In Ontolearn a knowledge base is represented
by an implementor of [AbstractKnowledgeBase](ontolearn.abstracts.AbstractKnowledgeBase) which contains two main
attributes, an ontology of type [AbstractOWLOntology](https://dice-group.github.io/owlapy/autoapi/owlapy/owl_ontology/index.html#owlapy.owl_ontology.AbstractOWLOntology)
and a reasoner of type [AbstractOWLReasoner](https://dice-group.github.io/owlapy/autoapi/owlapy/owl_reasoner/index.html#owlapy.owl_reasoner.AbstractOWLReasoner). Be careful, different implementations of these abstract classes 
are not compatible with each other. For example, you can not use [TripleStore](ontolearn.triple_store.TripleStore) 
knowledge base with 
[StructuralReasoner](https://dice-group.github.io/owlapy/autoapi/owlapy/owl_reasoner/index.html#owlapy.owl_reasoner.StructuralReasoner), 
but you can use _TripleStore_ KB with [TripleStoreReasoner](ontolearn.triple_store.TripleStoreReasoner).
_AbstractKnowledgeBase_ contains the necessary methods to facilitate _Structured Machine Learning_.

Currently, there are two implementation of _AbstractKnowledgeBase_:

- [KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase) &rarr; used for local datasets.
- [TripleStore](ontolearn.triple_store.TripleStore) &rarr; used for datasets hosted on a server.

## Knowledge Base vs Ontology

These terms may be used interchangeably sometimes but in Ontolearn they are not the same thing,
although they share a lot of similarities. An ontology in owlapy, as explained 
[here](https://dice-group.github.io/owlapy/usage/ontologies.html) is the object where we load 
the OWL 2.0 ontologies from a _.owl_ file containing the ontology in an RDF/XML or OWL/XML format.
On the other side a knowledge base combines an ontology and a reasoner together.
Therefore, differently from the ontology you can use methods that require reasoning. You can check 
the methods for each in the links below:

- [AbstractKnowledgeBase](ontolearn.knowledge_base.AbstractKnowledgeBase)
- [AbstractOWLOntology](https://dice-group.github.io/owlapy/autoapi/owlapy/owl_ontology/index.html#owlapy.owl_ontology.AbstractOWLOntology)

In summary:

- An implementation of `AbstractKnowledgeBase` contains an ontology and a reasoner and 
is required to run a learning algorithm.

- An ontology represents the OWL 2 ontology you have locally or hosted on triplestore server. Using class methods you
can retrieve information from signature of this ontology. In case of a local the ontology, it can be modified and 
saved.

- Although they have some similar functionalities, there are a lot of other distinct 
functionalities that each of them has.


## Create an Object of KnowledgeBase

Let us show how you can initialize an object of `KnowledgeBase`.
We consider that you have already an OWL 2.0 ontology (containing *.owl* extension).

The simplest way is to use the path of your _.owl_ file as follows:

```python 
from ontolearn.knowledge_base import KnowledgeBase

kb = KnowledgeBase(path="file://KGs/Family/father.owl")
```

What happens in the background is that the ontology located in this path will be loaded
in the `AbstractOWLOntology` object of `kb` as done [here](https://dice-group.github.io/owlapy/usage/ontologies.html#loading-an-ontology).


## Ignore Concepts

During concept learning which we describe later, you may need to 
avoid trivial solutions from being learned. So in Ontolearn you
have the opportunity to ignore specific concepts. Since we pass a `KnowledgeBase`
object to the concept learner, we set this ignored concept using the method 
`ignore_and_copy` of the `KnowledgeBase` class.

We don't have such concept in our example ontology `KGs/Family/father.owl` but suppose that 
there is a class(concept) "Father" that we want to ignore, because we are trying
to learn this a meaningful class expression for 'Father' using other classes(e.g. male, female, ∃ hasChild.⊤... ).
So we need to ignore this concept before fitting a model (model fitting is covered in [concept learning](06_concept_learners.md)).
It can be done as follows:

<!--pytest-codeblocks:cont-->

```python
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI

iri = IRI('http://example.com/father#', 'Father')
father_concept = OWLClass(iri)
concepts_to_ignore = {father_concept}  # you can add more than 1

new_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
```
In this example, we have created an instance of 
[OWLClass](https://dice-group.github.io/owlapy/autoapi/owlapy/class_expression/owl_class/index.html#owlapy.class_expression.owl_class.OWLClass) 
by using an [IRI](https://dice-group.github.io/owlapy/autoapi/owlapy/iri/index.html#owlapy.iri.IRI). 
On the other side, an instance of `IRI` is created by passing two parameters which are
the namespace of the ontology and the remainder 'Father'.

## Accessing Individuals

You may need to work with individuals of a knowledge base. 
We cover different ways of accessing them.

Let us give a  simple example of how to get the individuals that
are classified by an [OWLClassExpression](https://dice-group.github.io/owlapy/autoapi/owlapy/class_expression/class_expression/index.html#owlapy.class_expression.class_expression.OWLClassExpression). As a class expression, we will simply use the 
concept 'male'.

<!--pytest-codeblocks:cont-->
```python
NS = 'http://example.com/father#'
male_concept = OWLClass(IRI(NS,'male'))

male_individuals = kb.individuals(male_concept)
```
Note that the namespace has to match the Namespace/IRI that is defined
in the Ontology.

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
from owlapy.class_expression import OWLThing

all_individuals_set  = kb.individuals_set(OWLThing)
```
The difference is that `individuals()` return type is generator. 
and `individuals_set()` return type is frozenset.

For large amount of data `individuals()` is more computationally efficient:

<!--pytest-codeblocks:cont-->
```python
male_individuals = kb.individuals(male_concept)

[print(ind) for ind in male_individuals] # print male individuals
```

## Sampling the Knowledge Base

Sometimes ontologies and therefore knowledge bases can get very large and our
concept learners become inefficient in terms of runtime. Sampling is an approach
to extract a portion of the whole knowledge base without changing its semantic and
still being expressive enough to yield results with as little loss of quality as 
possible. [OntoSample](https://github.com/alkidbaci/OntoSample/tree/main) is 
a library that we use to perform the sampling process. It offers different sampling 
techniques which fall into the following categories:

- Node-based samplers
- Edge-based samplers
- Exploration-based samplers

and almost each sampler is offered in 3 modes:

- Classic
- Learning problem first (LPF)
- Learning problem centered (LPC)

You can check them [here](https://github.com/alkidbaci/OntoSample/tree/main).

When operated on its own, Ontosample uses a light version of Ontolearn (`ontolearn_light`) 
to reason over ontologies, but when both packages are installed in the same environment 
it will use `ontolearn` module instead. This is made for compatibility reasons.

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

Let's see an example where we use [RandomNodeSampler](https://github.com/alkidbaci/OntoSample/blob/bc0e65a3bcbf778575fe0a365ea94250ea7910a1/ontosample/classic_samplers.py#L17C7-L17C24) to sample a 
knowledge base:

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
from owlapy.owl_individual import OWLNamedIndividual,IRI
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
> form an inescapable loop for the "walker". In this scenario you can use their "jumping" 
> version to make the "walker" escape these loops and ensure termination.

To see how to use a sampled knowledge base for the task of concept learning check
the `sampling_example.py` in [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) 
folder. You will find descriptive comments in that script that will help you understand it better.

For more details about OntoSample you can see [this paper](https://dl.acm.org/doi/10.1145/3583780.3615158).

## TripleSore Knowledge Base

Instead of going through nodes using expensive computation resources why not just make use of the
efficient approach of querying a triplestore using SPARQL queries. We have brought this 
functionality to Ontolearn for our learning algorithms, and we take care of the conversion part behind the scene.
Let's see what it takes to make use of it.

First of all you need a server which should host the triplestore for your ontology. If you don't
already have one, see [Loading and Launching a Triplestore](#loading-and-launching-a-triplestore) below.

Now you can simply initialize an instance of `TripleStore` class that will serve as an input for your desired 
concept learner:

```python
from ontolearn.triple_store import TripleStore

kb = TripleStore(url="http://your_domain/some_path/sparql")
```

Notice that the triplestore endpoint is enough to initialize an object of `TripleStore`.
Also keep in mind that this knowledge base can be initialized by using either one of 
[TripleStoreOntology](ontolearn.triple_store.TripleStoreOntology) or [TripleStoreReasoner](ontolearn.triple_store.TripleStoreReasoner). Using the `TripleStore` KB means that 
every querying process taking place during concept learning is now done using SPARQL queries.

> **Important notice:** The performance of a concept learner may differentiate when using TripleStore instead of
> KnowledgeBase for the same ontology. This happens because some SPARQL queries may not yield the exact same results
> as the local querying methods.


## Loading and Launching a Triplestore

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

Now just load the 'father' ontology using the following commands:

```shell
cd ..

Fuseki/apache-jena-4.7.0/bin/tdb2.tdbloader --loader=parallel --loc Fuseki/apache-jena-fuseki-4.7.0/databases/father/ KGs/Family/father.owl
```

Launch the server, and it will be waiting eagerly for your queries.

```shell
cd Fuseki/apache-jena-fuseki-4.7.0 

java -Xmx4G -jar fuseki-server.jar --tdb2 --loc=databases/father /father
```

Notice that we launched the database found in `Fuseki/apache-jena-fuseki-4.7.0/databases/father` to the path `/father`.
By default, jena-fuseki runs on port 3030 so the full URL would be: `http://localhost:3030/father`. When 
you pass this url to `triplestore_address` argument, you have to add the
`/sparql` sub-path indicating to the server that we are querying via SPARQL queries. Full path now should look like:
`http://localhost:3030/father/sparql`.

You can now create a triplestore knowledge base, a reasoner or an ontology that uses this URL for their 
operations.

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

-----------------------------------------------------------------------------------------------------

Since we cannot cover everything here in details, check the API docs for knowledge base related classes
to see all the methods that these classes have to offer.

In the next guide we will walk through _how to define a learning problem_, _how to use concept learners_ and other fancy 
stuff, like evaluating a class expression.