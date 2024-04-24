# Concept Learning

This is a guide to show how to use a concept learner to generate hypotheses for a target
concept in an ontology.
In this guide we will show how to use the following concept learners
of Ontolearn library:
- [EvoLearner](ontolearn.concept_learner.EvoLearner)
- [CELOE](ontolearn.concept_learner.CELOE)
- [OCEL](ontolearn.concept_learner.OCEL)

The other concept learners are not covered here in details, but we have provided
examples for them. Check the jupyter notebook files  as well as other example scripts
for the corresponding learner inside the 
[examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder 
(direct links are given at the end of this guide).

It is worth mentioning that NCES2 and NERO are not yet implemented in Ontolearn,
but they will be soon.

### Expressiveness

Evolearner → _**ALCQ(D)**_.

DRILL  → _**ALC**_

NCES  → **_ALC_**

NCES2 → **_ALCHIQ(D)_**

NERO → **_ALC_**

CLIP → **_ALC_**

CELOE and OCEL → **_ALC_**

-----------------------------------


The three algorithms that we mentioned in the beginning are similar in execution, for that reason, we are 
describing them in a general manner. To test them separately see [_Quick try-out_](#quick-try-out).
Each algorithm may have different available configuration. However, at
minimum, they require a [knowledge base](04_knowledge_base.md) and a
[learning problem](04_knowledge_base.md#construct-a-learning-problem).

Let's see the prerequisites needed to run the concept learners:

## Prerequisites

Before configuring and running an algorithm, we recommend you to store the dataset path
that ends with `.owl` and the IRIs as string of the **learning problem** instances in a json file as shown below.
The learning problem is further divided in positive and negative examples. We have saved ourselves some
hardcoded lines which we can now simply access by loading the json file. Below is 
an example file that we are naming `synthetic_problems.json`  showing how should it look:

    {  
      "data_path": "../KGs/Family/family-benchmark_rich_background2.owl",  
      "learning_problem": {
        "positive_examples": [  
        "http://www.benchmark.org/family#F2F28",  
        "http://www.benchmark.org/family#F2F36",  
        "http://www.benchmark.org/family#F3F52"  
        ],  
        "negative_examples": [  
        "http://www.benchmark.org/family#F6M69",  
        "http://www.benchmark.org/family#F6M100",  
        "http://www.benchmark.org/family#F2F30"  
        ]
      }  
    }

We are considering that you are trying this script inside `examples` folder, and
therefore we have stored the ontology path like that.

> Note: The KGs directory contains datasets, and it's not part of the project.
> They have to be downloaded first, see [Download External Files](02_installation.md#download-external-files).

## Configuring Input Parameters 

Before starting with the configuration you can enable logging to see the logs
which give insights about the main processes of the algorithm:

```python
from ontolearn.utils import setup_logging

setup_logging()
```

We then start by loading the `synthetic_problems.json` where we have 
stored the knowledge base path and the learning problems in the variable `settings`:

<!--pytest-codeblocks:cont-->
```python
import json

with open('synthetic_problems.json') as json_file:    
    settings = json.load(json_file)
```

### Load the ontology

Load the ontology by simply creating an instance of the class 
[KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase)
and passing the ontology path stored 
under `data_path` property of `settings`:

<!--pytest-codeblocks:cont-->
```python
from ontolearn.knowledge_base import KnowledgeBase

kb = KnowledgeBase(path=settings['data_path'])
```

## Configure the Learning Problem

The Structured Machine Learning implemented in our Ontolearn library
is working with a type of [supervised
learning](https://en.wikipedia.org/wiki/Supervised_learning). One of
the first things to do after loading the Ontology to a `KnowledgeBase` object 
is thus to define the learning problem for which the 
learning algorithm is trying to generate hypothesis (class expressions).

First and foremost, load the learning problem examples from the json file
into sets as shown below:

<!--pytest-codeblocks:cont-->
```python
positive_examples = set(settings['learning_problem']['positive_examples'])  
negative_examples = set(settings['learning_problem']['negative_examples'])
```

In Ontolearn you represent the learning problem as an object of the class
`PosNegLPStandard` which has two parameters `pos` and `neg` respectively
for the positive and negative examples.  These parameters are of 
type `set[OWLNamedIndividual]`.  We create these sets by mapping 
each individual (stored as `string`) from the set `positive_examples `
and `negative_examples` to `OWLNamedIndividual`: 

<!--pytest-codeblocks:cont-->

```python
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import IRI, OWLNamedIndividual

typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
```

To construct an [OWLNamedIndividual](https://dice-group.github.io/owlapy/autoapi/owlapy/owl_individual/index.html#owlapy.owl_individual.OWLNamedIndividual) 
object an [IRI](https://dice-group.github.io/owlapy/autoapi/owlapy/iri/index.html#owlapy.iri.IRI) is required as an input. 
You can simply create an `IRI` object by calling the static method `create` 
and passing the IRI as a `string`.


## Configuring & Executing a Concept Learner
To learn class expressions we need to build a model of the concept learner
that we want to use. It can be either EvoLearner, CELOE or OCEL. Depending on the algorithm you chose there are
different initialization parameters which you can check [here](ontolearn.concept_learner).
Let's start by setting a quality function. 


### Quality metrics

There is a default quality function to evaluate the quality of the
found expressions but different concept learners have different
default quality function. Therefore, you may want to set it explicitly.
There are the following quality function:[F1 Score](ontolearn.metrics.F1),
[Predictive Accuracy](ontolearn.metrics.Accuracy),
[Precision](ontolearn.metrics.Precision), and
[Recall](ontolearn.metrics.Recall). To use a quality function, first create an instance of its class:

<!--pytest-codeblocks:cont-->
```python
from ontolearn.metrics import Accuracy

pred_acc = Accuracy()
```

In the following example we have built a model of [OCEL](ontolearn.concept_learner.OCEL) and 
we have specified some of the parameters which can be set for OCEL.

*(Optional)* If you have target concepts that you want to ignore check 
[_how to ignore concepts_](04_knowledge_base.md#ignore-concepts).

### Create a model

<!--pytest-codeblocks:cont-->
```python
from ontolearn.concept_learner import OCEL

model = OCEL(knowledge_base=kb, 
              quality_func = pred_acc,
              max_runtime=600,  
              max_num_of_concepts_tested=10_000_000_000,  
              iter_bound=10_000_000_000)
```

The parameter `knowledge_base` which is the only required parameter, specifies the
knowledge base that is used to learn and test concepts. 
The following parameters are optional.
- `quality_func` - function to evaluate the quality of solution concepts. (Default value = F1()) 
- `max_runtime` - runtime limit in seconds. (Default value = 5)
- `max_num_of_concepts_tested` - limit to stop the algorithm after _n_ concepts tested. (Default value = 10_000)
- `iter_bound` - limit to stop the algorithm after _n_ refinement steps are done. (Default value = 10_000)


### Execute and fetch the results

After creating the model you can **fit** the learning problem
into this model, and it will find 
the **hypotheses** that explain the positive and negative examples.
You can do that by calling the method `fit` :

<!--pytest-codeblocks:cont-->
```python
model.fit(lp)
```

The hypotheses can be saved:

<!--pytest-codeblocks:cont-->
```python
model.save_best_hypothesis(n=3, path='Predictions')
```

`save_best_hypothesis` method creates a `.owl` file of the RDF/XML format 
containing the generated (learned) hypotheses. 
The number of hypotheses is specified by the parameter `n`. 
`path` parameter specifies the name of the file.

If you want to print the hypotheses you can use the method `best_hypotheses`
which will return the `n` best hypotheses together with some insights such 
as quality value, length, tree length and tree depth of
the hypotheses, and the number of individuals that each of them is covering, use 
the method `best_hypotheses` where `n` is the number of hypotheses you want to return.

<!--pytest-codeblocks:cont-->
```python
hypotheses = model.best_hypotheses(n=3)  
[print(hypothesis) for hypothesis in hypotheses]
```

You can also create a binary classification for the specified individuals by using the 
`predict` method as below:

```python
binary_classification = model.predict(individuals=list(typed_pos | typed_neg), hypotheses=hypotheses)
```

Here we are classifying the positives and negatives individuals using the generated hypotheses.
This will return a data frame where 1 means True and 0 means False.


### Verbalization

You can as well verbalize or visualize the generated hypotheses by using the
static method `verbalize`. This functionality requires an external package which
is not part of the required packages for Ontolearn as well as _**graphviz**_. 

1. Install deeponto. `pip install deeponto` + further requirements like JDK, etc. 
   Check https://krr-oxford.github.io/DeepOnto/ for full instructions.
2. Install graphviz at https://graphviz.org/download/.

After you are done with that you can simply verbalize predictions:

```python
model.verbalize('Predictions.owl')
```
This will create for each class expression inside `Predictions.owl` a `.png` 
image that contain the tree representation of that class expression.


## Quick try-out

You can execute the script `deploy_cl.py` to deploy the concept learners in a local web server and try
the algorithms using an interactive interface made possible by [gradio](https://www.gradio.app/). Currently, 
you can only deploy the following concept learners: **NCES**, **EvoLearner**, **CELOE** and **OCEL**.

**Warning!** Gradio is not part of the required packages. Therefore, if you want to use this functionality
you need to install gradio in addition to the other dependencies:

```shell
pip install gradio
```

> **NOTE**: In case you don't have you own dataset, don't worry, you can use
> the datasets we store in our data server. See _[Download external files](02_installation.md#download-external-files)_.

For example the command below will launch an interface using **EvoLearner** as the model on 
the **Family** dataset which is a simple dataset with 202 individuals:


```shell
python deploy_cl.py --model evolearner --path_knowledge_base KGs/Family/family-benchmark_rich_background.owl
```

Once you run this command, a local URL where our model is deployed will be provided to you.

In the interface you need to enter the positive and the negative examples. For a quick run you can
click on the **Random Examples** checkbox, but you may as well enter some real examples for
the learning problem of **Aunt**, **Brother**, **Cousin**, etc. which
you can find in the folder `examples/synthetic_problems.json`. Just copy and paste the IRIs of
positive and negative examples for a certain learning problem directly
in their respective fields.

Run the help command to see the description on this script usage:

```shell
python deploy_cl.py --help
```

---------------------------------------------------------------------------------------
## Use Triplestore Knowledge Base

Instead of going through nodes using expensive computation resources why not just make use of the
efficient approach of querying a triplestore using SPARQL queries. We have brought this 
functionality to Ontolearn for our learning algorithms, and we take care of the conversion part behind the scene.
Let's see what it takes to make use of it.

First of all you need a server which should host the triplestore for your ontology. If you don't
already have one, see [Loading and Launching a Triplestore](#loading-and-launching-a-triplestore) below.

Now you can simply initialize a `TripleStoreKnowledgeBase` object that will server as an input for your desired 
concept learner as follows:

```python
from ontolearn.triple_store import TripleStoreKnowledgeBase

kb = TripleStoreKnowledgeBase("http://your_domain/some_path/sparql")
```

Notice that the triplestore endpoint is the only argument that you need to pass.
Also keep in mind that this knowledge base contains a 
[TripleStoreOntology](ontolearn.triple_store.TripleStoreOntology) 
and [TripleStoreReasoner](ontolearn.triple_store.TripleStoreReasoner) which means that
every querying process concerning concept learning is now using the triplestore.

> **Important notice:** The performance of a concept learner may differentiate when using triplestore.
>  This happens because some SPARQL queries may not yield the exact same results as the local querying methods.


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

Fuseki/apache-jena-4.7.0/bin/tdb2.tdbloader --loader=parallel --loc Fuseki/apache-jena-fuseki-4.7.0/databases/father/ KGs/father.owl
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

You can now create a triplestore knowledge base or a reasoner that uses this URL for their 
operations:

```python
from ontolearn.triple_store import TripleStoreKnowledgeBase

father_kb = TripleStoreKnowledgeBase("http://localhost:3030/father/sparql")

# ** Continue to execute the learning algorithm as you normally do. ** .
```

-------------------------------------------------------------------


In this guide, we have shown the prerequisites of running a concept learner,
how to configure it's input properties and how to run it to successfully
learn class expressions for learning problems in an ontology. We showed as well how to set up
a triplestore server that can be used to execute the concept learner. There is also a jupyter 
notebook for each of these concept learners:

- [NCES notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/simple-usage-NCES.ipynb)
- [CLIP notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/clip_notebook.ipynb)
- [DRILL notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/drill_notebook.ipynb)
- [EvoLearner notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/evolearner_notebook.ipynb)
- [CELOE notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/celoe_notebook.ipynb)
- [OCEL notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/ocel_notebook.ipynb)
- [TDL example](https://github.com/dice-group/Ontolearn/blob/develop/examples/concept_learning_with_tdl_and_triplestore_kb.py)
