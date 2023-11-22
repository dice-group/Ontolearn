# Concept Learning

This is a guide to show how to use a concept learner to generate hypotheses for a target
concept in an ontology.
In this guide we will show how to use the following concept learners
of Ontolearn library:
- [EvoLearner](ontolearn.concept_learner.EvoLearner)
- [CELOE](ontolearn.concept_learner.CELOE)
- [OCEL](ontolearn.concept_learner.OCEL)


> **Important Notice**: 
> 
> **_DRILL_ is not fully implemented in Ontolearn**. In the meantime you can refer to 
> [_DRILL's_ GitHub repo](https://github.com/dice-group/drill).
> 
> **_NCES_ is not currently documented here**. You can visit _NCES_ jupyter notebooks
> inside [examples folder](https://github.com/dice-group/Ontolearn/tree/develop/examples) to find the description on
> how it works.
> 
> NCES2, CLIP and NERO are not yet implemented in Ontolearn, they will be soon.

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

Before configuring and running an algorithm, we recommend you store the dataset path
having a `.owl` extension and the **learning problems** in a json file. For each learning problem, 
there are positive and negative examples that consist of IRIs of the respective individuals
as a string. Now we have the learning problems more organized, and we can access them later by loading the json file. Below is 
an example file `synthetic_problems.json` showing how should it look:

   

    {  
      "data_path": "../KGs/Family/family-benchmark_rich_background2.owl",  
      "problems": {  
        "lp1": {  
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
        },  
        "lp2": {  
          "positive_examples": [  
            "http://www.benchmark.org/family#F2M13",  
            "http://www.benchmark.org/family#F2M18"  
          ],  
          "negative_examples": [  
            "http://www.benchmark.org/family#F10M196",  
            "http://www.benchmark.org/family#F1M8"  
          ]  
        }  
      }  
    }

We have stored the ontology path under the property `data_path` and the 
**learning problems** under the property `problems`.

## Configuring Input Parameters 

Before starting with the configuration you can enable logging to see the logs
which give insights about the main processes of the algorithm:

```python
from ontolearn.utils import setup_logging

setup_logging()
```

We then start by loading the `synthetic_problems.json` where we have 
stored the knowledge base path and the learning problems:

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

> The concept learning algorithm requires an object of type `KnowledgeBase`, that
> is why we are loading the ontology like this.

<!--pytest-codeblocks:cont-->
```python
from ontolearn.knowledge_base import KnowledgeBase

kb = KnowledgeBase(path=settings['data_path'])
```

To run the algorithm for each learning problem in `problems` property of 
`settings` you need to call a `for` loop:

<!--pytest-codeblocks:cont-->
```python
for str_target_concept, examples in settings['problems'].items():
```

> **Note**: all the **code blocks** shown **below** this notice are inside the "`for`" loop.


## Configure the Learning Problem

The Structured Machine Learning implemented in our Ontolearn library
is working with a type of [supervised
learning](https://en.wikipedia.org/wiki/Supervised_learning).  One of
the first things to do after loading the Ontology is thus to define
the positive and negative examples whose description the learning
algorithm should attempt to find.

Store the positive and negative examples into sets and 
*(optional)* print `str_target_concept` to keep track of 
which target concept is currently being learned:

<!--pytest-codeblocks:cont-->
```python
    positive_examples = set(examples['positive_examples'])  
    negative_examples = set(examples['negative_examples'])  
    print('Target concept: ', str_target_concept)
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
    from owlapy.model import IRI, OWLNamedIndividual
    
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
```

To construct an [OWLNamedIndividual](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/__init__.py#L917) 
object an [IRI](https://github.com/dice-group/owlapy/blob/6a6338665a6df0845e67eda577327ca4c62f446b/owlapy/model/_iri.py#L47) is required as an input. 
You can simply create an `IRI` object by calling the static method `create` 
and passing the IRI as a `string`.


## Configuring & Executing a Concept Learner
To learn class expressions we need to build a model of the concept learner
that we want to use. It can be either EvoLearner, CELOE or OCEL. Depending on the algorithm you chose there are
different initialization parameters which you can check [here](ontolearn.concept_learner).
Let's start by setting a quality function. 


### Quality metrics

The default quality function to evaluate the quality of the
found expressions is [F1 Score](ontolearn.metrics.F1).
There are as well [Predictive Accuracy](ontolearn.metrics.Accuracy),
[Precision](ontolearn.metrics.Precision), and
[Recall](ontolearn.metrics.Recall). To use another quality function, first create an instance of the function:

<!--pytest-codeblocks:cont-->
```python
    from ontolearn.metrics import Accuracy

    pred_acc = Accuracy()
```

In the following example we have built a model of [OCEL](ontolearn.concept_learner.OCEL) and 
we have specified some of the parameters which OCEL offers.

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

Now, after creating the model you can **fit** the learning problem
into this model, and it will find 
the **hypotheses** that explain the positive and negative examples.
You can do that by calling the method `fit` :

<!--pytest-codeblocks:cont-->
```python
    model.fit(lp)
```

The hypotheses can be saved or printed:

<!--pytest-codeblocks:cont-->
```python
    model.save_best_hypothesis(n=3, path=f'Predictions_{str_target_concept}')
```

`save_best_hypothesis` method creates a `.owl` file containing the hypotheses. 
The number of hypotheses is specified by the parameter `n`. 
`path` parameter specifies the name of the file.

If you want to print the hypotheses you can use the method `best_hypotheses`
which will return the `n` best hypotheses together with some insights such 
as quality which by default is F1-score, length, tree length, tree depth of 
the hypotheses, and the number of individuals that each of them is covering, use 
the method `best_hypotheses` where `n` is the number of hypotheses you want to return.

<!--pytest-codeblocks:cont-->
```python
    hypotheses = model.best_hypotheses(n=3)  
    [print(hypothesis) for hypothesis in hypotheses]
```


## Quick try-out

You can execute the script `deploy_cl.py` to deploy the concept learners in a local web server and try
the algorithms using an interactive interface made possible by [gradio](https://www.gradio.app/). Currently, 
you can only deploy the following concept learners: **NCES**, **EvoLearner**, **CELOE** and **OCEL**.

**Warning:** Gradio is not part of the required packages. Therefore, if you want to use this functionality
you need to install gradio in addition to the other dependencies:

```shell
pip install gradio
```

> **NOTE: In case you don't have you own dataset, don't worry, you can use
> the datasets we store in our data server. See _[Download external files](02_installation.md#download-external-files)_.**

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

Now you can simply initialize the `KnowledgeBase` object that will server as an input for your desired 
concept learner as follows:

```python
from ontolearn.knowledge_base import KnowledgeBase

kb = KnowledgeBase(triplestore_address="http://your_domain/some_path/sparql")
```

Notice that we did not provide a value for the `path` argument. When using triplestore, it is not required. Keep
in mind that the `kb` will create a default reasoner that uses the triplestore. Passing a custom
reasoner will not make any difference, because they all behave the same when using the triplestore.
You may wonder what happens to the `Ontology` object of the `kb` since no path was given. A default ontology 
object is created that will also use the triplestore for its processes. Basically every querying process concerning
concept learning is now using the triplestore.

> **Important notice:** The performance of a concept learner may differentiate when using triplestore.

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

You can now create a knowledge base or a reasoner object that uses this URL for their 
operations:

```python
from ontolearn.knowledge_base import KnowledgeBase

father_kb = KnowledgeBase(triplestore_address="http://localhost:3030/father/sparql")
# ** Execute the learning algorithm as you normally would. ** .
```

-------------------------------------------------------------------


In this guide, we have shown the prerequisites of running a concept learner,
how to configure it's input properties and how to run it to successfully
learn class expressions for learning problems in an ontology. We showed as well how to set up
a triplestore server that can be used to execute the concept learner. There is also a jupyter 
notebook for each of these concept learners:

- [NCES notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/simple-usage-NCES.ipynb)
- [EvoLearner notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/evolearner_notebook.ipynb)
- [CELOE notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/celoe_notebook.ipynb)
- [OCEL notebook](https://github.com/dice-group/Ontolearn/blob/develop/examples/ocel_notebook.ipynb)
