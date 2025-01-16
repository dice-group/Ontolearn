# Concept Learning

This is a guide to show how to use a concept learner to generate hypotheses for a target
concept in an ontology.
In this guide we will show how to use the following concept learners
of Ontolearn library:
- [TDL](ontolearn.learners.learners.tree_learner.TDL)
- [EvoLearner](ontolearn.concept_learner.EvoLearner)
- [CELOE](ontolearn.learners.celoe.CELOE)
- [OCEL](ontolearn.learners.ocel.OCEL)
- [Drill](ontolearn.learners.drill.Drill)

It is worth mentioning that NCES2 and NERO are not yet implemented in Ontolearn,
but they will be soon.

### Expressiveness

TDL → **SHOIN**

Evolearner → _**ALCQ(D)**_.

DRILL  → _**ALC**_

NCES  → **_ALC_**

NCES2 → **_ALCHIQ(D)_**

NERO → **_ALC_**

CLIP → **_ALC_**

CELOE and OCEL → **_ALC_**

-----------------------------------


The learning models that we mentioned in the beginning are similar to execute, for that reason, we are 
describing them in a general manner. To test them separately see [_Quick try-out_](#quick-try-out).
Each algorithm has different available configuration. However, at
minimum, they require a [knowledge base](04_knowledge_base.md) to initialize and a [learning problem](04_knowledge_base.md#construct-a-learning-problem) to learn predictions for.

Let's see the prerequisites needed to run the concept learners:

## Prerequisites

Before configuring and running an algorithm, we recommend you to store the dataset path
that ends with `.owl` and the IRIs as string of the **learning problem** instances in a json file as shown below.
The learning problem is further divided in positive and negative examples. We have saved ourselves some
hardcoded lines which we can now simply access by loading the json file. Below is 
an example file that we are naming `synthetic_problems.json`  showing how should it look:

    {  
      "data_path": "../KGs/Family/family-benchmark_rich_background.owl",  
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

Note: The KGs directory contains datasets, and it's not part of the project.
They have to be downloaded first, see [Download External Files](02_installation.md#download-external-files).
There you will also find instructions to download LPs folder which contains learning problems for those KGs or you can 
just use the direct downloading links below:
- [KGs.zip](https://files.dice-research.org/projects/Ontolearn/KGs.zip)
- [LPs.zip](https://files.dice-research.org/projects/Ontolearn/LPs.zip)

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
[KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase) (or [TripleStore](ontolearn.triple_store.TripleStore) )
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

typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
```

To construct an [OWLNamedIndividual](https://dice-group.github.io/owlapy/autoapi/owlapy/owl_individual/index.html#owlapy.owl_individual.OWLNamedIndividual) 
object an [IRI](https://dice-group.github.io/owlapy/autoapi/owlapy/iri/index.html#owlapy.iri.IRI) is required as an input. 
You can simply create an `IRI` object by calling the static method `create` 
and passing the IRI as a `string`.


## Configuring & Executing a Concept Learner
To learn class expressions we need to build a model of the concept learner
that we want to use. Depending on the model you chose there are
different initialization parameters which you can check [here](ontolearn.learners).
With exception of TDL, for other models you can specify the quality function used during learning.
Let's see how you can do that. 


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

*(Optional)* If you are using `KnowledeBase` and you want the learning model to ignore a target concepts see 
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
into this model, and it will generate a 
**hypotheses** that explain the positive and negative examples.
You can do that by calling the method `fit` :

<!--pytest-codeblocks:cont-->
```python
model.fit(lp)
```

You can retrieve the hypotheses using
the method `best_hypotheses` where `n` is the number of hypotheses you want to return. 

<!--pytest-codeblocks:cont-->
```python
hypotheses = model.best_hypotheses(n=3)
```

The class expressions can be rendered in DL syntax using [DLSyntaxObjectRenderer]() from owlapy.

```python
from owlapy.render import DLSyntaxObjectRenderer

render = DLSyntaxObjectRenderer()
for h in hypotheses:
    dl_concept_as_str = render.render(h)
    print(dl_concept_as_str)
```

The hypotheses can also be saved locally:

<!--pytest-codeblocks:cont-->
```python
model.save_best_hypothesis(n=3, path='Predictions')
```

`save_best_hypothesis` method creates a `.owl` file of the RDF/XML format 
containing the generated hypotheses. 
The number of hypotheses is specified by the parameter `n`. 
`path` parameter specifies the filepath where the predictions will be stored.


Furthermore, you can create a binary classification for the specified individuals, given the hypotheses,
by using the `predict` method:

```python
binary_classification = model.predict(individuals=list(typed_pos | typed_neg), hypotheses=hypotheses)
```

Here we are classifying the positives and negatives individuals using the generated hypotheses.
This will return a data frame where 1 means True (covered by the hypothesis) and 0 means False 
(not covered by the hypothesis).


### Verbalization

You can as well verbalize or visualize the generated hypotheses into images by using the
static function `verbalize`. This functionality requires an external package which
is not part of the required packages for Ontolearn as well as _**graphviz**_. 

1. Install deeponto. `pip install deeponto` + further requirements like JDK, etc. 
   Check https://krr-oxford.github.io/DeepOnto/ for full instructions.
2. Install graphviz at https://graphviz.org/download/.

After you are done with that you can simply verbalize predictions:

```python
from ontolearn.utils.static_funcs import verbalize

verbalize('Predictions.owl')
```
This will create for each class expression inside `Predictions.owl` a `.png` 
image that contain the tree representation of that class expression.

---------------------------------------------------------------------------------------

In the next guide you will find further resources about Ontolearn including papers to cite, further directions for 
examples inside the project, code coverage, etc.