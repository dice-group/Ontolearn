# Concept Learners

This is a guide to show how to use a concept learner to generate hypotheses for a target
concept in an ontology.
In this guide we will show how to use the following concept learners
of Ontolearn library:
- [EvoLearner](ontolearn.concept_learner.EvoLearner)
- [CELOE](ontolearn.concept_learner.CELOE)
- [OCEL](ontolearn.concept_learner.OCEL)

These algorithms are similar in execution, for that reason, we are 
describing them in a general manner. To test them separately we have provided
a jupyter notebook file for each of them (find them [here](#end-notes)).
Each algorithm may have different available configuration. However, at
minimum, they require a [knowledge base](01_knowledge_base.md) and a
[learning problem](02_learning_problem.md).

Let's see the prerequisites needed to run the concept learners:

----------------------------------------------------------------------------

## Prerequisites

Before configuring and running an algorithm, we recommend you store the dataset path
having a `.owl` extension and the **learning problems** in a json file. For each learning problem, 
there are positive and negative examples that consist of IRIs of the respective individuals
as a string. Now we have the learning problems more organized, and we can access them later by loading the json file. Below is 
an example file `synthetic_problems.json` showing how should it look:

   

    {  
      "data_path": "../KGs/Family/family-benchmark_rich_background2.owl",  
      "problems": {  
        "concept_1": {  
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
        "concept_2": {  
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

----------------------------------------------------------------------------

## Configuring input parameters 

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



#### Load the knowledge base

Load the knowledge base by simply creating an instance of the class 
[KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase)
and passing the knowledge base path stored 
under `data_path` property of `settings`:

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



#### Configure the learning problem

Store the positive and negative examples into sets and 
*(optional)* print `str_target_concept` to keep track of 
which target concept is currently being learned:

<!--pytest-codeblocks:cont-->
```python
    positive_examples = set(examples['positive_examples'])  
    negative_examples = set(examples['negative_examples'])  
    print('Target concept: ', str_target_concept)
```

*(Optional)* If you have target concepts that you want to ignore check 
[how to ignore concepts](01_knowledge_base.md#ignore-concepts)


In Ontolearn you represent the learning problem as an object of the class
`PosNegLPStandard` which has two parameters `pos` and `neg` respectively
for the positive and negative examples.  These parameters are of 
type `set[OWLNamedIndividual]`.  We create these sets by mapping 
each individual (stored as `string`) from the set `positive_examples `
and `negative_examples` to `OWLNamedIndividual`: 

<!--pytest-codeblocks:cont-->
```python
    from ontolearn.learning_problem import PosNegLPStandard
    from owlapy.model import IRI,OWLNamedIndividual
    
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))  
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
```

To construct an [OWLNamedIndividual](owlapy.model.OWLNamedIndividual) object an [IRI](owlapy.model.IRI) is required as an input. 
You can simply create an `IRI` object by calling the inbuilt method `create` 
and passing the IRI as a `string`.

----------------------------------------------------------------------------

## Configuring and Executing a Concept Learner
To learn class expressions we need to build a model of the concept learner
that we want to use. It can be either EvoLearner, CELOE or OCEL. Depending on the algorithm you chose there are
different initialization parameters which you can check [here](ontolearn.concept_learner).
Let's start by setting a quality function. 


#### Quality metrics

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

In the following example we have build a model of [OCEL](ontolearn.concept_learner.OCEL) and 
we have specified some of the parameters which OCEL offers:

#### Create a model

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


#### Execute and fetch the results

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

----------------------------------------------------------------------------


### End Notes

In this guide, we have shown the prerequisites of running a concept learner,
how to configure its input properties and how to run it to successfully
learn class expressions for learning problems in an ontology. You can try the concept
learners that we mentioned in this guide by executing the following jupyter notebook 
files:

- [EvoLearner notebook](evolearner_notebook.ipynb)
- [CELOE notebook](celoe_notebook.ipynb)
- [OCEL notebook](ocel_notebook.ipynb)
