# Learning Algorithm

With a [learning problem](02_learning_problem) defined, it is now
possible to configure the learning algorithm.

Currently, two [Base Concept
Learners](ontolearn.base_concept_learner.BaseConceptLearner) are
provided in our Ontolearn library. The [length base
learner](ontolearn.concept_learner.LengthBaseLearner) and the
[modified CELOE algorithm](ontolearn.concept_learner.CELOE). Each
algorithm may have different available configuration. However at
minimum they require a [knowledge
base](./02_learning_problem.md#knowledge-base-over-the-ontology) and a
learning problem.

## Configuring CELOE

Let us now configure the modified CELOE algorithm:

```py
from ontolearn.concept_learner import CELOE

alg = CELOE(kb, learning_problem=lp)
```

There are further configuration choices of CELOE, such as the
[Refinement Operator](ontolearn.abstracts.BaseRefinement) to use in
the search process.  The quality function ([Predictive
Accuracy](ontolearn.metrics.Accuracy), [F1
Score](ontolearn.metrics.F1),
[Precision](ontolearn.metrics.Precision), or
[Recall](ontolearn.metrics.Recall)) to evaluate the quality of the
found expressions can be configured. There is the heuristic function
to evaluate the quality of expressions during the search process. And
some options limit the run-time, such as `max_runtime` (maximum
run-time in seconds) or `max_num_of_concepts_tested` (maximum number
of concepts that will be tested before giving up) or `iter_bound`
(maximum number of refinement attempts).

### Changing the quality function

To use another quality function, first create an instance of the
function:

```py
from ontolearn.metrics import Accuracy

pred_acc = Accuracy(learning_problem=lp)
```

### Configuring the heuristic

```py
from ontolearn.heuristics import CELOEHeuristic

heur = CELOEHeuristic(
    expansionPenaltyFactor=0.05,
    startNodeBonus=1.0,
    nodeRefinementPenalty=0.01)
```

Then, configure everything on the algorithm:

```py
alg = CELOE(kb, learning_problem=lp, quality_func=pred_acc, heuristic_func=heur)
```

## Running the algorithm

To run the algorithm, you have to call the
[fit](ontolearn.base_concept_learner.BaseConceptLearner.fit)
method. Afterwards, you can fetch the result using the
[best_hypotheses](ontolearn.base_concept_learner.BaseConceptLearner.best_hypotheses)
method:

```py
from owlapy.render import DLSyntaxObjectRenderer

dlsr = DLSyntaxRenderer()

alg.fit()

for desc in alg.best_hypotheses(1):
    print('The result:', dlsr.render(desc.concept), 'has quality', desc.quality)
```

In this example code, we have created a DL-Syntax renderer in order to display the
[OWL Class Expression](https://www.w3.org/TR/owl-quick-reference/#Class_Expressions)
in a Description Logics style. This is purely for aesthetic purposes.
