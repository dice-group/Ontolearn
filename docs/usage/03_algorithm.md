<!--
```python
from owlapy.namespaces import Namespaces
from owlapy.model import OWLNamedIndividual, IRI
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard

NS = Namespaces('ex', 'http://example.com/father#')
kb = KnowledgeBase(path="KGs/father.owl")

positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}
negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}
lp = PosNegLPStandard(pos=positive_examples, neg=negative_examples)
```
-->

# Learning Algorithm

Currently, one [Base Concept
Learner](ontolearn.base_concept_learner.BaseConceptLearner) is
provided in our Ontolearn library. The
[modified CELOE algorithm](ontolearn.concept_learner.CELOE). Each
algorithm may have different available configuration. However at
minimum they require a [knowledge
base](./02_learning_problem.md#knowledge-base-over-the-ontology) and a
learning problem.

## Configuring CELOE

Let us now configure the modified CELOE algorithm:

<!--pytest-codeblocks:cont-->
```python
from ontolearn.concept_learner import CELOE

alg = CELOE(kb)
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

<!--pytest-codeblocks:cont-->
```python
from ontolearn.metrics import Accuracy

pred_acc = Accuracy()
```

### Configuring the heuristic

<!--pytest-codeblocks:cont-->
```python
from ontolearn.heuristics import CELOEHeuristic

heur = CELOEHeuristic(
    expansionPenaltyFactor=0.05,
    startNodeBonus=1.0,
    nodeRefinementPenalty=0.01)
```

Then, configure everything on the algorithm:

<!--pytest-codeblocks:cont-->
```python
alg = CELOE(kb, quality_func=pred_acc, heuristic_func=heur)
```

## Running the algorithm

To run the algorithm, you have to call the
[fit](ontolearn.base_concept_learner.BaseConceptLearner.fit)
method. Afterwards, you can fetch the result using the
[best_hypotheses](ontolearn.base_concept_learner.BaseConceptLearner.best_hypotheses)
method:

<!--pytest-codeblocks:cont-->
```python
from owlapy.render import DLSyntaxObjectRenderer

dlsr = DLSyntaxObjectRenderer()

alg.fit(learning_problem=lp)

for desc in alg.best_hypotheses(1):
    print('The result:', dlsr.render(desc.concept), 'has quality', desc.quality)
```

In this example code, we have created a DL-Syntax renderer in order to display the
[OWL Class Expression](https://www.w3.org/TR/owl-quick-reference/#Class_Expressions)
in a Description Logics style. This is purely for aesthetic purposes.
