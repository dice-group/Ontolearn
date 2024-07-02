# Examples

In this guide we will show some non-trival examples of typical use-cases of Ontolearn
which you can also find in the
[examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder.


## Ex. 1: Learning Over a Local Ontology

The first example is about using EvoLearner to learn class expressions about the 
following target concepts: "_Aunt_", "_Brother_", "_Cousin_", "_Granddaughter_", 
"_Uncle_" and "_Grandgrandfather_". 

```python
import json
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass
from ontolearn.utils import setup_logging

setup_logging()

# Access the learning problem and the ontology file path
with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

# Define the Knowledge Base from a local ontology file
kb = KnowledgeBase(path=settings['data_path'])

# For each target concept
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)

    # Lets inject more background info where we ignore some trivial concepts
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister', 'Brother']:
        NS = 'http://www.benchmark.org/family#'
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Sister')),
            OWLClass(IRI(NS, 'Daughter')),
            OWLClass(IRI(NS, 'Mother')),
            OWLClass(IRI(NS, 'Grandmother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'PersonWithASibling')),
            OWLClass(IRI(NS, 'Granddaughter')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    else:
        target_kb = kb
    
    # Define learning problem
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    
    # Define learning model
    model = EvoLearner(knowledge_base=target_kb, max_runtime=600)
    
    # Fit the learning problem to the model
    model.fit(lp, verbose=False)
    
    # Save top n hypotheses
    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    
    # Get top n hypotheses
    hypotheses = list(model.best_hypotheses(n=3))
    
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                hypotheses=hypotheses)
    # Print hypothesis
    [print(_) for _ in hypotheses]
```

In `synthetic_problems.json` we store the learning problem individuals as string
of IRIs, as well as the path to the ontology. 
You can check that file [here](https://github.com/dice-group/Ontolearn/blob/develop/examples/synthetic_problems.json).

Instead of EvoLearner, you can use the other models. More details about this example
are given in the [_Concept Learning_](06_concept_learners.md) guide.


----------------------------------------------------------


## Ex. 2: Learning Over a Triplestore Ontology

In our next example we will see how to use another model, specifically 
the _Tree-based Description Logic Concept Learner_ or **TDL** for short in
a dataset which is hosted on a triplestore.

```python
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.triple_store import TripleStore
from ontolearn.utils.static_funcs import save_owl_class_expressions
from owlapy.render import DLSyntaxObjectRenderer

# (1) Initialize Triplestore
kb = TripleStore(url="http://dice-dbpedia.cs.upb.de:9080/sparql")

# (2) Initialize a DL renderer.
render = DLSyntaxObjectRenderer()

# (3) Initialize a learner.
model = TDL(knowledge_base=kb)

# (4) Define a description logic concept learning problem.
lp = PosNegLPStandard(pos={OWLNamedIndividual(IRI.create("http://dbpedia.org/resource/Angela_Merkel"))},
                      neg={OWLNamedIndividual(IRI.create("http://dbpedia.org/resource/Barack_Obama"))})
# (5) Learn description logic concepts best fitting (4).

h = model.fit(learning_problem=lp).best_hypotheses()
str_concept = render.render(h)
print("Concept:", str_concept)  # e.g.  ∃ predecessor.WikicatPeopleFromBerlin

# (6) Save ∃ predecessor.WikicatPeopleFromBerlin into disk
save_owl_class_expressions(expressions=h, path="owl_prediction")
```

Here we have used the triplestore endpoint as you see in step _(1)_ which is
available only on a private network. However, you can host your own triplestore
server following [this guide](06_concept_learners.md#loading-and-launching-a-triplestore)
and run TDL using you own local endpoint.

--------------------------------------------------------------

## Ex. 3: CMD Friendly Execution

Now let's see how you can create a script which you can execute via cmd and 
pass arguments dynamically. For this example we will use the learning model: 
**Drill**.

```python
import json
from argparse import ArgumentParser
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import Drill
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer


def start(args):
    # Define knowledge base
    kb = KnowledgeBase(path=args.path_knowledge_base)
    
    # Define Learning model: Drill
    drill = Drill(knowledge_base=kb,
                  path_embeddings=args.path_embeddings,
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                  quality_func=F1(),
                  reward_func=CeloeBasedReward(),
                  epsilon_decay=args.epsilon_decay,
                  learning_rate=args.learning_rate,
                  num_of_sequential_actions=args.num_of_sequential_actions,
                  num_episode=args.num_episode,
                  iter_bound=args.iter_bound,
                  max_runtime=args.max_runtime)

    if args.path_pretrained_dir:
        # load pretrained data
        drill.load(directory=args.path_pretrained_dir)
    else:
        # train the model (Drill needs training)
        drill.train(num_of_target_concepts=args.num_of_target_concepts,
                    num_learning_problems=args.num_of_training_learning_problems)
        drill.save(directory="pretrained_drill")

        
    # Define the larning problem and split them into "train" and "test"
    with open(args.path_learning_problem) as json_file:
        examples = json.load(json_file)
    p = examples['positive_examples']
    n = examples['negative_examples']
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
    X = np.array(p + n)
    Y = np.array([1.0 for _ in p] + [0.0 for _ in n])
    dl_render = DLSyntaxObjectRenderer()
    for (ith, (train_index, test_index)) in enumerate(kf.split(X, Y)):
        train_pos = {pos_individual for pos_individual in X[train_index][Y[train_index] == 1]}
        train_neg = {neg_individual for neg_individual in X[train_index][Y[train_index] == 0]}
        test_pos = {pos_individual for pos_individual in X[test_index][Y[test_index] == 1]}
        test_neg = {neg_individual for neg_individual in X[test_index][Y[test_index] == 0]}
        train_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, train_pos))),
                                    neg=set(map(OWLNamedIndividual, map(IRI.create, train_neg))))

        test_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, test_pos))),
                                   neg=set(map(OWLNamedIndividual, map(IRI.create, test_neg))))

        pred_drill = drill.fit(train_lp).best_hypotheses()
        # Quality on train data
        train_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                          pos=train_lp.pos,
                                          neg=train_lp.neg)
        # Quality on test data
        test_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                         pos=test_lp.pos,
                                         neg=test_lp.neg)
        
        # Print predictions
        print(
            f"Prediction: {dl_render.render(pred_drill)} | Train Quality: {train_f1_drill:.3f} | Test Quality: {test_f1_drill:.3f} \n")


# Define the cmd arguments
if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='../KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_embeddings", type=str,
                        default='../embeddings/Keci_entity_embeddings.csv')
    parser.add_argument("--num_of_target_concepts",
                        type=int,
                        default=1)
    parser.add_argument("--num_of_training_learning_problems",
                        type=int,
                        default=1)
    parser.add_argument("--path_pretrained_dir", type=str, default=None)

    parser.add_argument("--path_learning_problem", type=str, default='uncle_lp2.json',
                        help="Path to a .json file that contains 2 properties 'positive_examples' and "
                             "'negative_examples'. Each of this properties should contain the IRIs of the respective"
                             "instances. e.g. 'some/path/lp.json'")
    parser.add_argument("--max_runtime", type=int, default=10, help="Max runtime")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds of cross validation.")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    # DQL related
    parser.add_argument("--num_episode", type=int, default=1, help='Number of trajectories created for a given lp.')

    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=2,
                        help='Number of epochs on experience replay memory')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')

    # NN related
    parser.add_argument("--learning_rate", type=int, default=.01)

    start(parser.parse_args())
```

----------------------------------------------------------------------

## Ex. 4: Using Model Adaptor

To simplify the connection between all the
components, there is a
model adaptor available that automatically constructs and connects them.
Here is how to implement the previous example using the [ModelAdapter](ontolearn.mode_adapter.ModelAdapter):

```python
from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.metrics import Accuracy
from ontolearn.model_adapter import ModelAdapter
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.namespaces import Namespaces
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_reasoner import SyncReasoner
from owlapy.render import DLSyntaxObjectRenderer

# Create an reasoner instance
manager = OntologyManager()
onto = manager.load_ontology(IRI.create("KGs/Family/father.owl"))
sync_reasoner = SyncReasoner(onto)

# Define the learning problem
NS = Namespaces('ex', 'http://example.com/father#')
positive_examples = {OWLNamedIndividual(IRI.create(NS, 'stefan')),
                     OWLNamedIndividual(IRI.create(NS, 'markus')),
                     OWLNamedIndividual(IRI.create(NS, 'martin'))}
negative_examples = {OWLNamedIndividual(IRI.create(NS, 'heinz')),
                     OWLNamedIndividual(IRI.create(NS, 'anna')),
                     OWLNamedIndividual(IRI.create(NS, 'michelle'))}

# Define the learning model using ModelAdapter
# Only the class of the learning algorithm is specified
model = ModelAdapter(learner_type=CELOE,
                     reasoner=sync_reasoner,  # (*)
                     path="KGs/Family/father.owl",
                     quality_type=Accuracy,
                     heuristic_type=CELOEHeuristic,  # (*)
                     expansionPenaltyFactor=0.05,
                     startNodeBonus=1.0,
                     nodeRefinementPenalty=0.01,
                     )

# No need to construct the IRI here ourselves
model.fit(pos=positive_examples, neg=negative_examples)

# Create a Description Logics renderer
dlsr = DLSyntaxObjectRenderer()

# Render the hypothesis to DL syntax
for desc in model.best_hypotheses(1):
    print('The result:', dlsr.render(desc.concept), 'has quality', desc.quality)
```

Lines marked with `(*)` are not strictly required as they happen to be
the default choices. For now, you can use ModelAdapter only for EvoLearner, CELOE and OCEL.

-----------------------------------------------------------

In the next guide we will explore the [KnowledgeBase](ontolearn.knowledge_base.KnowledgeBase) class that is needed to 
run a concept learner.

