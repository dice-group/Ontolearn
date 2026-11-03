import json

from deap import gp
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard


def test_reported_values_belong_to_the_reported_hypothesis():
    with open('examples/synthetic_problems.json') as stream:
        settings = json.load(stream)
    examples = settings['problems']['Uncle']
    lp = PosNegLPStandard(pos={OWLNamedIndividual(i) for i in examples['positive_examples']},
                          neg={OWLNamedIndividual(i) for i in examples['negative_examples']})
    model = EvoLearner(knowledge_base=KnowledgeBase(path=settings['data_path'][3:]), population_size=60,
                       num_generations=4, max_runtime=600, terminate_on_goal=False)
    model.fit(lp)

    population = model._result_population
    best = max(population, key=lambda ind: ind.fitness.values[0])
    other = next(ind for ind in sorted(population, key=lambda ind: -ind.fitness.values[0])
                 if ind.quality.values[0] != best.quality.values[0])
    # Duplicates of the best individual must not shift the values reported for the next distinct one.
    model._result_population = [best, model.toolbox.clone(best), model.toolbox.clone(best), other]

    first, second = model.best_hypotheses(n=2, return_node=True)

    assert first.quality == best.quality.values[0]
    assert second.concept == gp.compile(other, model.pset)
    assert second.quality == other.quality.values[0]
    assert second.tree_length == len(other)
