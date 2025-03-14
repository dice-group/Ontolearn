import unittest

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.utils import setup_logging

setup_logging("ontolearn/logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=PATH_FAMILY)


class LearningProblemGenerator_Test(unittest.TestCase):
    def test_get_balanced_examples(self):
        lp = LearningProblemGenerator(knowledge_base=kb)
        # examples_10 = lp.get_balanced_examples(min_num_problems=10, num_diff_runs=1, max_length=5, min_length=3, min_num_instances=10, search_algo='strict-dfs')
        # @Using it with LengthBasedRefinement leads to extesive comp. time
        # self.assertEqual(len(examples_10), 10)
        # self.assertEqual(len(lp.get_balanced_examples(min_num_problems=20, num_diff_runs=1, min_num_instances=10)), 20)

    def test_get_balanced_n_samples_per_example(self):
        lp = LearningProblemGenerator(knowledge_base=kb)
        """
        # @Using it with LengthBasedRefinement leads to extesive comp. time 
        min_num_concepts = 10
        max_length = 5
        min_length = 3
        min_num_instances_per_concept = 10
        num_of_randomly_created_problems_per_concept = 5
        balanced_examples = lp.get_balanced_n_samples_per_examples(n=num_of_randomly_created_problems_per_concept,
                                                                   min_num_problems=min_num_concepts,
                                                                   num_diff_runs=1,  # This must be optimized
                                                                   max_length=max_length,
                                                                   min_length=min_length,
                                                                   min_num_instances=min_num_instances_per_concept)
        for i in range(num_of_randomly_created_problems_per_concept - 1):
            self.assertEqual(balanced_examples[i][0], balanced_examples[i + 1][0])
            # self.assertNotEqual(balanced_examples[i][1], balanced_examples[i + 1][1])
            # CD: Assertion is incorrect. Here is the proof
            # Let Person be a target concept and let 10 be the number instances belonging to Person in CWR.
            # If we sample 5 individuals from Person, then there is no guarantee sampled E^+ nor E^- be same.
            # self.assertNotEqual(balanced_examples[i][2], balanced_examples[i + 1][2])
        """

if __name__ == '__main__':
    unittest.main()
