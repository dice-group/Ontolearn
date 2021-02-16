import json
from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.utils import setup_logging

setup_logging("logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(PATH_FAMILY)


class TestLearningProblemGenerator:
    def test_get_balanced_examples(self):
        lp = LearningProblemGenerator(knowledge_base=kb)
        assert len(lp.get_balanced_examples(min_num_problems=10, num_diff_runs=1, max_length=5, min_length=3,
                                            min_num_instances=10, search_algo='strict-dfs')) == 10
        assert len(lp.get_balanced_examples(min_num_problems=20, num_diff_runs=1, min_num_instances=10)) == 20
