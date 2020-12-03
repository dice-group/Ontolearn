import json
from ontolearn import KnowledgeBase, LearningProblemGenerator

PATH_FAMILY = 'data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(PATH_FAMILY)


class TestLearningProblemGenerator:
    def test_get_balanced_examples(self):
        lp = LearningProblemGenerator(knowledge_base=kb, num_diff_runs=10)

        assert len(lp.get_concepts(num_problems=10, search_algo='strict-dfs')) == 10
        assert len(lp.get_concepts(num_problems=10, search_algo='dfs')) >= 10

        assert len(lp.get_balanced_examples(num_problems=10, min_num_instances=10, search_algo='strict-dfs')) == 10
        assert len(lp.get_balanced_examples(num_problems=10, min_num_instances=10, search_algo='dfs')) >= 10
