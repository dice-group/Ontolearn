import json
from ontolearn import KnowledgeBase, LearningProblemGenerator

PATH_FAMILY = 'data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(PATH_FAMILY)


class TestLearningProblemGenerator:
    def test_get_balanced_examples(self):
        lp = LearningProblemGenerator(knowledge_base=kb)
        assert len(lp.get_balanced_examples(num_problems=10, num_diff_runs=1,min_num_instances=10)) == 10