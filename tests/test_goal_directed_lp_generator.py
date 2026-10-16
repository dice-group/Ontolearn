import unittest

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.lp_generator import ComplexityProfile, DLConstruct, GoalDirectedLPGenerator

PATH_FAMILY = "KGs/Family/family-benchmark_rich_background.owl"


class TestGoalDirectedLPGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase(path=PATH_FAMILY)
        cls.profile = ComplexityProfile(
            name="conj_exists",
            allowed_constructs=frozenset({DLConstruct.ATOMIC, DLConstruct.INTERSECTION, DLConstruct.EXISTENTIAL}),
            max_length=8, max_depth=2, min_pos=2, min_neg=2, oversample_factor=2,
        )

    def test_generate_respects_profile(self):
        gen = GoalDirectedLPGenerator(self.kb, seed=1)
        lps = gen.generate(self.profile, num_lps=3, max_attempts=500)
        self.assertGreater(len(lps), 0)
        self.assertLessEqual(len(lps), 3)
        for lp in lps:
            self.assertEqual(lp.profile_name, "conj_exists")
            self.assertTrue(self.profile.min_length <= lp.length <= self.profile.max_length)
            self.assertLessEqual(lp.depth, self.profile.max_depth)
            self.assertGreaterEqual(len(lp.pos), self.profile.min_pos)
            self.assertGreaterEqual(len(lp.neg), self.profile.min_neg)
            self.assertFalse(set(lp.pos) & set(lp.neg))
            self.assertLessEqual(lp.baseline_f1, self.profile.max_baseline_f1)
            self.assertEqual(len(lp.to_lp().pos), len(lp.pos))

    def test_seed_is_reproducible(self):
        a = GoalDirectedLPGenerator(self.kb, seed=7).generate(self.profile, num_lps=2, max_attempts=500)
        b = GoalDirectedLPGenerator(self.kb, seed=7).generate(self.profile, num_lps=2, max_attempts=500)
        self.assertEqual([x.dl for x in a], [x.dl for x in b])


if __name__ == "__main__":
    unittest.main()
