import itertools
import os
import tempfile
import unittest

owlready2 = None
try:
    import owlready2
except ImportError:  # owlready2 is an optional extra
    pass

from ontolearn.incomplete_kb import make_kb_incomplete, make_kb_incomplete_ass, make_kb_inconsistent  # noqa: E402


_counter = itertools.count()


def _build_ontology(path: str, n: int = 20) -> None:
    # The functions under test load into owlready2's shared default world, so every test needs its own
    # ontology IRI; otherwise a previous test's already-mutated ontology would be reused.
    onto = owlready2.get_ontology(f"http://example.org/inc{next(_counter)}#")
    with onto:
        class Person(owlready2.Thing):
            pass

        class Place(owlready2.Thing):
            pass

        class knows(owlready2.ObjectProperty):
            pass

        class age(owlready2.DataProperty):
            pass

        people = [Person(f"p{i}") for i in range(n)]
        for i, p in enumerate(people):
            p.knows.append(people[(i + 1) % n])
            p.knows.append(people[(i + 2) % n])
            p.age = [i]
        Place("somewhere")
    onto.save(file=path, format="rdfxml")


def _load(path):
    return owlready2.get_ontology(path).load()


@unittest.skipIf(owlready2 is None, "owlready2 not installed")
class TestIncompleteKB(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "kb.owl")
        self.out = os.path.join(self.tmp.name, "out.owl")
        _build_ontology(self.src)
        self.n_individuals = len(list(_load(self.src).individuals()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_remove_individuals(self):
        make_kb_incomplete(self.src, self.out, rate=50, seed=1)
        self.assertTrue(os.path.exists(self.out))
        remaining = len(list(_load(self.out).individuals()))
        self.assertEqual(remaining, self.n_individuals - int(self.n_individuals * 0.5))

    def test_remove_individuals_rate_zero_keeps_everything(self):
        make_kb_incomplete(self.src, self.out, rate=0, seed=1)
        self.assertEqual(len(list(_load(self.out).individuals())), self.n_individuals)

    def test_remove_statements(self):
        def count(p):
            return sum(len(prop[i]) for i in _load(p).individuals() for prop in i.get_properties())

        before = count(self.src)
        make_kb_incomplete_ass(self.src, self.out, rate=50, seed=1)
        after = count(self.out)
        self.assertGreater(before, 0)
        self.assertLess(after, before)
        # all individuals are kept; only statements are dropped
        self.assertEqual(len(list(_load(self.out).individuals())), self.n_individuals)

    def test_make_inconsistent_returns_added_statements(self):
        added = make_kb_inconsistent(self.src, self.out, rate=10, seed=3)
        self.assertTrue(os.path.exists(self.out))
        self.assertGreater(len(added), 0)
        self.assertTrue(all(s.startswith("Added incorrect") for s in added))

    def test_make_inconsistent_rate_zero_adds_nothing(self):
        added = make_kb_inconsistent(self.src, self.out, rate=0, seed=3)
        self.assertEqual(added, [])


if __name__ == "__main__":
    unittest.main()
