import unittest

from owlapy.class_expression import OWLClass
from owlapy.iri import IRI

from ontolearn.search import DRILLSearchTreePriorityQueue, RL_State


def _make_state(name: str, quality: float, heuristic: float) -> RL_State:
    # Each node needs a distinct concept: RL_State/DRILLSearchTreePriorityQueue
    # key nodes by their DL string representation, so identical concepts would
    # collapse into a single tree entry.
    concept = OWLClass(IRI.create(f'http://example.com/{name}'))
    state = RL_State(concept, is_root=True)
    state.quality = quality
    state.heuristic = heuristic
    return state


class DRILLSearchTreePriorityQueueTest(unittest.TestCase):

    def setUp(self):
        self.tree = DRILLSearchTreePriorityQueue(verbose=0)
        self.tree.add(_make_state('A', quality=0.2, heuristic=0.9))
        self.tree.add(_make_state('B', quality=0.8, heuristic=0.1))
        self.tree.add(_make_state('C', quality=0.5, heuristic=0.5))

    def test_get_top_n_does_not_raise(self):
        # Regression test for ontolearn/search.py:798, where get_top_n()
        # referenced the never-assigned self.refined_nodes and tried to
        # add it to a dict_values object, raising AttributeError/TypeError.
        top_n = self.tree.get_top_n(2, key='quality')
        self.assertEqual(len(top_n), 2)

    def test_get_top_n_orders_by_quality(self):
        top_n = self.tree.get_top_n(3, key='quality')
        self.assertEqual([node.quality for node in top_n], [0.8, 0.5, 0.2])

    def test_get_top_n_orders_by_heuristic(self):
        top_n = self.tree.get_top_n(3, key='heuristic')
        self.assertEqual([node.heuristic for node in top_n], [0.9, 0.5, 0.1])

    def test_get_top_n_limits_result_size(self):
        top_n = self.tree.get_top_n(1, key='quality')
        self.assertEqual(len(top_n), 1)


if __name__ == '__main__':
    unittest.main()
