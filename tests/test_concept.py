""" Test the concept module"""

from core.base import Concept
from core.base import KnowledgeBase


def test_concept():
    # Processes input kb
    path_of_example_kb = 'data/family-benchmark_rich_background.owl'
    kb = KnowledgeBase(path_of_example_kb)
    assert kb.name == 'family-benchmark_rich_background'
    assert len(kb.concepts) > 1
    for k, v in kb.concepts.items():
        assert len(v) == 1
        assert v.instances.issubset(kb.get_all_individuals())