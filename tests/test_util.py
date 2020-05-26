""" Test the util module"""

from core.util import serializer, deserializer
from core.base import KnowledgeBase


def test_serializer_deserializer():
    path_of_example_kb = 'data/family-benchmark_rich_background.owl'
    kb = KnowledgeBase(path_of_example_kb)

    serializer(object_=list(kb.concepts.keys()), path='.', serialized_name='concepts')
    temp = deserializer(path='.', serialized_name='concepts')
    assert list(kb.concepts.keys()) == temp
