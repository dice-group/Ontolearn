""" Test the util module"""

from core.util import serializer, deserializer
from core.base import KnowledgeBase


def test_serializer_deserializer():
    m=list(range(10))

    serializer(object_=m, path='.', serialized_name='concepts')
    temp = deserializer(path='.', serialized_name='concepts')
    assert m == temp
