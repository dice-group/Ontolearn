""" Test the base module"""
# import os

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.utils import setup_logging

setup_logging("logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
PATH_FATHER = 'KGs/father.owl'


def test_knowledge_base():
    KnowledgeBase(path=PATH_FAMILY)
    # assert kb.name == 'family-benchmark_rich_background'
    #
    # assert kb.property_hierarchy
    # assert kb.property_hierarchy.all_properties
    # assert len(kb.property_hierarchy.all_properties) >= \
    #        len(kb.property_hierarchy.data_properties)
    # assert len(kb.property_hierarchy.all_properties) >= \
    #        len(kb.property_hierarchy.object_properties)


def test_multiple_knowledge_bases():
    KnowledgeBase(path=PATH_FAMILY)

    # There should not be an exception here
    # (that refers to the family ontology)


# def test_knowledge_base_save():
#     kb = KnowledgeBase(path=PATH_FAMILY)
#     kb.save('test_kb_save', rdf_format='nt')
#     assert os.stat('test_kb_save.nt').st_size > 0
#     os.remove('test_kb_save.nt')


if __name__ == '__main__':
    test_knowledge_base()
    test_multiple_knowledge_bases()
    # test_knowledge_base_save()
