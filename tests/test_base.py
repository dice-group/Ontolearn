""" Test the base module"""

from ontolearn import KnowledgeBase


def test_knowledge_base():
    path_of_example_kb = '../data/family-benchmark_rich_background.owl'
    kb = KnowledgeBase(path_of_example_kb)
    assert kb.name == 'family-benchmark_rich_background'

    assert kb.property_hierarchy
    assert kb.property_hierarchy.all_properties
    assert len(kb.property_hierarchy.all_properties) >=\
        len(kb.property_hierarchy.data_properties)
    assert len(kb.property_hierarchy.all_properties) >=\
        len(kb.property_hierarchy.object_properties)
