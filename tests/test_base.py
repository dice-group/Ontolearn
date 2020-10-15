""" Test the base module"""

from ontolearn import KnowledgeBase

PATH_FAMILY = 'data/family-benchmark_rich_background.owl'
PATH_FATHER = 'data/father.owl'
PATH_EMPTY = 'data/empty.owl'


def test_knowledge_base():
    kb = KnowledgeBase(PATH_FAMILY)
    assert kb.name == 'family-benchmark_rich_background'

    assert kb.property_hierarchy
    assert kb.property_hierarchy.all_properties
    assert len(kb.property_hierarchy.all_properties) >=\
        len(kb.property_hierarchy.data_properties)
    assert len(kb.property_hierarchy.all_properties) >=\
        len(kb.property_hierarchy.object_properties)


def test_multiple_knowledge_bases():
    kb = KnowledgeBase(PATH_FAMILY)

    # There should not be an exception here (that refers to the family ontology)
    kb = KnowledgeBase(PATH_EMPTY)