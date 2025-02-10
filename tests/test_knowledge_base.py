from ontolearn.knowledge_base import KnowledgeBase
import argparse

class TestKnowledgeBase:
    def test_reading_data(self):
        kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        # All concepts.
        for i in kb.get_concepts():
            print(i)
        print('*' * 100)
        # All object properties.
        for i in kb.get_object_properties():
            print(i)
        print('*' * 100)
        # All individuals.
        for i in kb.individuals():
            print(i)
        print('*' * 100)
        # Count of individuals for each class
        for i in kb.get_concepts():
            print(f'{i} ==> {kb.individuals_count(i)}')
        print('*' * 100)
        # IRIs of all individuals.
        for i in kb.individuals():
            print(i.str)
        print('*' * 100)
        # Direct concept hierarchy from Top to Bottom.
        for concept in kb.class_hierarchy.items():
            print(f'{concept.str} => {[c.str for c in kb.get_direct_sub_concepts(concept)]}')
