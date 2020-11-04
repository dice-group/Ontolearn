""" Test the concept module"""
import json
from ontolearn import KnowledgeBase

with open('../examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
kb = KnowledgeBase(path=settings['data_path'])

def test_concept():
    # Processes input kb
    assert kb.name == 'family-benchmark_rich_background'
    assert len(kb.concepts) > 1
    for _, v in kb.concepts.items():
        assert len(v) == 1
        assert v.instances.issubset(kb.get_all_individuals())
