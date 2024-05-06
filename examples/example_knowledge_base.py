from ontolearn.concept_generator import ConceptGenerator
from ontolearn.knowledge_base import KnowledgeBase
import os

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

kb_path = '../KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=kb_path)
generator = ConceptGenerator()

# All concepts.
for i in kb.get_concepts():
    print(i)
print('*' * 100)

# All object properties.
for i in kb.get_object_properties():
    print(i)
print('*' * 100)

# All individuals.
for i in kb.all_individuals_set():
    print(i)
print('*' * 100)

# Count of individuals for each class
for i in kb.get_concepts():
    print(f'{i} ==> {kb.individuals_count(i)}')
print('*' * 100)


# IRIs of all individuals.
for i in kb.all_individuals_set():
    print(i.str)
print('*' * 100)

# Direct concept hierarchy from Top to Bottom.
for concept in kb.class_hierarchy.items():
    print(f'{concept.str} => {[c.str for c in kb.get_direct_sub_concepts(concept)]}')
print('*' * 100)


for concept in kb.most_general_existential_restrictions(domain=generator.thing):
    print(concept)

print('*' * 100)
for concept in kb.most_general_universal_restrictions(domain=generator.thing):
    print(concept)

print('*' * 100)
for concept in kb.most_general_existential_restrictions(domain=generator.nothing):
    print(concept)

print('*' * 100)
for concept in kb.most_general_universal_restrictions(domain=generator.nothing):
    print(concept)

print('*' * 100)
for c in kb.get_concepts():
    neg_c = generator.negation(c)
    neg_neg_c = generator.negation(neg_c)
    assert neg_neg_c == c
