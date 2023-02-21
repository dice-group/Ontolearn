from ontolearn.knowledge_base import KnowledgeBase
import os

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

kb_path = '../KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=kb_path)


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
    print(i.get_iri().as_str())
print('*' * 100)

# Direct concept hierarchy from Top to Bottom.
for concept in kb.class_hierarchy().items():
    print(f'{concept.get_iri().as_str()} => {[c.get_iri().as_str() for c in kb.get_direct_sub_concepts(concept)]}')
print('*' * 100)


for concept in kb.most_general_existential_restrictions(domain=kb.thing):
    print(concept)

print('*' * 100)
for concept in kb.most_general_universal_restrictions(domain=kb.thing):
    print(concept)

print('*' * 100)
for concept in kb.most_general_existential_restrictions(domain=kb.nothing):
    print(concept)

print('*' * 100)
for concept in kb.most_general_universal_restrictions(domain=kb.nothing):
    print(concept)

print('*' * 100)
for c in kb.get_concepts():
    neg_c = kb.negation(c)
    neg_neg_c = kb.negation(neg_c)
    assert neg_neg_c == c
