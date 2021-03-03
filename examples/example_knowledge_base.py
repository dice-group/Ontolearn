from ontolearn import KnowledgeBase
import os

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

kb_path = '../KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(kb_path)
kb.save('family-benchmark_rich_background', rdf_format='nt')

# All concepts.
for i in kb.get_all_concepts():
    print(i)
print('*' * 100)

# All individuals.
for i in kb.individuals:
    print(i)
print('*' * 100)

# URIs of all individuals.
for i in kb.convert_owlready2_individuals_to_uri_from_iterable(kb.individuals):
    print(i)
print('*' * 100)

# Direct concept  hierarchy from Top to Bottom.
for concept, direct_sub_concepts in kb.top_down_direct_concept_hierarchy.items():
    print(f'{concept.str} => {[i.str for i in direct_sub_concepts]}')
print('*' * 100)

# Concept  hierarchy from Top to Bottom.
for concept, direct_sub_concepts in kb.top_down_concept_hierarchy.items():
    print(f'{concept.str} => {[i.str for i in direct_sub_concepts]}')

print('*' * 100)

# Direct concept  hierarchy from Bottom to Top.
for concept, direct_sub_concepts in kb.top_down_direct_concept_hierarchy.items():
    print(f'{concept.str} => {[i.str for i in direct_sub_concepts]}')
print('*' * 100)

# Concept  hierarchy from Bottom to Top.
for concept, direct_sub_concepts in kb.down_top_concept_hierarchy.items():
    print(f'{concept.str} => {[i.str for i in direct_sub_concepts]}')

print('*' * 100)
for concept in kb.most_general_existential_restrictions(kb.thing):
    print(concept.str)

print('*' * 100)
for concept in kb.most_general_universal_restrictions(kb.thing):
    print(concept.str)

print('*' * 100)
for concept in kb.most_general_existential_restrictions(kb.nothing):
    print(concept.str)

print('*' * 100)
for concept in kb.most_general_universal_restrictions(kb.nothing):
    print(concept.str)

# CD: We should add this into our tests.
print('*' * 100)
for c in kb.get_all_concepts():
    neg_c = kb.negation(c)
    neg_neg_c = kb.negation(neg_c)
    assert neg_neg_c == c
