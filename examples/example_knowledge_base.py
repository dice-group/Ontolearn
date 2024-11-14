from ontolearn.knowledge_base import KnowledgeBase
import argparse

def example(args):
    kb = KnowledgeBase(path=args.path_kb)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path_kb", type=str, default="KGs/Family/family-benchmark_rich_background.owl")
    example(parser.parse_args())