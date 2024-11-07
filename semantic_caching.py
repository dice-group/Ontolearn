"""python examples/retrieval_eval.py"""
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity, f1_set_similarity, concept_reducer, concept_reducer_properties
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectOneOf,
    OWLObjectComplementOf,
    OWLClass,
)
from owlapy.owl_property import (
    OWLDataProperty,
    OWLObjectInverseOf,
    OWLObjectProperty,
    OWLProperty,
)
import time
from typing import Tuple, Set
import pandas as pd
from owlapy import owl_expression_to_dl
from itertools import chain
import argparse 
import os
import random
import itertools
from owlready2 import *
from collections import OrderedDict




def concept_generator(path_kg):
    # (1) Initialize knowledge base.
    assert os.path.isfile(path_kg)
   
    symbolic_kb = KnowledgeBase(path=path_kg)

    # GENERATE ALCQ CONCEPTS TO EVALUATE RETRIEVAL PERFORMANCES
    # (3) R: Extract object properties.
    object_properties = sorted({i for i in symbolic_kb.get_object_properties()})
    
    object_properties = set(object_properties)    
    
    # (4) R⁻: Inverse of object properties.
    object_properties_inverse = {i.get_inverse_property() for i in object_properties}

    # (5) R*: R UNION R⁻.
    object_properties_and_inverse = object_properties.union(object_properties_inverse)

    # (6) NC: Named owl concepts.
    nc = sorted({i for i in symbolic_kb.get_concepts()})

    nc = set(nc) # return to a set

    # (7) NC⁻: Complement of NC.
    nnc = {i.get_object_complement_of() for i in nc}

    # (8) UNNC: NC UNION NC⁻.
    unnc = nc.union(nnc)

    # (9) Retrieve 10 random Nominals.
    nominals = set(random.sample(symbolic_kb.all_individuals_set(), 3))

    # (10) All Combinations of 3 for Nominals.
    nominal_combinations = set(
        OWLObjectOneOf(combination)
        for combination in itertools.combinations(nominals, 3)
    )
    # (11) NC UNION NC.
    unions = concept_reducer(nc, opt=OWLObjectUnionOf)
    # (12) NC INTERSECTION NC.
    intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
    # (13) UNNC UNION UNNC.
    unions_unnc = concept_reducer(unnc, opt=OWLObjectUnionOf)
    # (14) UNNC INTERACTION UNNC.
    intersections_unnc = concept_reducer(unnc, opt=OWLObjectIntersectionOf)
    # (15) \exist r. C s.t. C \in UNNC and r \in R* .
    exist_unnc = concept_reducer_properties(
        concepts=unnc,
        properties=object_properties,#object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )
    # (16) \forall r. C s.t. C \in UNNC and r \in R* .
    for_all_unnc = concept_reducer_properties(
        concepts=unnc,
        properties=object_properties,#object_properties_and_inverse,
        cls=OWLObjectAllValuesFrom,
    )
    # (17) >= n r. C  and =< n r. C, s.t. C \in UNNC and r \in R* .
    min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3 = (
        concept_reducer_properties(
            concepts=unnc,
            properties=object_properties_and_inverse,
            cls=OWLObjectMinCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3 = (
        concept_reducer_properties(
            concepts=unnc,
            properties=object_properties_and_inverse,
            cls=OWLObjectMaxCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    # (18) \exist r. Nominal s.t. Nominal \in Nominals and r \in R* .
    exist_nominals = concept_reducer_properties(
        concepts=nominal_combinations,
        properties=object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )

    ###################################################################

    # () Converted to list so that the progress bar works.
    concepts = list(
        chain(nc, nnc, unions, intersections,# exist_unnc, for_all_unnc,
            # nc, unions, intersections, nnc, unions_unnc, intersections_unnc,
            # exist_unnc, for_all_unnc,
            # min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3,
            # max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3,
            # exist_nominals,
        )
    )

    return concepts

def concept_retrieval(retriever_func, c) -> Set[str]:

    
    return {i.str for i in retriever_func.individuals(c)}


# Implement the caching mechanism here

def subsumption_based_caching(func):
    # Dictionary to store the cached results
    cache = {}

    def wrapper(*args):
        # Load the ontology
        path_onto = args[1]
        onto = get_ontology(path_onto).load()
        
        # Synchronize the reasoner (e.g., using Pellet)
        with onto:
            sync_reasoner(infer_property_values=True)

        # all individuals
        all_individuals = {a for a in onto.individuals()}

       
        str_expression = owl_expression_to_dl(args[0])
        owl_expression = args[0]

       
        if str_expression in cache:
            return cache[str_expression]

        super_concepts = set()
        for D in list(cache.keys()):
            if D in owl_expression.ancestors(): # this check if expression := C ⊑ D
                super_concepts.add(D)

        # Compute instances
        if len(super_concepts)==0:
            instances = all_individuals
        else:
            # instances = set.intersection(
            #     *[wrapper(D, path_onto) for D in super_concepts]
            # )
            pass

        # Filter instances: Check if each instance belongs to the concept C
        instance_set = set()
        for individual in instances:
            if individual.is_instance_of(owl_expression):
                instance_set.add(individual)

        # Store the result in the cache
        cache[str_expression] = instance_set

        return instance_set


        

    return wrapper








def semantic_caching_size(func, cache_size=5):
    cache = OrderedDict()
    stats = {
        'hits': 0,
        'misses': 0,
        'time': 0
    }

    def wrapper(*args):
        nonlocal stats
        start_time = time.time()

        # Load the ontology
        path_onto = args[1]
        onto = get_ontology(path_onto).load()
        All_individuals = {a.iri for a in list(onto.individuals())}

        # Convert expression to DL format
        str_expression = owl_expression_to_dl(args[0])
        owl_expression = args[0]

        # Function to retrieve cached expression and count hits
        def retrieve_from_cache(expression):
            if expression in cache:
                stats['hits'] += 1
                return cache[expression]
            else:
                stats['misses'] += 1
                return set()

        # Check if the expression itself is in the cache
        result = retrieve_from_cache(str_expression)
        if len(result) > 0:
            return result

        # Handle different OWL expression types and use cache when needed
        if isinstance(owl_expression, OWLClass):
            result = func(*args)
        elif isinstance(owl_expression, OWLObjectComplementOf):
            not_str_expression = str_expression.split("¬")[-1]
            result = All_individuals - retrieve_from_cache(not_str_expression)
        elif isinstance(owl_expression, OWLObjectIntersectionOf):
            C_and_D = [owl_expression_to_dl(i) for i in owl_expression.operands()]
            result = retrieve_from_cache(C_and_D[0]).intersection(retrieve_from_cache(C_and_D[1]))
        elif isinstance(owl_expression, OWLObjectUnionOf):
            C_or_D = [owl_expression_to_dl(i) for i in owl_expression.operands()]
            result = retrieve_from_cache(C_or_D[0]).union(retrieve_from_cache(C_or_D[1]))
        elif isinstance(owl_expression, OWLObjectSomeValuesFrom):
            object_property = owl_expression.get_property()
            filler_expression = owl_expression.get_filler()
            instances = retrieve_from_cache(owl_expression_to_dl(filler_expression)) or set()
            result = set()

            if isinstance(object_property, OWLObjectInverseOf):
                r = onto.search_one(iri=object_property.get_inverse_property().str)
            else:
                r = onto.search_one(iri=object_property.str)

            with onto:
                sync_reasoner()

            for ind_a in All_individuals:
                for ind_b in instances:
                    a = onto.search_one(iri=ind_a)
                    b = onto.search_one(iri=ind_b)
                    if isinstance(object_property, OWLObjectInverseOf):
                        if a in getattr(b, r.name):
                            result.add(a)
                    else:
                        if b in getattr(a, r.name):
                            result.add(ind_a)
        elif isinstance(owl_expression, OWLObjectAllValuesFrom):
            all_values_expr = owl_expression_to_dl(owl_expression)
            some_values_expr = transform_forall_to_exists(all_values_expr)
            result = All_individuals - retrieve_from_cache(some_values_expr)
        else:
            result = func(*args)

        # Add the result to the cache with LIFO eviction strategy
        if len(cache) >= cache_size:
            cache.popitem(last=False)
        cache[str_expression] = result

        stats['time'] += (time.time() - start_time)
        return result

    def transform_forall_to_exists(expression):
        pattern_negated = r'∀ (\w+)\.\(¬(\w+)\)'
        replacement_negated = r'∃ \1.\2'
        pattern_non_negated = r'∀ (\w+)\.(\w+)'
        replacement_non_negated = r'∃ \1.(¬\2)'

        transformed_expression = re.sub(pattern_negated, replacement_negated, expression)
        transformed_expression = re.sub(pattern_non_negated, replacement_non_negated, transformed_expression)

        return transformed_expression

    def get_stats():
        total_requests = stats['hits'] + stats['misses']
        hit_ratio = stats['hits'] / total_requests if total_requests > 0 else 0
        miss_ratio = stats['misses'] / total_requests if total_requests > 0 else 0
        avg_time = stats['time'] / total_requests if total_requests > 0 else 0

        return {
            'hit_ratio': hit_ratio,
            'miss_ratio': miss_ratio,
            'average_time_per_request': avg_time,
            'total_time': stats['time']
        }

    wrapper.get_stats = get_stats
    return wrapper


 
# @semantic_caching_size 
def retrieve(expression:str, path_kg:str, path_kge_model:str) -> Tuple[Set[str], Set[str]]:
    'take a concept c and returns it set of retrieved individual'

    if path_kge_model:
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_neural_embedding=path_kge_model, gamma=0.9
        )
    else:
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb=path_kg, gamma=0.9
        )


    retrievals = concept_retrieval(neural_owl_reasoner, expression) # Retrieving with our reasoner
    
    return retrievals


parser = argparse.ArgumentParser()
parser.add_argument('--cache_size', type=int, default=20)
parser.add_argument('--path_kg', type=str, default="/home/dice/Desktop/Ontolearn/KGs/Family/father.owl")
parser.add_argument('--path_kge', type=str, default=None)
args = parser.parse_args()


cached_retriever = semantic_caching_size(retrieve, cache_size=args.cache_size)




symbolic_kb = KnowledgeBase(path=args.path_kg)

D = []
Avg_jaccard = []
alc_concepts = concept_generator(args.path_kg)
total_time_ebr = 0

for expr in alc_concepts:

    A  = cached_retriever(expr, args.path_kg, args.path_kge) # Caching retrieval
    ground_truth = concept_retrieval(symbolic_kb, expr)

    time_start = time.time()
    retrieve_ebr = retrieve(expr, args.path_kg, args.path_kge)
    time_ebr = time.time()-time_start
    total_time_ebr += time_ebr

    jacc = jaccard_similarity(A,ground_truth)
    Avg_jaccard.append(jacc)
    D.append({'Expression':owl_expression_to_dl(expr), "Jaccard Similarity":jacc, 'Retrieval_caching':A, "Retrieval_true":ground_truth,\
              "time_ebr":time_ebr})
    print(f'Expression: {owl_expression_to_dl(expr)}')
    print(f'Jaccard similarity: {jacc}')
     

stats = cached_retriever.get_stats()
data = args.path_kg.split("/")[-1]
print('-'*50)
print("Cache Statistics:")
print(f"Hit Ratio: {stats['hit_ratio']:.2f}")
print(f"Miss Ratio: {stats['miss_ratio']:.2f}")
print(f"Average Time per Request: {stats['average_time_per_request']:.4f} seconds")
print(f"Total Time with Caching: {stats['total_time']:.4f} seconds")
print(f"Total Time Without Caching: {total_time_ebr:.4f} seconds")
print(f"Total number of concepts: {len(alc_concepts)}")
print(f"Average Jaccard for the {data} dataset", sum(Avg_jaccard)/len(Avg_jaccard))




   
# Data = pd.DataFrame(D)
# Data.to_csv('analyse_cache.csv')
