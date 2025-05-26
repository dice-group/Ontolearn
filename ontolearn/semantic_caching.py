# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""python examples/retrieval_eval.py"""
from owlapy.owl_literal import OWLBottomObjectProperty, OWLTopObjectProperty

from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.utils import jaccard_similarity, concept_reducer, concept_reducer_properties
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
from owlapy.owl_property import OWLObjectInverseOf
import time
from typing import Tuple, Set
from owlapy import owl_expression_to_dl
from itertools import chain
import os
import random
import itertools
from owlready2 import *
from collections import OrderedDict
from owlapy.owl_reasoner import SyncReasoner
import pickle
from tqdm import tqdm


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

    # (9) Retrieve 3 random Nominals.
    inds = list(symbolic_kb.individuals())
    nominals = set(random.sample(inds, 3))

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
    random.seed(0)
    if len(intersections_unnc) > 500:
        intersections_unnc = random.sample(intersections_unnc, k=500)
    if len(unions_unnc) > 500:
        unions_unnc = random.sample(unions_unnc, k=500)
    if len(exist_unnc) > 200:
        exist_unnc = set(list(exist_unnc)[:200])  
    if len(for_all_unnc) > 200:
        for_all_unnc = set(list(for_all_unnc)[:200])  

    concepts = list(
        chain(nc, nnc, unions_unnc, intersections_unnc, exist_unnc, for_all_unnc,
        )
    )
    return concepts




def get_saved_concepts(path_kg, data_name, shuffle):
    """Shuffle or not the generated concept and save it in a folder for reproducibility."""
    
    # Create the directory if it does not exist
    cache_dir = f"caching_results_{data_name}"
    os.makedirs(cache_dir, exist_ok=True)

    # Determine the filename based on shuffle flag
    filename = "shuffled_concepts.pkl" if shuffle else "unshuffled_concepts.pkl"
    save_file = os.path.join(cache_dir, filename)

    if os.path.exists(save_file):
        with open(save_file, "rb") as f:
            alc_concepts = pickle.load(f)
        print(f"Loaded concepts from {filename}.")
    else:
        # Generate concepts and optionally shuffle
        alc_concepts = concept_generator(path_kg)
        if shuffle:
            random.seed(0)
            random.shuffle(alc_concepts)

        # Save the concepts
        with open(save_file, "wb") as f:
            pickle.dump(alc_concepts, f)

        print(f"Generated and saved {'shuffled' if shuffle else 'unshuffled'} concepts.")
    
    return alc_concepts


def concept_retrieval(retriever_func, c) -> Set[str]:
    return {i.str for i in retriever_func.individuals(c)}


class CacheWithEviction:
    def __init__(self, cache_size, strategy='LIFO', random_seed=10):
        self.cache = OrderedDict()  # Store the actual cache
        self.access_times = {}      # Track last access times for LRU and MRU
        self.cache_size = cache_size
        self.strategy = strategy
        self.random_seed = random_seed 
        self.initialized = False  # Track if cache is already initialized

    def _evict(self):
        '''empty the cache when it is full using different strategy'''
        if len(self.cache) > self.cache_size:
            if self.strategy == 'FIFO':
                self.cache.popitem(last=False)  # Evict the oldest item (first in)
            elif self.strategy == 'LIFO':
                self.cache.popitem(last=True)  # Evict the most recently added item
            elif self.strategy == 'LRU':
                # Evict the least recently used item based on `access_times`
                lru_key = min(self.access_times, key=self.access_times.get)
                del self.cache[lru_key]
                del self.access_times[lru_key]
            elif self.strategy == 'MRU':
                # Evict the most recently used item based on `access_times`
                mru_key = max(self.access_times, key=self.access_times.get)
                del self.cache[mru_key]
                del self.access_times[mru_key]
            elif self.strategy == 'RP':
                # Random eviction
                random.seed(self.random_seed) 
                random_key = random.choice(list(self.cache.keys()))
                del self.cache[random_key]
                self.access_times.pop(random_key, None)

    def get(self, key):
        """
        Retrieve an item from the cache. Updates access time for LRU/MRU.
        """
        if key in self.cache:
            if self.strategy in ['LRU', 'MRU']:
                self.access_times[key] = time.time()  # Update access timestamp
            return self.cache[key]
        return None

    def put(self, key, value):
        """
        Add an item to the cache. Evicts an entry if the cache is full.
        """
        if key in self.cache:
            del self.cache[key]  # Remove existing entry to re-insert and maintain order

        self._evict()  # Evict if necessary

        self.cache[key] = value
        if self.strategy in ['LRU', 'MRU']:
            self.access_times[key] = time.time()  # Record access timestamp

    def initialize_cache(self, func, path_onto, third, All_individuals, handle_restriction_func, concepts):
        """
        Initialize the cache with precomputed results for OWLClass and Existential concepts.
        :param ontology: The loaded ontology.
        :param func: Function to retrieve individuals for a given expression.
        :param concepts: List of OWL concepts to precompute and store instances for.
        """
        if self.initialized:
            return

        # Filter OWLClass and OWLObjectSomeValuesFrom concepts
        class_concepts = [concept for concept in concepts if isinstance(concept, OWLClass)]
        negated_class_concepts = [concept for concept in concepts if isinstance(concept, OWLObjectComplementOf)]
        existential_concepts = [concept for concept in concepts if isinstance(concept, OWLObjectSomeValuesFrom)]
        
        # Process OWLClass concepts
        for cls in tqdm(class_concepts, desc=f"Adding OWLClass concepts"):
            concept_str = owl_expression_to_dl(cls)
            self.put(concept_str, func(cls, path_onto, third))

        for negated_cls in tqdm(negated_class_concepts, desc=f"Adding Complement concepts"):
            # Compute and store complement
            negated_cls_str = owl_expression_to_dl(negated_cls)
            cached = self.cache.get(negated_cls_str.split("¬")[-1])
            if cached is None:
                cached = func(negated_cls, path_onto, third)
            neg = All_individuals - cached
            self.put(negated_cls_str, neg)
            
        # Process Existential concepts
        for existential in tqdm(existential_concepts, desc=f"Adding Existential concepts"):
            existential_str = owl_expression_to_dl(existential)
            self.put(existential_str, handle_restriction_func(existential))
            
        self.initialized = True

        
    def get_all_items(self):
        return list(self.cache.keys())
    
    def is_full(self):
        """Check if the cache is full."""
        return len(self.cache) >= self.max_size
    

def semantic_caching_size(func, cache_size, eviction_strategy, random_seed, cache_type, concepts):

    '''This function implements the semantic caching algorithm for ALC concepts as presented in the paper'''

    cache = CacheWithEviction(cache_size, strategy=eviction_strategy, random_seed=random_seed)  # Cache for instances
    loaded_ontologies = {} #Cache for ontologies
    loaded_individuals = {} #cache for individuals
    cache_type = cache_type
    concepts = concepts
    stats = {
        'hits': 0,
        'misses': 0,
        'time': 0
    }
    time_initialization = 0
    
    def wrapper(*args):
        nonlocal stats
        nonlocal time_initialization

        # Load ontology and individuals if not already cached
        path_onto = args[1]
        if path_onto not in loaded_ontologies:
            loaded_ontologies[path_onto] = get_ontology(path_onto).load()
            loaded_individuals[path_onto] = {a.iri for a in list(loaded_ontologies[path_onto].individuals())}
        onto = loaded_ontologies[path_onto]
        All_individuals = loaded_individuals[path_onto]

        str_expression = owl_expression_to_dl(args[0])
        owl_expression = args[0]

        # Function to retrieve cached expression and count hits
        def retrieve_from_cache(expression):
            cached_result = cache.get(expression)
            if cached_result is not None:
                stats['hits'] += 1
                return cached_result
            else: 
                stats['misses'] += 1
                return None
            
        def handle_owl_some_values_from(owl_expression):
            """
            Process the OWLObjectSomeValuesFrom expression locally.
            When called, return the retrieval of OWLObjectSomeValuesFrom
            based on the Algorithm described in the paper
            """
            
            if len(All_individuals)<1000: # The loop beomes unscalable when there are too many individuals 
                object_property = owl_expression.get_property()
                if object_property == OWLBottomObjectProperty or object_property == OWLTopObjectProperty:
                    return set()
                filler_expression = owl_expression.get_filler()
                instances = retrieve_from_cache(owl_expression_to_dl(filler_expression))
                if instances is not None:
                    result = set()
                    if isinstance(object_property, OWLObjectInverseOf):
                        r = onto.search_one(iri=object_property.get_inverse_property().str)
                    else:
                        r = onto.search_one(iri=object_property.str)
                    individual_map = {ind: onto.search_one(iri=ind) for ind in All_individuals | instances}
                    for ind_a in All_individuals:
                        a = individual_map[ind_a]
                        for ind_b in instances:
                            b = individual_map[ind_b]
                            if isinstance(object_property, OWLObjectInverseOf):
                                if a in getattr(b, r.name):
                                    result.add(a)
                            else:
                                if b in getattr(a, r.name):
                                    result.add(ind_a) 
                else:
                    result = func(*args)
            else:
                result = func(*args)
            return result

        start_time = time.time() #state the timing before the cache initialization 

        # Cold cache initialization
        start_time_initialization = time.time()
        if cache_type == 'cold' and not cache.initialized:
            cache.initialize_cache(func, path_onto, args[-1], All_individuals, handle_owl_some_values_from, concepts)
        time_initialization = time.time()- start_time_initialization

        # start_time = time.time() #state the timing after the cache initialization 

        # Handle different OWL expression types and use cache when needed
        if isinstance(owl_expression, OWLClass):
            cached_result = retrieve_from_cache(str_expression)
            result = cached_result if cached_result is not None else func(*args)

        elif isinstance(owl_expression, OWLObjectComplementOf):
            if cache_type == 'cold': #If it is cold then all complement object are already cached at initialisation time
                cached_result_cold = retrieve_from_cache(str_expression)
                result =  cached_result_cold if cached_result_cold is not None else func(*args)
            else: 
                not_str_expression = str_expression.split("¬")[-1]
                cached_result = retrieve_from_cache(not_str_expression)
                result = (All_individuals - cached_result) if cached_result is not None else func(*args)

        elif isinstance(owl_expression, OWLObjectIntersectionOf):
            C_and_D = [owl_expression_to_dl(i) for i in owl_expression.operands()]
            cached_C = retrieve_from_cache(C_and_D[0])
            cached_D = retrieve_from_cache(C_and_D[1])
            if cached_C is not None and cached_D is not None:
                result = cached_C.intersection(cached_D)
            else:
                result = func(*args)

        elif isinstance(owl_expression, OWLObjectUnionOf):
            C_or_D = [owl_expression_to_dl(i) for i in owl_expression.operands()]
            cached_C = retrieve_from_cache(C_or_D[0])
            cached_D = retrieve_from_cache(C_or_D[1])
            if cached_C is not None and cached_D is not None:
                result = cached_C.union(cached_D)
            else:
                result = func(*args)
                
        elif isinstance(owl_expression, OWLObjectSomeValuesFrom):
            if cache_type == 'cold':
                cached_result_cold = retrieve_from_cache(str_expression)
                if cached_result_cold is not None:
                    result = cached_result_cold
                else:
                    result = handle_owl_some_values_from(owl_expression)   
            else:
               result = handle_owl_some_values_from(owl_expression)

        elif isinstance(owl_expression, OWLObjectAllValuesFrom):
            all_values_expr = owl_expression_to_dl(owl_expression)
            some_values_expr = transform_forall_to_exists(all_values_expr)
            cached_result = retrieve_from_cache(some_values_expr)
            result = (All_individuals - cached_result) if cached_result is not None else func(*args)
    
        else:
            result = func(*args)

        stats['time'] += (time.time() - start_time)
        cache.put(str_expression, result)
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
            'total_time': stats['time'],
            'time_initialization': time_initialization
        }

    wrapper.get_stats = get_stats
    return wrapper




def non_semantic_caching_size(func, cache_size):
    '''This function implements a caching algorithm for ALC concepts without semantics.'''
    cache = OrderedDict()  # Cache for instances
    stats = {
        'hits': 0,
        'misses': 0,
        'time': 0
    }
    
    def wrapper(*args):
        nonlocal stats
        str_expression = owl_expression_to_dl(args[0])  

        def retrieve_from_cache(expression):
            if expression in cache:
                # Move the accessed item to the end to mark it as recently used
                cache.move_to_end(expression)
                stats['hits'] += 1
                return cache[expression]
            else:
                stats['misses'] += 1
                return None

        # Start timing before cache access and function execution
        start_time = time.time()
        
        # Try to retrieve the result from the cache If result is in cache, return it directly
        cached_result = retrieve_from_cache(str_expression)
        if cached_result is not None:
            stats['time'] += (time.time() - start_time)
            return cached_result
        
        # Compute the result and store it in the cache
        result = func(*args)
        cache[str_expression] = result
        
        # Apply LRU strategy: remove the least recently used item if the cache exceeds its size
        if len(cache) > cache_size:
            cache.popitem(last=False)  

        stats['time'] += (time.time() - start_time)
        return result
    
    # Function to get cache statistics
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



def retrieve(expression:str, path_kg:str, path_kge_model:str) -> Tuple[Set[str], Set[str]]:
    '''Retrieve instances with neural reasoner'''
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


def retrieve_other_reasoner(expression, path_kg, name_reasoner='HermiT'): 
    '''Retrieve instances with symbolic reasoners'''
    
    reasoner = SyncReasoner(path_kg, reasoner=name_reasoner)
   
    if reasoner.has_consistent_ontology():
        return {i.str for i in (reasoner.instances(expression, direct=False))}
    else:
        print("The knowledge base is not consistent") 
         

def run_semantic_cache(path_kg:str, path_kge:str, cache_size:int, name_reasoner:str, eviction:str, random_seed:int, cache_type:str, shuffle_concepts:bool):
    '''Return cache performnace with semantics'''

    symbolic_kb = KnowledgeBase(path=path_kg)
    D = []
    Avg_jaccard = []
    Avg_jaccard_reas = []
    data_name = path_kg.split("/")[-1].split("/")[-1].split(".")[0]

    if shuffle_concepts:
        alc_concepts = get_saved_concepts(path_kg, data_name=data_name, shuffle=True) 
    else:
        alc_concepts = get_saved_concepts(path_kg, data_name=data_name, shuffle=False) 

    if name_reasoner == 'EBR':
        cached_retriever = semantic_caching_size(retrieve, cache_size=cache_size, eviction_strategy=eviction, random_seed=random_seed, cache_type=cache_type, concepts=alc_concepts)
    else:
        cached_retriever = semantic_caching_size(retrieve_other_reasoner, cache_size=cache_size, eviction_strategy=eviction, random_seed=random_seed, cache_type=cache_type, concepts=alc_concepts)

    total_time_ebr = 0

    for expr in alc_concepts: 
        if name_reasoner == 'EBR':
            time_start_cache = time.time()
            A  = cached_retriever(expr, path_kg, path_kge) #Retrieval with cache
            time_cache = time.time()-time_start_cache

            time_start = time.time()
            retrieve_ebr = retrieve(expr, path_kg, path_kge) #Retrieval without cache
            time_ebr = time.time()-time_start
            total_time_ebr += time_ebr

        else:
            time_start_cache = time.time()
            A  = cached_retriever(expr, path_kg, name_reasoner)  #Retrieval with cache
            time_cache = time.time()-time_start_cache

            time_start = time.time()
            retrieve_ebr = retrieve_other_reasoner(expr, path_kg, name_reasoner=name_reasoner) #Retrieval without cache
            time_ebr = time.time()-time_start
            total_time_ebr += time_ebr

        ground_truth = concept_retrieval(symbolic_kb, expr)

        jacc = jaccard_similarity(A, ground_truth)
        jacc_reas = jaccard_similarity(retrieve_ebr, ground_truth)
        Avg_jaccard.append(jacc)
        Avg_jaccard_reas.append(jacc_reas)
        D.append({'dataset':data_name,'Expression':owl_expression_to_dl(expr), "Type": type(expr).__name__ ,'cache_size':cache_size, "time_ebr":time_ebr, "time_cache": time_cache, "Jaccard":jacc})
        print(f'Expression: {owl_expression_to_dl(expr)}')
        print(f'Jaccard similarity: {jacc}')
        # assert jacc == 1.0 

    stats = cached_retriever.get_stats()
    
    print('-'*50)
    print("Cache Statistics:")
    print(f"Hit Ratio: {stats['hit_ratio']:.2f}")
    print(f"Miss Ratio: {stats['miss_ratio']:.2f}")
    print(f"Average Time per Request: {stats['average_time_per_request']:.4f} seconds")
    print(f"Total Time with Caching: {stats['total_time']:.4f} seconds")
    print(f"Total Time Without Caching: {total_time_ebr:.4f} seconds")
    print(f"Total number of concepts: {len(alc_concepts)}")
    print(f"Average Jaccard for the {data_name} dataset", sum(Avg_jaccard)/len(Avg_jaccard))

    return {
        'dataset': data_name,
        'cache_size': cache_size,
        'hit_ratio': f"{stats['hit_ratio']:.2f}",
        'miss_ratio': f"{stats['miss_ratio']:.2f}",
        'RT_cache': f"{stats['total_time']:.3f}",
        'RT': f"{total_time_ebr:.3f}",
        '#concepts': len(alc_concepts),
        'avg_jaccard': f"{sum(Avg_jaccard) / len(Avg_jaccard):.3f}",
        'avg_jaccard_reas':  f"{sum(Avg_jaccard_reas) / len(Avg_jaccard_reas):.3f}",
        'strategy': eviction
    }, D



def run_non_semantic_cache(path_kg:str, path_kge:str, cache_size:int, name_reasoner:str, shuffle_concepts:bool):
    '''Return cache performnace without any semantics'''

    symbolic_kb = KnowledgeBase(path=path_kg)
    D = []
    Avg_jaccard = []
    Avg_jaccard_reas = []
    data_name = path_kg.split("/")[-1].split("/")[-1].split(".")[0]

    if shuffle_concepts:
        alc_concepts = get_saved_concepts(path_kg, data_name=data_name, shuffle=True) 
    else:
        alc_concepts = get_saved_concepts(path_kg, data_name=data_name, shuffle=False) 

    if name_reasoner == 'EBR':
        cached_retriever = non_semantic_caching_size(retrieve, cache_size=cache_size)
    else:
        cached_retriever = non_semantic_caching_size(retrieve_other_reasoner, cache_size=cache_size)

    total_time_ebr = 0

    for expr in alc_concepts: 
        if name_reasoner == 'EBR':
            time_start_cache = time.time()
            A  = cached_retriever(expr, path_kg, path_kge) #Retrieval with cache
            time_cache = time.time()-time_start_cache

            time_start = time.time()
            retrieve_ebr = retrieve(expr, path_kg, path_kge) #Retrieval without cache
            time_ebr = time.time()-time_start
            total_time_ebr += time_ebr

        else:
            time_start_cache = time.time()
            A  = cached_retriever(expr, path_kg, name_reasoner)  #Retrieval with cache
            time_cache = time.time()-time_start_cache

            time_start = time.time()
            retrieve_ebr = retrieve_other_reasoner(expr, path_kg, name_reasoner=name_reasoner) #Retrieval without cache
            time_ebr = time.time()-time_start
            total_time_ebr += time_ebr

        ground_truth = concept_retrieval(symbolic_kb, expr)

        jacc = jaccard_similarity(A, ground_truth)
        jacc_reas = jaccard_similarity(retrieve_ebr, ground_truth)
        Avg_jaccard.append(jacc)
        Avg_jaccard_reas.append(jacc_reas)
        D.append({'dataset':data_name,'Expression':owl_expression_to_dl(expr), "Type": type(expr).__name__ ,'cache_size':cache_size, "time_ebr":time_ebr, "time_cache": time_cache, "Jaccard":jacc})
        print(f'Expression: {owl_expression_to_dl(expr)}')
        print(f'Jaccard similarity: {jacc}')
        # assert jacc == 1.0 

    stats = cached_retriever.get_stats()
    
    print('-'*50)
    print("Cache Statistics:")
    print(f"Hit Ratio: {stats['hit_ratio']:.2f}")
    print(f"Miss Ratio: {stats['miss_ratio']:.2f}")
    print(f"Average Time per Request: {stats['average_time_per_request']:.4f} seconds")
    print(f"Total Time with Caching: {stats['total_time']:.4f} seconds")
    print(f"Total Time Without Caching: {total_time_ebr:.4f} seconds")
    print(f"Total number of concepts: {len(alc_concepts)}")
    print(f"Average Jaccard for the {data_name} dataset", sum(Avg_jaccard)/len(Avg_jaccard))

    return {
        'dataset': data_name,
        'cache_size': cache_size,
        'hit_ratio': f"{stats['hit_ratio']:.2f}",
        'miss_ratio': f"{stats['miss_ratio']:.2f}",
        'RT_cache': f"{stats['total_time']:.3f}",
        'RT': f"{total_time_ebr:.3f}",
        '#concepts': len(alc_concepts),
        'avg_jaccard': f"{sum(Avg_jaccard) / len(Avg_jaccard):.3f}",
        'avg_jaccard_reas':  f"{sum(Avg_jaccard_reas) / len(Avg_jaccard_reas):.3f}"
    }, D

