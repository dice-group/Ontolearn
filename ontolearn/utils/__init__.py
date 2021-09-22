import datetime
import os
import pickle
import random
import time
from typing import Callable, Set, TypeVar, Generic, Optional

from ontolearn.utils.log_config import setup_logging  # noqa: F401
from owlapy.model import OWLNamedIndividual, IRI, OWLClass, HasIRI

Factory = Callable

# DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'
DEFAULT_FMT = 'Func:{name} took {elapsed:0.8f}s'
flag_for_performance = False


def parametrized_performance_debugger(fmt=DEFAULT_FMT):
    def decorate(func):
        if flag_for_performance:
            def clocked(*_args):
                t0 = time.time()
                _result = func(*_args)
                elapsed = time.time() - t0
                name = func.__name__
                args = ', '.join(repr(arg) for arg in _args)
                result = repr(_result)
                print(fmt.format(**locals()))
                return _result

            return clocked
        else:
            return func

    return decorate


def performance_debugger(func_name):
    def function_name_decorator(func):
        def debug(*args, **kwargs):
            start = time.time()
            r = func(*args, **kwargs)
            print(func_name, ' took ', round(time.time() - start, 4), ' seconds')

            return r

        return debug

    return function_name_decorator


def create_experiment_folder(folder_name='Log'):
    from ontolearn.utils import log_config
    if log_config.log_dirs:
        path_of_folder = log_config.log_dirs[-1]
    else:
        directory = os.getcwd() + '/' + folder_name + '/'
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path_of_folder = directory + folder_name
        os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def serializer(*, object_: object, path: str, serialized_name: str):
    with open(path + '/' + serialized_name + ".p", "wb") as f:
        pickle.dump(object_, f)
    f.close()


def deserializer(*, path: str, serialized_name: str):
    with open(path + "/" + serialized_name + ".p", "rb") as f:
        obj_ = pickle.load(f)
    f.close()
    return obj_


def apply_TSNE_on_df(df) -> None:
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    low_emb = TSNE(n_components=2).fit_transform(df)
    plt.scatter(low_emb[:, 0], low_emb[:, 1])
    plt.title('Instance Representatons via TSNE')
    plt.show()


def balanced_sets(a: set, b: set) -> (set, set):
    """
    Balance given two sets through sampling without replacement.
    Returned sets have the same length.
    @param a:
    @param b:
    @return:
    """

    if len(a) > len(b):
        sampled_a = random.sample(a, len(b))
        return set(sampled_a), b
    elif len(b) > len(a):
        sampled_b = random.sample(b, len(a))
        return a, set(sampled_b)
    else:
        assert len(a) == len(b)
        return a, b


def read_csv(path):
    """
    Path leads a folder containing embeddings in csv format.
    indexes correspond subjects or predicates or objects in n-triple.
    @param path:
    @return:
    """
    import pandas as pd
    assertion_path_isfile(path)
    df = pd.read_csv(path, index_col=0)
    assert (df.all()).all()  # all columns and all rows are not none.
    return df


def assertion_path_isfile(path) -> None:
    try:
        assert path is not None
    except AssertionError:
        print(f'Path can not be:{path}')
        raise

    try:
        assert os.path.isfile(path)
    except (AssertionError, TypeError):
        print(f'Input:{path} not found.')
        raise


def sanity_checking_args(args):
    try:
        assert os.path.isfile(args.path_knowledge_base)
    except AssertionError:
        print(f'--path_knowledge_base ***{args.path_knowledge_base}*** does not lead to a file.')
        raise

    try:
        assert os.path.isfile(args.path_knowledge_base_embeddings)
    except AssertionError:
        print(f'--path_knowledge_base_embeddings ***{args.path_knowledge_base_embeddings}*** does not lead to a file.')
        raise

    assert args.min_length > 0
    assert args.max_length > 0
    assert args.min_num_concepts > 0
    assert args.min_num_concepts > 0
    assert args.min_num_instances_per_concept > 0
    assert os.path.isfile(args.path_knowledge_base)
    if hasattr(args, 'num_fold_for_k_fold_cv'):
        assert args.num_fold_for_k_fold_cv > 0
    if hasattr(args, 'max_test_time_per_concept'):
        assert args.max_test_time_per_concept > 1

    if hasattr(args, 'num_of_sequential_actions'):
        assert args.num_of_sequential_actions > 0

    if hasattr(args, 'batch_size'):
        assert args.batch_size > 1


_T = TypeVar('_T', bound=HasIRI)


def _read_iri_file(file: str, type_: Factory[[IRI], _T]) -> Set[_T]:
    """Read a text file containing IRIs (one per line) and return the content as a set of instances created by the
    given type

    Args:
        file: path to the text file with the IRIs of the named individuals
        type_: factory or type to create from the IRI

    Returns:
        set of type_ instances with these IRIs
    """

    def optional_angles(iri: str):
        if iri.startswith('<'):
            return iri[1:-1]
        else:
            return iri

    with open(file, 'r') as f:
        inds = map(type_,
                   map(IRI.create,
                       map(optional_angles,
                           f.read().splitlines())))
    return set(inds)


def read_individuals_file(file: str) -> Set[OWLNamedIndividual]:
    """Read a text file containing IRIs of Named Individuals (one per line) and return the content as a set of OWL
    Named Individuals

    Args:
        file: path to the text file with the IRIs of the named individuals

    Returns:
        set of OWLNamedIndividual with these IRIs
    """
    return _read_iri_file(file, OWLNamedIndividual)


def read_named_classes_file(file: str) -> Set[OWLClass]:
    """Read a text file containing IRIs of OWL Named Classes (one per line) and return the content as a set of OWL
    Classes

    Args:
        file: path to the text file with the IRIs of the classes

    Returns:
        set of OWLNamedIndividual with these IRIs
    """
    return _read_iri_file(file, OWLClass)


_K = TypeVar('_K')
_V = TypeVar('_V')


# adapted from functools.lru_cache
class LRUCache(Generic[_K, _V]):
    # Constants shared by all lru cache instances:
    sentinel = object()  # unique object used to signal cache misses
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

    def __init__(self, maxsize: Optional[int] = None):
        from _thread import RLock

        self.cache = {}
        self.hits = self.misses = 0
        self.full = False
        self.cache_get = self.cache.get  # bound method to lookup a key or return None
        self.cache_len = self.cache.__len__  # get cache size without calling len()
        self.lock = RLock()  # because linkedlist updates aren't threadsafe
        self.root = []  # root of the circular doubly linked list
        self.root[:] = [self.root, self.root, None, None]  # initialize by pointing to self
        self.maxsize = maxsize
        
    def __contains__(self, item: _K) -> bool:
        with self.lock:
            link = self.cache_get(item)
            if link is not None:
                self.hits += 1
                return True
            self.misses += 1
            return False

    def __getitem__(self, item: _K) -> _V:
        with self.lock:
            link = self.cache_get(item)
            if link is not None:
                # Move the link to the front of the circular queue
                link_prev, link_next, _key, result = link
                link_prev[LRUCache.NEXT] = link_next
                link_next[LRUCache.PREV] = link_prev
                last = self.root[LRUCache.PREV]
                last[LRUCache.NEXT] = self.root[LRUCache.PREV] = link
                link[LRUCache.PREV] = last
                link[LRUCache.NEXT] = self.root
                return result

    def __setitem__(self, key: _K, value: _V):
        with self.lock:
            if key in self.cache:
                # Getting here means that this same key was added to the
                # cache while the lock was released.  Since the link
                # update is already done, we need only return the
                # computed result and update the count of misses.
                pass
            elif self.full:
                # Use the old root to store the new key and result.
                oldroot = self.root
                oldroot[LRUCache.KEY] = key
                oldroot[LRUCache.RESULT] = value
                # Empty the oldest link and make it the new root.
                # Keep a reference to the old key and old result to
                # prevent their ref counts from going to zero during the
                # update. That will prevent potentially arbitrary object
                # clean-up code (i.e. __del__) from running while we're
                # still adjusting the links.
                self.root = oldroot[LRUCache.NEXT]
                oldkey = self.root[LRUCache.KEY]
                _oldresult = self.root[LRUCache.RESULT]
                self.root[LRUCache.KEY] = self.root[LRUCache.RESULT] = None
                # Now update the cache dictionary.
                del self.cache[oldkey]
                # Save the potentially reentrant cache[key] assignment
                # for last, after the root and links have been put in
                # a consistent state.
                self.cache[key] = oldroot
            else:
                # Put result in a new link at the front of the queue.
                last = self.root[LRUCache.PREV]
                link = [last, self.root, key, value]
                last[LRUCache.NEXT] = self.root[LRUCache.PREV] = self.cache[key] = link
                # Use the cache_len bound method instead of the len() function
                # which could potentially be wrapped in an lru_cache itself.
                if self.maxsize is not None:
                    self.full = (self.cache_len() >= self.maxsize)

    def cache_info(self):
        """Report cache statistics"""
        with self.lock:
            from collections import namedtuple
            return namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])(
                self.hits, self.misses, self.maxsize, self.cache_len())

    def cache_clear(self):
        """Clear the cache and cache statistics"""
        with self.lock:
            self.cache.clear()
            self.root[:] = [self.root, self.root, None, None]
            self.hits = self.misses = 0
            self.full = False
