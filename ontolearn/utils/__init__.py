import datetime
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from ontolearn.utils.log_config import setup_logging  # noqa: F401

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
    assertion_path_isfile(path)
    df = pd.read_csv(path, index_col=0)
    assert (df.all()).all()  # all columns and all rows are not none.
    return df


def assertion_path_isfile(path):
    try:
        assert os.path.isfile(path)
    except AssertionError:
        print(f'{path} is not found.')
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
