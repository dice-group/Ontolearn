import datetime
import logging
import os
import time
import numpy as np


def compute_confidence_interval(results):
    for metric, values in results.items():
        margin_of_error = 1.96 * (values.std() / np.sqrt(len(values)))
        print(f'Confidence interval of {metric} => {values.mean()} +- {margin_of_error}')


def create_experiment_folder(folder_name='Experiments'):
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def get_experiments(path: str):
    """
    :param path: str represents path of a KB or path of folder containg KBs
    :return:
    """
    valid_exp = list()

    must_contain = {'info.log', 'model.pt', 'settings.json'}

    for root, dir, files in os.walk(path):
        files = set(files)
        if files.issuperset(must_contain):
            valid_exp.append(root)
    if len(valid_exp) == 0:
        print(
            '{0} is not a path for a file or a folder containing any .nq or .nt formatted files'.format(path))
        print('Execution is terminated.')
        exit(1)
    return valid_exp


def performance_debugger(func_name):
    def function_name_decoratir(func):
        def debug(*args, **kwargs):
            long_string = ''
            starT = time.time()
            print('\n\n######', func_name, ' starts ######')
            r = func(*args, **kwargs)
            print(func_name, ' took ', time.time() - starT, ' seconds\n')
            long_string += str(func_name) + ' took:' + str(time.time() - starT) + ' seconds'

            return r

        return debug

    return function_name_decoratir
