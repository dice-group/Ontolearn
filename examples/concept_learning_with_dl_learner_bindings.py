import json

from ontolearn.binders import DLLearnerBinder

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

path_of_background = '/home/demir/Desktop/Onto-learn_dev/data/family-benchmark_rich_background.owl'
# To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
path_dl_learner = '/home/demir/Desktop/DL/dllearner-1.4.0/'
knowledge_base_path = path_of_background

dl_leaner = DLLearnerBinder(path=path_dl_learner)
for str_target_concept, examples in settings['problems'].items():
    print('\nTARGET CONCEPT:', str_target_concept)
    positives = examples['positive_examples']
    negatives = examples['negative_examples']

    # Create Config file
    # Run Config file
    algorithm = 'celoe'
    str_best_concept, f_1score = dl_leaner.pipeline(
        knowledge_base_path=knowledge_base_path,
        algorithm=algorithm,
        positives=positives,
        negatives=negatives,
        path_name=str_target_concept,)

    print('JAVA:{0}:BestPrediction:\t{1}\tF-score:{2}'.format(
        algorithm, str_best_concept, f_1score))
