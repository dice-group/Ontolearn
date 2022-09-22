import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from ontolearn.utils import setup_logging


df = pd.DataFrame(columns=['LP', 'max_runtime', 'tournament_size', 'height_limit','use_data_properties','value_splitter', 'F1_train', 'Accuracy_test','F1_test','length'])

def grid_search(target_kb, str_target_concept, space_grid, lp):
    best_quality = None
    best_parameter = None

    for parameter_grid in space_grid:        
        model = EvoLearner(knowledge_base=target_kb, **parameter_grid)
        model.fit(lp, verbose=False)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))      
        hypotheses = list(model.best_hypotheses(n=1))      
        quality = hypotheses[0].quality
        
        if best_quality is None:
            best_quality = quality
            best_parameter = parameter_grid
        elif best_quality <= quality:
            best_quality = quality
            best_parameter = parameter_grid

    return best_parameter

def calc_prediction(predictions,test_pos,test_neg):    
    concepts_sorted = sorted(predictions)   
    concepts_dict = {}    
    for con in concepts_sorted:        
        positive_indivuals = predictions[predictions[con].values > 0.0].index.values
        negative_indivuals = predictions[predictions[con].values <= 0.0].index.values
        concepts_dict[con] = {"Pos":positive_indivuals,"Neg":negative_indivuals}        

    for key in concepts_dict:        
        tp = len(list(set(get_short_names(list(test_pos))).intersection(set(concepts_dict[key]["Pos"]))))
        tn = len(list(set(get_short_names(list(test_neg))).intersection(set(concepts_dict[key]["Neg"]))))
        fp = len(list(set(get_short_names(list(test_neg))).intersection(set(concepts_dict[key]["Pos"]))))
        fn = len(list(set(get_short_names(list(test_pos))).intersection(set(concepts_dict[key]["Neg"]))))
        f1 = F1()
        accuracy = Accuracy()
        f1_score = list(f1.score2(tp=tp,fn=fn,fp=fp,tn=tn))
        accuracy_score = list(accuracy.score2(tp=tp,fn=fn,fp=fp,tn=tn))
        concept_and_score = [key,f1_score]        
        return f1_score, accuracy_score

def custom_split(x_pos,x_neg, n_splits):
    #Return validation set and train set indices
    indices_x_pos = np.arange(len(x_pos))
    indices_x_neg = np.arange(len(x_neg))
    for val_index_x_pos, val_index_x_neg in mask_value(x_pos, x_neg, n_splits):       
        train_index_x_pos = indices_x_pos[np.logical_not(val_index_x_pos)]
        test_index_x_pos = indices_x_pos[val_index_x_pos]
        train_index_x_neg = indices_x_neg[np.logical_not(val_index_x_neg)]
        test_index_x_neg = indices_x_neg[val_index_x_neg]
        yield train_index_x_pos,test_index_x_pos, train_index_x_neg,test_index_x_neg

def mask_value(X_pos,X_neg,n_splits):
    #Mask the Indices    
    for test_index_x_pos, text_index_x_neg in generate_folds(X_pos,X_neg, n_splits):        
        test_mask_x_pos = np.zeros(len(X_pos), dtype=bool)
        test_mask_x_pos[test_index_x_pos] = True
        text_mask_x_neg = np.zeros(len(X_neg), dtype=bool)
        text_mask_x_neg[text_index_x_neg] = True        
        yield test_mask_x_pos, text_mask_x_neg

def generate_folds(sample_x_pos,sample_x_neg, n_splits):    
    num_of_samples_x_pos = len(sample_x_pos)
    num_of_samples_x_neg = len(sample_x_neg)

    if num_of_samples_x_neg < n_splits or  num_of_samples_x_neg < n_splits:
        print("Choose a smaller fold size")
    else:
        indices_x_pos = np.arange(num_of_samples_x_pos)
        indices_x_neg = np.arange(num_of_samples_x_neg)
        fold_sizes_x_pos = np.full(n_splits, num_of_samples_x_pos // n_splits, dtype=int)
        fold_sizes_x_pos[: num_of_samples_x_pos % n_splits] += 1
        fold_sizes_x_neg = np.full(n_splits, num_of_samples_x_neg // n_splits, dtype=int)
        fold_sizes_x_neg[: num_of_samples_x_neg % n_splits] += 1        
        current_pos = 0
        current_neg = 0
        for index in range (0,len(fold_sizes_x_pos)):
            start_x_pos, stop_x_pos = current_pos, current_pos + fold_sizes_x_pos[index]
            start_x_neg, stop_x_neg = current_neg, current_neg + fold_sizes_x_neg[index]            
            yield indices_x_pos[start_x_pos:stop_x_pos], indices_x_neg[start_x_neg:stop_x_neg] 
            current_pos = stop_x_pos
            current_neg = stop_x_neg

def grid_search_with_custom_cv(target_kb,str_target_concept, train_pos, train_neg, space_grid, n_splits=2):
    # Takes Train Pos and Train Neg
    # Fit each iteration 
    # To Do: Calculate and Compare the Score
    # To Do: return the mean score
    X_pos = np.array(train_pos)
    X_neg = np.array(train_neg)
    best_quality = None
    best_parameter = None

    for parameter_grid in space_grid: 
        for train_index_x_pos, test_index_x_pos, train_index_x_neg, test_index_x_neg in  custom_split(X_pos, X_neg, n_splits):
            model = EvoLearner(knowledge_base=target_kb, quality_func=F1(), **parameter_grid)
            X_pos_train, X_pos_test = set(X_pos[train_index_x_pos].tolist()), set(X_pos[test_index_x_pos].tolist())
            X_neg_train, X_neg_test = set(X_neg[train_index_x_neg].tolist()), set(X_neg[test_index_x_neg].tolist())       

            lp = PosNegLPStandard(pos=X_pos_train, neg=X_neg_train)           
            model.fit(lp, verbose=False)
            model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))      
            hypotheses = list(model.best_hypotheses(n=1))
            quality = hypotheses[0].quality          
            predictions = model.predict(individuals=list(X_pos_test | X_neg_test),
                                    hypotheses=hypotheses)
            f1_measure, accuracy = calc_prediction(predictions,X_pos_test,X_neg_test)            
            df.loc[len(df.index)] = [str_target_concept,parameter_grid['max_runtime'],parameter_grid['tournament_size'],parameter_grid['height_limit'],parameter_grid['use_data_properties'],parameter_grid['value_splitter'],quality,accuracy[1],f1_measure[1],hypotheses[0]._len]

       
def get_short_names(individuals):
    short_names = []
    for individual in individuals :
        sn = individual.get_iri().get_short_form()
        short_names.append(sn)

    return short_names      
