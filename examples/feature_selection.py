import pandas as pd
import json
import os
import time
from random import shuffle
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectFpr
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse as sp
from sklearn.feature_selection import chi2, f_classif,  mutual_info_classif
from owlready2 import get_ontology, default_world
import logging

from examples.evolearner_feature_selection import EvoLearnerFeatureSelection
from examples.search import calc_prediction
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI

LOG_FILE = 'featureSelection.log'
DATASET = 'hepatitis'

logging.basicConfig(filename=LOG_FILE,
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    logging.error(FileNotFoundError)
    pass

with open(f'dataset/{DATASET}.json') as json_file:
    settings = json.load(json_file)


def get_data_properties(onto):
    try:
        data_properties = onto.data_properties()
    except Exception as e:
        data_properties = None
        logging.error(e)
    return data_properties


def get_object_properties(onto):
    try:
        object_properties = onto.object_properties()
    except Exception as e:
        object_properties = None
        logging.error(e)
    return object_properties


def calc_variance_threshold(pd3):
    pd3 = (pd3.notnull()).astype('int')
    sparse_matrix = sp.csr_matrix.tocsr(pd3.iloc[:, 1:].values)
    pd3.to_csv(f"{DATASET}_dataset_converted_to_bool.csv")

    try:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        output_variance_threshold = sel.fit_transform(sparse_matrix)
        output_variance_threshold.to_csv(f"{DATASET}_best_features_variance_threshold.csv")
    except ValueError:
        logging.error("None of the Features meet the minimum variance threshold")


def select_k_best_features(pd3, method):
    '''
    Select K-Best features based on chi2/mutual_info_classif analysis
    '''
    X = pd3.iloc[:, 1:].values
    labels = pd3.iloc[:, 0].values

    try:
        select_k_best_classifier = SelectKBest(
                                method, k=5).fit(
                                X, labels.tolist())
        mask = select_k_best_classifier.get_support(indices=True)
        k_best_features = pd.DataFrame(index=pd3.index)
        k_best_features = pd3.iloc[:, [x+1 for x in mask.tolist()]]
        k_best_features.to_csv(f"{DATASET}_kbest_features_chi2.csv")
        k_best_features_names = k_best_features.columns

    except Exception as e:
        logging.error(e)
        k_best_features_names = []
    return k_best_features_names


def select_fpr_features(pd3, method):
    '''
    Select features based on chi2/f_classif analysis
    False Positive Rate test checks the total amt. of False detection
    '''
    X = pd3.iloc[:, 1:].values
    labels = pd3.iloc[:, 0].values

    try:
        select_k_best_classifier = SelectFpr(
                                method, alpha=0.01).fit(
                                X, labels.tolist())
        mask = select_k_best_classifier.get_support(indices=True)
        fpr_features = pd.DataFrame(index=pd3.index)
        fpr_features = pd3.iloc[:, [x+1 for x in mask.tolist()]]
        fpr_features.to_csv(f"{DATASET}_fpr_features_chi2.csv")
        fpr_features_names = fpr_features.columns

    except Exception as e:
        logging.error(e)
        fpr_features_names = []
    return fpr_features_names


def one_hot_encoder(pd3):
    mlb = MultiLabelBinarizer(sparse_output=True)
    column_names = pd3.columns
    pd3 = pd3.replace(np.nan, '', regex=True)

    for cn in column_names[1:]:
        mlb.fit(pd3[cn])
        new_col_names = [cn+"_feature_name_%s" % c for c in mlb.classes_]
        pd3 = pd3.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(pd3.pop(cn)),
                index=pd3.index,
                columns=new_col_names)
            )
    return pd3


def iterate_object_properties(properties, pos, neg):
    '''
    Iterate over properties and convert it to tabular data
    '''
    properties_iri = []
    dict_of_triples = {}
    column_headers = ['subject']
    if properties is not None:
        properties_value = list(properties)
        for prop in properties_value:
            column_headers.append(prop.name)
        column_headers.append('label')

        for op in properties_value:
            temp_prop_iri = op.iri
            properties_iri.append(temp_prop_iri)
            triples_list = list(default_world.sparql("""
            SELECT ?subject ?object
            WHERE {?subject <""" + str(temp_prop_iri) + """> ?object}
            """))

            for items in triples_list:
                sub, obj = items[0].iri, items[1].iri
                if sub in pos:
                    label = 1
                elif sub in neg:
                    label = 0
                else:
                    label = 2

                if sub not in dict_of_triples:
                    dict_of_triples[sub] = {'label': label, op.name: [obj]}
                elif sub in dict_of_triples:
                    if op.name in dict_of_triples[sub].keys():
                        dict_of_triples[sub][op.name].append(obj)
                    else:
                        dict_of_triples[sub][op.name] = [obj]

        pandas_dataframe = pd.DataFrame(dict_of_triples)
        df_transposed = pandas_dataframe.transpose()
        df_transposed.to_csv(f"{DATASET}_Object_properties_to_tabular_form.csv")
    else:
        df_transposed = pd.DataFrame()
    return df_transposed


def transform_object_properties(obj_prop, pos, neg):
    '''
    Read Object properties and call
    iterate properties to convert to tabular form
    '''
    with onto:
        obj_prop = get_object_properties(onto)
        df_object_prop = iterate_object_properties(obj_prop, pos, neg)
        return df_object_prop


def transform_data_properties(onto, pos, neg):
    '''
    Read data properties and and call iterate
    properties to convert to tabular form
    '''
    with onto:
        data_prop = get_data_properties(onto)

        properties_iri = []
        dict_of_triples_num = {}
        dict_of_triples_bool = {}
        column_headers = ['subject']

        if data_prop is not None:
            properties_value = list(data_prop)
            for prop in properties_value:
                column_headers.append(prop.name)
            column_headers.append('label')

            for op in properties_value:
                if (op.range[0] is int):
                    flag = 0
                elif (op.range[0] is float):
                    flag = 0
                elif(op.range[0] is bool):
                    flag = 1
                else:
                    continue

                temp_prop_iri = op.iri
                properties_iri.append(temp_prop_iri)
                triples_list = list(default_world.sparql("""
                SELECT ?subject ?object
                WHERE {?subject <""" + str(temp_prop_iri) + """> ?object}
                """))

                for items in triples_list:
                    sub = items[0].iri
                    if flag:
                        obj = int(items[1])
                    else:
                        obj = items[1]

                    if sub in pos:
                        label = 1
                    elif sub in neg:
                        label = 0
                    else:
                        label = 2

                    if flag:
                        data_structure = dict_of_triples_bool
                    else:
                        data_structure = dict_of_triples_num

                    if sub not in data_structure:
                        data_structure[sub] = {'label': label, op.name: obj}
                    elif sub in data_structure:
                        data_structure[sub][op.name] = obj

            pandas_dataframe_numeric_data_types = pd.DataFrame(dict_of_triples_num)
            pandas_dataframe_categorical_data_type = pd.DataFrame(dict_of_triples_bool)
            df_transposed_numeric_dtype = pandas_dataframe_numeric_data_types.transpose()
            df_transposed_categorical_dtype = pandas_dataframe_categorical_data_type.transpose()

            df_transposed_numeric_dtype.to_csv(f"{DATASET}_Numeric_data_properties_to_tabular_form.csv")
            df_transposed_categorical_dtype.to_csv(f"{DATASET}_Boolean_data_properties_to_tabular_form.csv")

        if not df_transposed_numeric_dtype.empty:
            df_transposed_numeric_dtype = df_transposed_numeric_dtype.replace(np.nan, 0, regex=True)
        if not df_transposed_categorical_dtype.empty:
            df_transposed_categorical_dtype = df_transposed_categorical_dtype.replace(np.nan, 0, regex=True)

        return df_transposed_categorical_dtype, df_transposed_numeric_dtype


if __name__ == "__main__":
    onto = get_ontology(settings['data_path']).load()
    kb = KnowledgeBase(path=settings['data_path'])
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        typed_pos = list(set(map(OWLNamedIndividual, map(IRI.create, p))))
        typed_neg = list(set(map(OWLNamedIndividual, map(IRI.create, n))))

        # shuffle the Positive and Negative Sample
        shuffle(typed_pos)
        shuffle(typed_neg)

        # Split the data into Training Set and Test Set
        train_pos = set(typed_pos[:int(len(typed_pos)*0.8)])
        train_neg = set(typed_neg[:int(len(typed_neg)*0.8)])
        test_pos = set(typed_pos[-int(len(typed_pos)*0.2):])
        test_neg = set(typed_neg[-int(len(typed_neg)*0.2):])
        lp = PosNegLPStandard(pos=train_pos, neg=train_neg)

        object_properties_df = transform_object_properties(onto, p, n)
        bool_dtype_df, numeric_dtype_df = transform_data_properties(onto, p, n)
        feature_names = []
        features_data_properties_categoical = []
        feature_data_prop_numeric = []
        features_object_properties = []

        if not object_properties_df.empty:
            pd_one_hot = one_hot_encoder(object_properties_df)
            features_object_properties = list(select_k_best_features(pd_one_hot, chi2))
            features_object_properties = list(set(map(lambda x: x.split("_feature_name_")[0], features_object_properties)))

        if not bool_dtype_df.empty:
            features_data_properties_categoical = list(select_k_best_features(bool_dtype_df, chi2))

        if not numeric_dtype_df.empty:
            feature_data_prop_numeric = list(select_k_best_features(numeric_dtype_df, mutual_info_classif))

        logging.info(f"OBJECT_PROP:{features_object_properties}")
        logging.info(f"DATA_PROP_Numeric:{feature_data_prop_numeric}")
        logging.info(f"DATA_PROP_BOOL:{feature_data_prop_numeric}")

        st = time.time()
        model = EvoLearnerFeatureSelection(knowledge_base=kb,
                                           feature_obj_prop_name=features_object_properties,
                                           feature_data_categorical_prop=features_data_properties_categoical,
                                           feature_data_numeric_prop=feature_data_prop_numeric)

        model.fit(lp)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(test_pos | test_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality
        et = time.time()
        elapsed_time = et - st
        with open(f'{DATASET}_featureSelection.txt', 'a') as f:
            print('F1 Score', f1_score[1], file=f)
            print('Accuracy', accuracy[1], file=f)
            print('Time Taken', elapsed_time, file=f)
