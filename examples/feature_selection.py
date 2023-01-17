import pandas as pd
import json
import os
from random import shuffle
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse as sp
from sklearn.feature_selection import chi2
from owlready2 import get_ontology, default_world

from examples.evolearner_feature_selection import EvoLearnerFeatureSelection
from examples.search import calc_prediction

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI


try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    print(os.getcwd())
    pass

with open('dataset/premier_league.json') as json_file:
    settings = json.load(json_file)


def get_data_properties(onto):
    try:
        data_properties = onto.data_properties()
    except Exception as e:
        data_properties = None
        print(e)
    return data_properties


def get_object_properties(onto):
    try:
        object_properties = onto.object_properties()
    except Exception as e:
        object_properties = None
        print(e)
    return object_properties


def calc_variance_threshold(pd3):
    pd3 = (pd3.notnull()).astype('int')
    sparse_matrix = sp.csr_matrix.tocsr(pd3.iloc[:, 1:].values)
    pd3.to_csv("dataset_converted_to_bool.csv")

    try:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        output_variance_threshold = sel.fit_transform(sparse_matrix)
        output_variance_threshold.to_csv(
                        'best_features_variance_threshold.csv')
    except ValueError:
        print("None of the Features meet the minimum variance threshold")
        sel = []
        output_variance_threshold = []

    print(output_variance_threshold)


def calc_chi2(pd3):
    '''
    Select K-Best features based on chi2 analysis
    '''
    X = pd3.iloc[:, 1:].values
    labels = pd3.iloc[:, 0].values

    try:
        select_k_best_classifier = SelectKBest(
                                chi2, k=5).fit(
                                X, labels.tolist())
        mask = select_k_best_classifier.get_support(indices=True)
        k_best_features = pd.DataFrame(index=pd3.index)
        k_best_features = pd3.iloc[:, [x+1 for x in mask.tolist()]]
        k_best_features.to_csv("kbest_features_chi2.csv")
        k_best_features_names = k_best_features.columns

    except Exception as e:
        print(e)
        k_best_features = []
        k_best_features_names = []

    return k_best_features_names


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


def iterate_properties(properties, pos, neg, prop_type):
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

        flag = 0
        for op in properties_value:
            if not prop_type:
                if (op.range[0] is int):
                    pass
                elif (op.range[0] is float):
                    pass
                elif(op.range[0] is bool):
                    flag = 1
                    pass
                else:
                    continue
            temp_prop_iri = op.iri
            properties_iri.append(temp_prop_iri)
            triples_list = list(default_world.sparql("""
            SELECT ?subject ?object
            WHERE {?subject <""" + str(temp_prop_iri) + """> ?object}
            """))

            triples_iri = []
            for items in triples_list:
                temp = []
                temp.append(items[0].iri)
                obj_val = ''
                if prop_type:
                    obj_val = items[1].iri
                    temp.append(obj_val)
                else:
                    if flag == 1:
                        obj_val = int(items[1])
                    else:
                        obj_val = items[1]
                    temp.append(obj_val)
                triples_iri.append(temp)

            for sub, obj in triples_iri:
                label = 2
                if sub in pos:
                    label = 1
                elif sub in neg:
                    label = 0

                if prop_type:
                    if sub not in dict_of_triples:
                        dict_of_triples[sub] = {'label': label, op.name: [obj]}
                    elif sub in dict_of_triples:
                        if op.name in dict_of_triples[sub].keys():
                            dict_of_triples[sub][op.name].append(obj)
                        else:
                            dict_of_triples[sub][op.name] = [obj]
                else:
                    if sub not in dict_of_triples:
                        dict_of_triples[sub] = {'label': label, op.name: obj}
                    elif sub in dict_of_triples:
                        dict_of_triples[sub][op.name] = obj

        pandas_dataframe = pd.DataFrame(dict_of_triples)
        df_transposed = pandas_dataframe.transpose()
        if prop_type:
            df_transposed.to_csv("Object_properties_to_tabular_form.csv")
        else:
            df_transposed.to_csv("Data_properties_to_tabular_form.csv")
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
        df_object_prop = iterate_properties(obj_prop, pos, neg, True)
        return df_object_prop


def transform_data_properties(onto, pos, neg):
    '''
    Read data properties and and call iterate
    properties to convert to tabular form
    '''
    with onto:
        data_prop = get_data_properties(onto)
        df_data_prop = iterate_properties(data_prop, pos, neg, False)
        if not df_data_prop.empty:
            df_data_prop = df_data_prop.replace(np.nan, 0, regex=True)
        return df_data_prop


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

        pd3 = transform_object_properties(onto, p, n)
        print('Finished Converting Object Properties')
        pd4 = transform_data_properties(onto, p, n)
        print('Finished Converting Data Properties')

        feature_names = []
        if not pd3.empty:
            pd_one_hot = one_hot_encoder(pd3)
            if not pd4.empty:
                features_object_properties = list(calc_chi2(pd_one_hot))
                features_data_properties = list(calc_chi2(pd4))
                feature_names = features_object_properties + features_data_properties
            elif pd4.empty:
                feature_names = calc_chi2(pd_one_hot)
            feature_names = list(set(map(lambda x: x.split("_feature_name_")[0], feature_names)))
        elif not pd4.empty:
            feature_names = calc_chi2(pd4)
            print(feature_names)

        model = EvoLearnerFeatureSelection(knowledge_base=kb,
                           feature_names=feature_names)

        model.fit(lp)
        model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
        hypotheses = list(model.best_hypotheses(n=1))
        predictions = model.predict(individuals=list(test_pos | test_neg),
                                    hypotheses=hypotheses)
        f1_score, accuracy = calc_prediction(predictions, test_pos, test_neg)
        quality = hypotheses[0].quality
        print("_____________QUALITY______________")
        print(quality)
        print("f1_score", f1_score)
        print("accuracy", accuracy)
        print("-----------Predictions------------")
        print(predictions)
        break
