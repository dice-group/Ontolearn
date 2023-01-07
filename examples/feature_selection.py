import pandas as pd
import json
import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse as sp
from sklearn.feature_selection import chi2

from owlready2 import get_ontology, default_world

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    print(os.getcwd())
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)


def get_data_properties(onto):
    data_properties = onto.data_properties()
    return data_properties


def get_object_properties(onto):
    object_properties = onto.object_properties()
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
    labels[labels == ''] = 2

    try:
        select_k_best_classifier = SelectKBest(
                                chi2, k=25).fit(
                                X, labels.tolist())
        mask = select_k_best_classifier.get_support(indices=True)
        k_best_features = pd.DataFrame(index=pd3.index)
        k_best_features = pd3.iloc[:, [x+1 for x in mask.tolist()]]
        k_best_features.to_csv("kbest_features_chi2.csv")
    except Exception as e:
        print(e)
        k_best_features = []

    print(k_best_features)


def one_hot_encoder(pd3):
    mlb = MultiLabelBinarizer(sparse_output=True)
    column_names = pd3.columns
    pd3 = pd3.replace(np.nan, '', regex=True)

    for cn in column_names[1:]:
        mlb.fit(pd3[cn])
        new_col_names = [cn+"_%s" % c for c in mlb.classes_]
        pd3 = pd3.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(pd3.pop(cn)),
                index=pd3.index,
                columns=new_col_names)
                )

    pd3.to_csv("output_of_one_hot_encoding.csv")
    return pd3


def transform_data(onto, pos, neg):
    '''
    Read ontology and convert it to tabular data
    '''
    with onto:
        try:
            obj_properties = list(get_object_properties(onto))
        except Exception as e:
            print(e)

        obj_properties_iri = []
        dict_of_triples = {}
        column_headers = ['subject']
        for i in obj_properties:
            column_headers.append(i.name)
        column_headers.append('label')

        for op in obj_properties:
            temp_obj_prop_iri = op.iri
            obj_properties_iri.append(temp_obj_prop_iri)
            triples_list = list(default_world.sparql("""
            SELECT ?subject ?object
            WHERE {?subject <""" + str(temp_obj_prop_iri) + """> ?object}
            """))

            triples_iri = []
            for items in triples_list:
                temp = []
                temp.append(items[0].iri)
                temp.append(items[1].iri)
                triples_iri.append(temp)

            for sub, obj in triples_iri:
                label = np.nan
                if sub in pos:
                    label = 1
                elif sub in neg:
                    label = 0

                if sub not in dict_of_triples:
                    dict_of_triples[sub] = {'label': label, op.name: [obj]}
                elif sub in dict_of_triples:
                    if op.name in dict_of_triples[sub].keys():
                        dict_of_triples[sub]['label'] = label
                        dict_of_triples[sub][op.name].append(obj)
                    else:
                        dict_of_triples[sub][op.name] = [obj]

        pandas_dataframe = pd.DataFrame(dict_of_triples)
        pd2 = pandas_dataframe.transpose()
        pd2.to_csv("Graphs_to_Tables.csv")
        # calc_variance_threshold(pd2)
        return pd2


if __name__ == "__main__":
    onto = get_ontology(settings['data_path']).load()
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])

        pd3 = transform_data(onto, p, n)
        pd_one_hot = one_hot_encoder(pd3)
        # calc_variance_threshold(pd_one_hot)
        calc_chi2(pd_one_hot)
        break
