import pandas as pd
import json
import os

from owlready2 import get_ontology, default_world

try:
    os.chdir("Ontolearn/examples")
except FileNotFoundError:
    print(os.getcwd())
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)


def organise_data(onto, pos, neg):
    with onto:
        obj_properties = list(onto.object_properties())
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
                label = 0
                if sub in pos:
                    label = 1

                if sub not in dict_of_triples:
                    dict_of_triples[sub] = {'label': label, op.name: [obj]}
                elif sub in dict_of_triples:
                    if op.name in dict_of_triples[sub].keys():
                        dict_of_triples[sub]['label'] = label
                        dict_of_triples[sub][op.name].append(obj)
                    else:
                        dict_of_triples[sub]['label'] = label
                        dict_of_triples[sub][op.name] = [obj]

        pandas_dataframe = pd.DataFrame(dict_of_triples)
        pd2 = pandas_dataframe.transpose()
        pd2.to_csv("Graphs_to_Tables.csv")


if __name__ == "__main__":
    onto = get_ontology(settings['data_path']).load()
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        organise_data(onto, p, n)
