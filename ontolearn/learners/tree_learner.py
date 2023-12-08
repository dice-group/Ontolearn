import pandas as pd

from ontolearn.knowledge_base import KnowledgeBase
from typing import Dict, Set, Tuple, List
from ontolearn.learning_problem import PosNegLPStandard
import collections
import matplotlib.pyplot as plt
from sklearn import tree

from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataHasValue, OWLObjectHasValue


def extract_cbd(dataframe) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Extract concise bounded description for each entity, where the entity is a subject entity.
    Create a mapping from a node to out-going edges and connected nodes
    :param dataframe:
    :return:
    """
    # Extract concise bounded description for each entity, where the entity is a subject entity.
    data = dict()
    for i in dataframe.values.tolist():
        subject_, predicate_, object_ = i
        data.setdefault(subject_, set()).add((predicate_, object_))
    return data


def base_construct_second(cbd_entities: Dict[str, Set[Tuple[str, str]]], rows: List[str],
                          feature_names: List[Tuple[str, str]]):
    """
    :param cbd_entities: concise bounded description for each entity, where the entity is a subject entity that is
    mapped to a predict and an object entity
    :param rows: Individuals
    """
    assert cbd_entities is not None, "No cbd entities"
    result = []
    for s in rows:
        # (1) Initialize an empty row
        row = [False for _ in feature_names]
        for (p, o) in cbd_entities[s]:
            idx = feature_names.index((p, o))
            # (2) Fill th row with nodes/object entities
            assert row[idx] is False
            row[idx] = True
        result.append(row)
    result = pd.DataFrame(data=result, index=rows, columns=feature_names, dtype="category")
    # print(f"Constructed tabular shape: {result.shape}")
    # print("Features/Columns:", result.columns.tolist())
    return result


def explain_inference(clf, X_test, features, only_shared):
    reports = []
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    # Positives
    node_indicator = clf.decision_path(X_test)
    leaf_id = clf.apply(X_test)

    if only_shared:
        sample_ids = range(len(X_test))
        # boolean array indicating the nodes both samples go through
        common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
        # obtain node ids using position in array
        common_node_id = np.arange(n_nodes)[common_nodes]

        print(
            "The following samples {samples} share the node(s) {nodes} in the tree.".format(
                samples=sample_ids, nodes=common_node_id
            )
        )
        print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
        return None

    for sample_id in range(len(X_test)):
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ]
        # print("Rules used to predict sample {id}:\n".format(id=sample_id))
        decision_path = []
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            # report = f"decision node {node_id} : ({features[feature[node_id]]} = {X_test[sample_id, feature[node_id]]}) {threshold_sign} {threshold[node_id]})"
            decision_path.append({"decision_node": node_id, "feature": features[feature[node_id]],
                                  "value": X_test[sample_id, feature[node_id]]})
        reports.append(decision_path)
    return reports


class TreeLearner:
    def __init__(self, knowledge_base, dataframe_triples: pd.DataFrame, quality_func):
        assert isinstance(dataframe_triples, pd.DataFrame), "dataframe_triples must be a Pandas DataFrame"
        assert isinstance(knowledge_base, KnowledgeBase), "knowledge_base must be a KnowledgeBase instance"
        assert len(
            dataframe_triples) > 0, f"length of the dataframe must be greater than 0:Currently {dataframe_triples.shape}"
        self.quality_func = quality_func
        self.knowledge_base = knowledge_base
        self.owl_classes_dict = {c.get_iri().as_str(): c for c in self.knowledge_base.get_concepts()}
        self.owl_object_property_dict = {p.get_iri().as_str(): p for p in self.knowledge_base.get_object_properties()}
        self.owl_individuals = {i.get_iri().as_str(): i for i in self.knowledge_base.individuals()}

        self.dataframe_triples = dataframe_triples
        # Remove some triples triples
        self.dataframe_triples = self.dataframe_triples[
            ~((self.dataframe_triples["relation"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type") & (
                    (self.dataframe_triples["object"] == "http://www.w3.org/2002/07/owl#NamedIndividual") | (
                    self.dataframe_triples["object"] == "http://www.w3.org/2002/07/owl#Thing") | (
                            self.dataframe_triples["object"] == "Ontology")))]

        self.cbd_mapping: Dict[str, Set[Tuple[str, str]]]
        self.cbd_mapping = extract_cbd(self.dataframe_triples)

        self.str_individuals = list({i.get_iri().as_str() for i in self.knowledge_base.individuals()})

        self.cbd_mapping_entities = {k: v for k, v in self.cbd_mapping.items() if k in self.str_individuals}

        self.Xraw = base_construct_second(cbd_entities=self.cbd_mapping_entities,
                                          rows=self.str_individuals,
                                          feature_names=[("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                                                          c.get_iri().as_str()) for c in
                                                         self.knowledge_base.get_concepts()]
                                                        + [(r.get_iri().as_str(), i) for i in self.str_individuals for r
                                                           in self.knowledge_base.get_object_properties()])
        assert len(self.Xraw) == len(self.str_individuals), "Xraw must be equal to individuals"
        self.clf = None

    def labeling(self, pos, neg, apply_dummy=True):
        """

        """
        # (5) Labeling: Label each row/node
        # Drop "label" if exists

        self.Xraw.loc[:, "label"] = 0  # unknowns
        self.Xraw.loc[pos, "label"] = 1  # positives
        self.Xraw.loc[neg, "label"] = -1  # negatives
        # (5.1) drop unknowns although unknowns provide info
        X = self.Xraw  # self.Xraw[self.Xraw.label != 0]

        raw_features = X.columns.tolist()
        raw_features.remove("label")
        if apply_dummy:
            X_train_sparse = pd.get_dummies(X[raw_features])
        else:
            X_train_sparse = X[raw_features]
        y_train_sparse = X.loc[:, "label"]

        print(f"Train data shape:{X_train_sparse.shape}")
        return X_train_sparse, y_train_sparse

    def compute_quality(self, concept, pos, neg):
        instances = set(self.knowledge_base.individuals(concept))
        tp = len(pos.intersection(instances))
        tn = len(neg.difference(instances))

        fp = len(neg.intersection(instances))
        fn = len(pos.difference(instances))
        _, f1_score = self.quality_func.score2(tp=tp, fn=fn, fp=fp, tn=tn)
        return f1_score

    def fit(self, lp: PosNegLPStandard, max_runtime=None):
        str_pos_examples = [i.get_iri().as_str() for i in lp.pos]
        str_neg_examples = [i.get_iri().as_str() for i in lp.neg]

        X, y = self.labeling(pos=str_pos_examples, neg=str_neg_examples, apply_dummy=False)
        # Binaries
        self.clf = tree.DecisionTreeClassifier(max_depth=10, random_state=0).fit(X=X.values, y=y.values)

        # plt.figure(figsize=(30, 30))
        # tree.plot_tree(self.clf, fontsize=10, feature_names=X.columns.to_list())
        # plt.show()
        from owlapy.render import DLSyntaxObjectRenderer
        render=DLSyntaxObjectRenderer()
        representation_of_positives = X.loc[str_pos_examples].values
        dl_concepts = set()
        for sequence_of_reasoning_steps, single_positive_individuals in zip(
                explain_inference(self.clf, X_test=representation_of_positives, features=X.columns.to_list(),
                                  only_shared=False), str_pos_examples):
            predicted_class=self.clf.predict(X.loc[single_positive_individuals].values.reshape(1, -1))
            assert predicted_class==1
            for step, reasoning_step in enumerate(sequence_of_reasoning_steps):
                # print(f"\t{reasoning_step}")
                # tail can be individual or class
                relation, tail = reasoning_step["feature"]
                # from numpy.bool_ to python bool
                value = bool(reasoning_step["value"])
                if relation == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                    if value:
                        owl_class = self.owl_classes_dict[tail]
                    else:
                        owl_class = self.owl_classes_dict[tail].get_object_complement_of()

                    assert single_positive_individuals in [i.get_iri().as_str() for i in
                                                           self.knowledge_base.individuals(owl_class)]

                    dl_concepts.add(owl_class)
                else:

                    if value:
                        dl_concepts.add(OWLObjectHasValue(property=self.owl_object_property_dict[relation], individual=self.owl_individuals[tail]))
                        for i in self.knowledge_base.get_types(self.owl_individuals[tail]):
                            dl_concepts.add(OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[relation], filler=i))

                    else:
                        continue
                        #owl_class = OWLObjectHasValue(property=self.owl_object_property_dict[relation],
                        #                              individual=self.owl_individuals[tail]).get_object_complement_of()



                """
                predicate, tail = step["feature"]

                if predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                    if step['value']:
                        decision.append(self.owl_classes_dict[tail])
                    else:
                        decision.append(self.owl_classes_dict[tail].get_object_complement_of())
                else:
                    types = self.dataframe_triples[(self.dataframe_triples.subject == tail) & (
                            self.dataframe_triples.relation == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")][
                        "object"].tolist()
                    for type_tail in types:
                        exist_decision.append(self.knowledge_base.generator.existential_restriction(
                            property=self.owl_object_property_dict[predicate], filler=self.owl_classes_dict[type_tail]))
                """
        # @TODO: Do something with them !
        for i in dl_concepts:
            print(render.render(i), end="\t")
            f1_score = self.compute_quality(concept=i, pos=lp.pos, neg=lp.neg)
            print(f1_score)
        exit(1)
        pass
