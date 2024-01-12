import numpy as np
import owlapy.model
import pandas as pd

from ontolearn.knowledge_base import KnowledgeBase
from typing import Dict, Set, Tuple, List, Union, TypeVar, Callable
from ontolearn.learning_problem import PosNegLPStandard
import collections
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree

from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLDataHasValue, OWLObjectHasValue, OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer
from sklearn.model_selection import GridSearchCV

import time


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def compute_quality(instances, pos, neg, conf_matrix=False, quality_func=None):
    assert isinstance(instances, set)
    tp = len(pos.intersection(instances))
    tn = len(neg.difference(instances))

    fp = len(neg.intersection(instances))
    fn = len(pos.difference(instances))

    _, f1_score = quality_func.score2(tp=tp, fn=fn, fp=fp, tn=tn)
    if conf_matrix:
        return f1_score, f"TP:{tp}\tFN:{fn}\tFP:{fp}\tTN:{tn}"
    return f1_score


def extract_cbd(dataframe) -> Dict[str, List[Tuple[str, str]]]:
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
        data.setdefault(subject_, []).append((predicate_, object_))
    return data


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


def concepts_reducer(concepts: List[OWLClassExpression], reduced_cls: Callable) -> Union[
    OWLObjectUnionOf, OWLObjectIntersectionOf]:
    """ Reduces a list of OWLClassExpression instances into a single instance of OWLObjectUnionOf or OWLObjectIntersectionOf """
    dl_concept_path = None
    for c in concepts:
        assert isinstance(c, OWLClassExpression), f"c is not OWL: {type(c)}"
        if dl_concept_path is None:
            dl_concept_path = c
        else:
            dl_concept_path = reduced_cls((dl_concept_path, c))
    return dl_concept_path


def compute_f1_score(individuals, pos, neg):
    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))

    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0

    if precision == 0 or recall == 0:
        return 0

    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1


class TDL:
    """Tree-based Description Logic Concept Learner"""

    def __init__(self, knowledge_base, dataframe_triples: pd.DataFrame, kwargs_classifier,
                 on_fly_tabular: bool = True, max_runtime=1):
        assert isinstance(dataframe_triples, pd.DataFrame), "dataframe_triples must be a Pandas DataFrame"
        assert isinstance(knowledge_base, KnowledgeBase), "knowledge_base must be a KnowledgeBase instance"
        assert len(dataframe_triples) > 0, f"length of the dataframe must be greater than 0:{dataframe_triples.shape}"
        # print(f"Knowledge Base: {knowledge_base}")
        # print(f"Matrix representation of knowledge base: {dataframe_triples.shape}")
        self.knowledge_base = knowledge_base
        self.dataframe_triples = dataframe_triples
        # Mappings from string of IRI to named concepts.
        self.owl_classes_dict = {c.get_iri().as_str(): c for c in self.knowledge_base.get_concepts()}
        # Mappings from string of IRI to object properties.
        self.owl_object_property_dict = {p.get_iri().as_str(): p for p in self.knowledge_base.get_object_properties()}
        # Mappings from string of IRI to data properties.
        self.owl_data_property_dict = {p.get_iri().as_str(): p for p in self.knowledge_base.get_data_properties()}
        # Mappings from string of IRI to individuals.
        self.owl_individuals = {i.get_iri().as_str(): i for i in self.knowledge_base.individuals()}

        self.render = DLSyntaxObjectRenderer()
        # Keyword arguments for sklearn Decision tree.
        # Initialize classifier
        self.clf = None
        self.feature_names = None
        self.kwargs_classifier = kwargs_classifier

        self.max_runtime = max_runtime
        self.on_fly_tabular = on_fly_tabular
        self.best_pred = None
        # Remove uninformative triples if exists.
        # print("Removing uninformative triples...")
        self.dataframe_triples = self.dataframe_triples[
            ~((self.dataframe_triples["relation"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type") & (
                    (self.dataframe_triples["object"] == "http://www.w3.org/2002/07/owl#NamedIndividual") | (
                    self.dataframe_triples["object"] == "http://www.w3.org/2002/07/owl#Thing") | (
                            self.dataframe_triples["object"] == "Ontology")))]
        # print(f"Matrix representation of knowledge base: {dataframe_triples.shape}")
        self.cbd_mapping: Dict[str, Set[Tuple[str, str]]]
        self.cbd_mapping = extract_cbd(self.dataframe_triples)
        self.str_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        # Fix an ordering: Not quite sure whether we needed
        self.str_individuals = list(self.owl_individuals)
        # An entity to a list of tuples of predicate and objects
        self.first_hop = {k: v for k, v in self.cbd_mapping.items() if k in self.str_individuals}
        self.types_of_individuals = dict()

        for k, v in self.first_hop.items():
            for relation, tail in v:
                if relation == self.str_type:
                    self.types_of_individuals.setdefault(k, set()).add(tail)

        if self.on_fly_tabular:
            # Trade-off between runtime at inference and memory
            self.Xraw = None
        else:
            raise NotImplementedError()

    def built_sparse_training_data(self, entity_infos: Dict[str, Dict], individuals: List[str],
                                   feature_names: List[Tuple[str, Union[str, None]]]):
        """ Construct a tabular representations from fixed features """
        assert entity_infos is not None, "No entity_infos"
        result = []
        # () Iterate over individuals.
        for s in individuals:
            # () Initialize an empty row.
            representation_of_s = [0.0 for _ in feature_names]
            # All info about s should be in the features.
            for relation, hop_info in entity_infos[s].items():
                assert isinstance(relation, str), "Relation must be string"
                for t in hop_info:
                    if isinstance(t, str):
                        representation_of_s[feature_names.index((relation, t))] = 1.0
                    elif isinstance(t, tuple):
                        assert len(t) == 2
                        representation_of_s[feature_names.index((relation, *t))] = 1.0
            result.append(representation_of_s)
        result = pd.DataFrame(data=result, index=individuals, columns=feature_names, dtype=np.float32)
        result = result.loc[:, (result != False).any(axis=0)]
        return result

    def construct_hop(self, individuals: List[str]) -> Dict[str, Dict]:
        assert len(individuals) == len(set(individuals)), "There are duplicate individuals"

        # Nested dictionary
        hop = dict()
        features = set()
        # Iterate over individuals.
        for s in individuals:
            # iterate over triples where s is a subject.
            temp = dict()
            # Given a specific individual, iterate over matching relations and tails.
            for p, o in self.first_hop[s]:
                ##### SAVE FEATURE: (type, PERSON) #####
                if p == self.str_type:
                    # F3 hasChild Male.
                    assert o in self.owl_classes_dict
                    temp.setdefault(p, set()).add(o)
                    features.add((p, o))
                else:
                    # if the relation leads to an individual, iterate over the first hop of the individual, e.g.
                    # from F3 hasChild F12, => [type Female, type Mother, hasChild F44] since
                    # (F12 type Female), (F12 type Mother), (F12 hasChild F44) \in KG.
                    for (pp, oo) in self.first_hop[o]:
                        if pp == self.str_type:
                            ##### SAVE FEATURE: (hasChild, PERSON) #####
                            assert oo in self.owl_classes_dict
                            temp.setdefault(p, set()).add(oo)
                            features.add((p, oo))
                        else:
                            for c in self.types_of_individuals[oo]:
                                ##### SAVE FEATURE: (hasChild, married, Father) #####
                                temp.setdefault(p, set()).add((pp, c))
                                features.add((p, pp, c))
            hop[s] = temp
        return hop, features

    def labeling(self, Xraw, pos, neg, apply_dummy=False):
        """ Labelling """
        # (5) Labeling: Label each row/node
        # Drop "label" if exists

        Xraw.loc[:, "label"] = 0  # unknowns
        Xraw.loc[pos, "label"] = 1  # positives
        Xraw.loc[neg, "label"] = -1  # negatives
        # (5.1) drop unknowns although unknowns provide info
        X = Xraw  # self.Xraw[self.Xraw.label != 0]

        raw_features = X.columns.tolist()
        raw_features.remove("label")
        if apply_dummy:
            X_train_sparse = pd.get_dummies(X[raw_features])
        else:
            X_train_sparse = X[raw_features]
        y_train_sparse = X.loc[:, "label"]

        # print(f"Train data shape:{X_train_sparse.shape}")
        return X_train_sparse, y_train_sparse

    def decision_to_owl_class_exp(self, reasoning_step: dict):
        """ """
        # tail can be individual or class
        feature = reasoning_step["feature"]
        # relation, tail_info = reasoning_step["feature"]
        # from numpy.bool_ to python bool
        value = bool(reasoning_step["value"])
        if len(feature) == 2:
            relation, tail_info = feature
            if relation == self.str_type:
                assert isinstance(tail_info,
                                  str) and tail_info in self.owl_classes_dict, "Tail must be a string and a defined OWL class"
                if value:
                    owl_class = self.owl_classes_dict[tail_info]
                else:
                    owl_class = self.owl_classes_dict[tail_info].get_object_complement_of()
            else:
                rel1, tail = feature

                """
                owl_class = OWLObjectMinCardinality(property=self.owl_object_property_dict[rel1],
                                                    filler=self.owl_classes_dict[tail], cardinality=1)
                """

                owl_class = OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[rel1],
                                                    filler=self.owl_classes_dict[tail])

                """
                
                owl_class = OWLObjectHasValue(property=self.owl_object_property_dict[relation],
                                              individual=self.owl_individuals[tail_info])
                """
        else:
            assert len(feature) == 3
            rel1, rel2, concept = feature
            """
            owl_class = OWLObjectMinCardinality(property=self.owl_object_property_dict[rel1],
                                                filler=OWLObjectMinCardinality(
                                                    property=self.owl_object_property_dict[rel2],
                                                    filler=self.owl_classes_dict[concept], cardinality=1),
                                                cardinality=1)
            """
            owl_class = OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[rel1],
                                                filler=OWLObjectSomeValuesFrom(
                                                    property=self.owl_object_property_dict[rel2],
                                                    filler=self.owl_classes_dict[concept]))
            if value:
                pass
            else:
                owl_class = owl_class.get_object_complement_of()

        return owl_class

    def plot(self):
        pretified_feature_names = []
        for i in self.feature_names:
            f = []
            for x in i:
                x = x.replace("http://www.benchmark.org/family#", "")
                x = x.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "")
                f.append(x)
            pretified_feature_names.append(f)

        plt.figure(figsize=(10, 10))
        tree.plot_tree(self.clf, fontsize=10, feature_names=pretified_feature_names,
                       class_names=["Negative", "Positive"],
                       filled=True)
        plt.savefig('Aunt_Tree.pdf')
        plt.show()

        feature_importance = pd.Series(np.array(self.clf.feature_importances_),
                                       index=[",".join(i) for i in pretified_feature_names])
        feature_importance = feature_importance[feature_importance > 0.0]
        fig, ax = plt.subplots()
        feature_importance.plot.bar(ax=ax)
        ax.set_title("Feature Importance")
        fig.tight_layout()
        plt.savefig('feature_importance.pdf')
        plt.show()

    def fit(self, lp: PosNegLPStandard, max_runtime=None):
        if max_runtime is not None:
            self.max_runtime = max_runtime

        str_pos_examples = [i.get_iri().as_str() for i in lp.pos]
        str_neg_examples = [i.get_iri().as_str() for i in lp.neg]

        """self.features.extend([(str_r, None) for str_r in self.owl_data_property_dict])"""
        # Nested dictionary [inv][relation]: => [] Dict[str, Dict]
        hop_info, features = self.construct_hop(str_pos_examples + str_neg_examples)

        # list of tuples having length 2 or 3
        features = list(features)

        Xraw = self.built_sparse_training_data(entity_infos=hop_info,
                                               individuals=str_pos_examples + str_neg_examples,
                                               feature_names=features)
        X, y = self.labeling(Xraw=Xraw, pos=str_pos_examples, neg=str_neg_examples)
        if False:
            import umap
            print("Fitting")
            reducer = umap.UMAP(random_state=1)
            embedding = reducer.fit_transform(X)
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        c=["r" if x == 1 else "b" for x in y])
            plt.grid()
            plt.gca().set_aspect('equal', 'datalim')
            plt.savefig("UMAP_AUNT.pdf")
            plt.show()

        param_grid = {'criterion': ["entropy", "gini", "log_loss"],
                      "min_samples_leaf": [1, 5]
                      # 'max_depth': [3, 5, 10, 20, None],
                      # 'min_samples_split': [20, 30, 40],
                      # "max_leaf_nodes": [None, 4, 8]
                      }
        grid_search = GridSearchCV(tree.DecisionTreeClassifier(**self.kwargs_classifier),
                                   param_grid=param_grid,
                                   cv=10).fit(X.values, y.values)
        print(grid_search.best_params_)
        self.kwargs_classifier.update(grid_search.best_params_)
        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values)
        self.feature_names = X.columns.to_list()
        """
        print("Classification Report: Negatives: -1, Unknowns:0, Positives 1 ")
        print(sklearn.metrics.classification_report(y.values, self.clf.predict(X.values), target_names=None))
        plt.figure(figsize=(30, 30))
        tree.plot_tree(self.clf, fontsize=10, feature_names=X.columns.to_list())
        plt.show()
        """

        prediction_per_example = []
        # () Iterate over E^+
        for sequence_of_reasoning_steps, pos in zip(
                explain_inference(self.clf,
                                  X_test=X.loc[str_pos_examples].values,
                                  features=X.columns.to_list(),
                                  only_shared=False), str_pos_examples):
            sequence_of_concept_path_of_tree = [self.decision_to_owl_class_exp(reasoning_step) for
                                                reasoning_step in
                                                sequence_of_reasoning_steps]

            pred = concepts_reducer(concepts=sequence_of_concept_path_of_tree, reduced_cls=OWLObjectIntersectionOf)

            prediction_per_example.append((pred, pos))

        # Remove paths from the root to leafs if overallping
        prediction_per_example = {p for p, indv in prediction_per_example}
        # If debugging needed.
        """
        set_str_pos = set(str_pos_examples)
        set_str_neg = set(str_neg_examples)
        for i in prediction_per_example:
            quality = compute_f1_score(individuals={_.get_iri().as_str() for _ in self.knowledge_base.individuals(i)},
                                       pos=set_str_pos, neg=set_str_neg)
            print(self.render.render(i), quality)
        """
        predictions = [pred for pred in prediction_per_example]
        """
        print(len(predictions))
        for i in predictions:
            print(self.render.render(i))
        """

        self.best_pred = concepts_reducer(concepts=predictions,
                                          reduced_cls=OWLObjectUnionOf)

        # print(self.render.render(self.best_pred), compute_f1_score(individuals={_.get_iri().as_str() for _ in self.knowledge_base.individuals(self.best_pred)},
        #                               pos=set(str_pos_examples), neg=set(str_neg_examples)))

        return self

    def best_hypotheses(self, n=1):
        assert n == 1
        return self.best_pred

    def predict(self, X: List[OWLNamedIndividual]) -> np.ndarray:
        #
        owl_individuals = [i.get_iri().as_str() for i in X]
        X_vector = built_sparse_training_data(entity_infos=self.cbd_mapping_entities,
                                              individuals=owl_individuals,
                                              feature_names=self.features)
        return self.clf.predict(X_vector.values)
