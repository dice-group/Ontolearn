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
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLDataHasValue, OWLObjectHasValue
from owlapy.render import DLSyntaxObjectRenderer


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


def base_construct_second(cbd_entities: Dict[str, Set[Tuple[str, str]]], individuals: List[str],
                          feature_names: List[Tuple[str, Union[str, None]]]):
    """ Construct a tabular representations from fixed features """
    assert cbd_entities is not None, "No cbd entities"
    result = []
    # () Iterate over individuals.
    for s in individuals:
        # () Initialize an empty row.
        representation_of_s = [False for _ in feature_names]
        for (p, o) in cbd_entities[s]:
            """ o can be a IRI or a number a boolean"""
            # () if (p,o) not in feature_names, o must be a number
            if (p, o) in feature_names:
                if o is not None:
                    idx = feature_names.index((p, o))
                    value = True
                    assert representation_of_s[idx] is False
                else:
                    "Ignore information comes as p,o "
                    print(p, o)
                    exit(1)
                    idx = feature_names.index((p, None))
                    value = o

                representation_of_s[idx] = value
        result.append(representation_of_s)
    result = pd.DataFrame(data=result, index=individuals, columns=feature_names, dtype="category")
    # result = pd.DataFrame(data=result, index=individuals, columns=feature_names)
    # print("Tabular data representing positive and negative examples:", result.shape)
    result = result.loc[:, (result != False).any(axis=0)]
    # print("Tabular data representing positive and negative examples after removing uninformative features:",result.shape)
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
        # Keyword arguments for sklearn Decision tree.
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

        self.cbd_mapping_entities = {k: v for k, v in self.cbd_mapping.items() if k in self.str_individuals}
        # Type info
        self.features = [(self.str_type, str_c) for str_c in self.owl_classes_dict.keys()]
        # Object Info
        self.features.extend([(str_r, i) for i in self.str_individuals for str_r in self.owl_object_property_dict])
        # Data Info. None will be filled by object s.t. i str_r, object
        self.features.extend([(str_r, None) for str_r in self.owl_data_property_dict])
        # Initialize classifier
        self.clf = None

        if self.on_fly_tabular:
            # Trade-off between runtime at inference and memory
            self.Xraw = None
        else:
            self.Xraw = base_construct_second(cbd_entities=self.cbd_mapping_entities,
                                              individuals=self.str_individuals,
                                              feature_names=self.features)

    def labeling(self, Xraw, pos, neg, apply_dummy=True):
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

    def decision_to_owl_class_exp(self, reasoning_step: dict, single_positive_indv):
        """ """
        # print(f"\t{reasoning_step}")
        # tail can be individual or class
        relation, tail = reasoning_step["feature"]
        # from numpy.bool_ to python bool
        value = bool(reasoning_step["value"])
        if relation == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            if value:
                owl_class = self.owl_classes_dict[tail]
                # assert self.owl_individuals[single_positive_indv] in self.knowledge_base.individuals(owl_class)
            else:
                owl_class = self.owl_classes_dict[tail].get_object_complement_of()
                # assert self.owl_individuals[single_positive_indv] in self.knowledge_base.individuals(owl_class)
        else:
            if tail in self.owl_individuals:
                owl_class = OWLObjectHasValue(property=self.owl_object_property_dict[relation],
                                              individual=self.owl_individuals[tail])
            else:
                owl_class = OWLDataHasValue(property=self.owl_data_property_dict[relation], value=OWLLiteral(tail))

            if value:
                pass
                # assert self.owl_individuals[single_positive_indv] in self.knowledge_base.individuals(owl_class)
            else:
                owl_class = owl_class.get_object_complement_of()
                # assert self.owl_individuals[single_positive_indv] in self.knowledge_base.individuals(owl_class)

        return owl_class

    def best_hypotheses(self, n=1):
        assert n == 1
        return self.best_pred

    def fit(self, lp: PosNegLPStandard, max_runtime=None):
        if max_runtime is not None:
            self.max_runtime = max_runtime

        str_pos_examples = [i.get_iri().as_str() for i in lp.pos]
        str_neg_examples = [i.get_iri().as_str() for i in lp.neg]

        if self.on_fly_tabular:
            # print("Constructing representations on the fly...")
            Xraw = base_construct_second(cbd_entities=self.cbd_mapping_entities,
                                         individuals=str_pos_examples + str_neg_examples,
                                         feature_names=self.features)
            X, y = self.labeling(Xraw=Xraw, pos=str_pos_examples, neg=str_neg_examples, apply_dummy=False)
        else:
            X, y = self.labeling(Xraw=self.Xraw, pos=str_pos_examples, neg=str_neg_examples, apply_dummy=False)

        # Binaries
        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values)
        # print("Classification Report: Negatives: -1, Unknowns:0, Positives 1 ")
        # print(sklearn.metrics.classification_report(y.values, self.clf.predict(X.values), target_names=None))
        # plt.figure(figsize=(30, 30))
        # tree.plot_tree(self.clf, fontsize=10, feature_names=X.columns.to_list())
        # plt.show()

        prediction_per_example = []
        # () Iterate over E^+
        for sequence_of_reasoning_steps, pos in zip(
                explain_inference(self.clf,
                                  X_test=X.loc[str_pos_examples].values,
                                  features=X.columns.to_list(),
                                  only_shared=False), str_pos_examples):
            # () Ensure that e \in E^+ is classified as positive
            # assert 1 == self.clf.predict(X.loc[pos].values.reshape(1, -1))
            # () Reasoning behind of the prediction of a single positive example.

            sequence_of_concept_path_of_tree = [self.decision_to_owl_class_exp(reasoning_step, pos) for
                                                reasoning_step in
                                                sequence_of_reasoning_steps]
            pred = concepts_reducer(concepts=sequence_of_concept_path_of_tree, reduced_cls=OWLObjectIntersectionOf)
            prediction_per_example.append((pred, pos))

        self.best_pred = concepts_reducer(concepts=[pred for pred, pos in prediction_per_example],
                                          reduced_cls=OWLObjectUnionOf)

        return self
