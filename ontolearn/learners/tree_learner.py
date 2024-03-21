import numpy as np
import owlapy.model
import pandas as pd
import requests
import json
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.base import OWLOntologyManager_Owlready2
from owlapy.model import OWLEquivalentClassesAxiom, OWLOntologyManager, OWLOntology, AddImport, OWLImportsDeclaration, \
    IRI, OWLDataOneOf, OWLObjectProperty, OWLDataProperty

from typing import Dict, Set, Tuple, List, Union, TypeVar, Callable
from ontolearn.learning_problem import PosNegLPStandard
import collections
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
from tqdm import tqdm

from owlapy.model import OWLObjectSomeValuesFrom, OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLDataHasValue, OWLObjectHasValue, OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer, ManchesterOWLSyntaxOWLObjectRenderer
from sklearn.model_selection import GridSearchCV

import time

from sklearn.tree import export_text


# pd.pandas.set_option('display.max_columns', None)

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


class TDL:
    """Tree-based Description Logic Concept Learner"""

    def __init__(self, knowledge_base,
                 kwargs_classifier: dict = None,
                 max_runtime: int = 1,
                 grid_search_over: dict = None,
                 grid_search_apply: bool = False,
                 report_classification: bool = False,
                 plot_built_tree: bool = False,
                 plotembeddings: bool = False):
        assert isinstance(knowledge_base, KnowledgeBase), "knowledge_base must be a KnowledgeBase instance"
        print(f"Knowledge Base: {knowledge_base}")
        self.knowledge_base = knowledge_base
        if grid_search_over is None and grid_search_apply:
            grid_search_over = {'criterion': ["entropy", "gini", "log_loss"],
                                "splitter": ["random", "best"],
                                "max_features": [None, "sqrt", "log2"],
                                "min_samples_leaf": [1, 2, 3, 4, 5, 10],
                                "max_depth": [1, 2, 3, 4, 5, 10, None]}
        else:
            grid_search_over = dict()

        self.grid_search_over = grid_search_over
        self.report_classification = report_classification
        self.plot_built_tree = plot_built_tree
        self.plotembeddings = plotembeddings
        # Mappings from string of IRI to named concepts.
        self.owl_classes_dict = dict()
        # Mappings from string of IRI to object properties.
        self.owl_object_property_dict = dict()
        """
        # Mappings from string of IRI to data properties.
        self.owl_data_property_dict = {p.get_iri().as_str(): p for p in self.knowledge_base.get_data_properties()}
        # Mappings from string of IRI to individuals.
        self.owl_individuals = {i.get_iri().as_str(): i for i in self.knowledge_base.individuals()}
        """

        # Concept renderers
        self.dl_render = DLSyntaxObjectRenderer()
        self.manchester_render = ManchesterOWLSyntaxOWLObjectRenderer()
        # Keyword arguments for sklearn Decision tree.
        # Initialize classifier
        self.clf = None
        self.feature_names = None
        self.kwargs_classifier = kwargs_classifier if kwargs_classifier is not None else dict()
        self.max_runtime = max_runtime
        # best pred
        self.disjunction_of_conjunctive_concepts = None
        self.conjunctive_concepts = None

        # print(f"Matrix representation of knowledge base: {dataframe_triples.shape}")
        self.cbd_mapping: Dict[str, Set[Tuple[str, str]]]
        # self.cbd_mapping = extract_cbd(self.dataframe_triples)
        self.cbd_mapping = None
        self.type = IRI('http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'type')
        # Fix an ordering: Not quite sure whether we needed
        # self.str_individuals = list(self.owl_individuals)
        # An entity to a list of tuples of predicate and objects
        # self.first_hop = {k: v for k, v in self.cbd_mapping.items() if k in self.str_individuals}
        self.types_of_individuals = dict()
        """
        for k, v in self.first_hop.items():
            for relation, tail in v:
                if relation == self.str_type:
                    self.types_of_individuals.setdefault(k, set()).add(tail)
        """

        self.Xraw = None

    def fit(self, learning_problem: PosNegLPStandard = None, max_runtime: int = None):
        """ Fit the learner to the given learning problem

        (1) Extract multi-hop information about E^+ and E^- denoted by \mathcal{F}.
        (1.1) E = list of (E^+ \sqcup E^-).
        (2) Build a training data \mathbf{X} \in  \mathbb{R}^{ |E| \times |\mathcal{F}| } .
        (3) Create binary labels \mathbf{X}.

        (4) Construct a set of DL concept for each e \in E^+
        (5) Union (4)
        :param learning_problem: The learning problem
        :param max_runtime:total runtime of the learning

        """
        assert learning_problem is not None, "Learning problem cannot be None."
        if max_runtime is not None:
            self.max_runtime = max_runtime

        pos_examples = [i for i in learning_problem.pos]
        neg_examples = [i for i in learning_problem.neg]
        examples = pos_examples + neg_examples

        """self.features.extend([(str_r, None) for str_r in self.owl_data_property_dict])"""
        # Nested dictionary [inv][relation]: => [] Dict[str, Dict]
        hop_info, features = self.extract_multihop_info_about_individuals(examples)

        # list of tuples having length 2 or 3
        features = list(features)

        Xraw = self.built_sparse_training_data(entity_infos=hop_info,
                                               individuals=examples,
                                               feature_names=features)
        X, y = self.labeling(Xraw=Xraw, pos=str_pos_examples, neg=str_neg_examples)

        if self.grid_search_over:
            n, dim = X.shape
            if n > 10:
                num_cv = 10
            else:
                num_cv = 2
            grid_search = GridSearchCV(tree.DecisionTreeClassifier(**self.kwargs_classifier),
                                       param_grid=self.grid_search_over, cv=num_cv).fit(X.values, y.values)
            print(grid_search.best_params_)
            self.kwargs_classifier.update(grid_search.best_params_)

        self.clf = tree.DecisionTreeClassifier(**self.kwargs_classifier).fit(X=X.values, y=y.values)
        self.feature_names = X.columns.to_list()
        if self.report_classification:
            print("Classification Report: Negatives: -1 and Positives 1 ")
            print(sklearn.metrics.classification_report(y.values, self.clf.predict(X.values),
                                                        target_names=["Negative", "Positive"]))
        if self.plot_built_tree:
            self.plot()

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
        self.conjunctive_concepts = [pred for pred in prediction_per_example]

        self.disjunction_of_conjunctive_concepts = concepts_reducer(concepts=self.conjunctive_concepts,
                                                                    reduced_cls=OWLObjectUnionOf)
        return self

    def extract_multihop_info_about_individuals(self, individuals: List[OWLNamedIndividual]) -> Dict[str, Dict]:
        # (1) Ensure that no duplicates
        assert len(individuals) == len(set(individuals)), "There are duplicate individuals"
        # (2) Nested dictionary.
        hop = dict()
        # (3) Unique features/DL concepts.
        features = set()
        # (4) Iterate over individuals.
        for s in tqdm(individuals, desc="Building Feature Matrix"):
            temp = dict()
            # (5) iterate over triples of (s,p,o).
            for _, p, o in self.knowledge_base.abox(individuals=s,
                                                    class_assertions=True,
                                                    object_property_assertions=True,
                                                    data_property_assertions=True):
                # Ensure that s is the subject entity in all triples
                assert s == _
                # print(f"Iterating over triples of (s,p,o): {s} {p} {o}")
                if isinstance(o, OWLClass):
                    # (4.1) FEATURE: OWLClass Membership #####
                    assert p == self.type
                    temp.setdefault(p, set()).add(o)
                    features.add((p, o))
                elif isinstance(o, OWLNamedIndividual) and isinstance(p, OWLObjectProperty):
                    ##### FEATURE: \exist p. {o} #####
                    temp.setdefault(p, set()).add(o)
                    features.add((p, o))

                    ##### FEATURE: \exist p. type(o) #####
                    for type_o in self.knowledge_base.get_types(o):
                        temp.setdefault(p, set()).add((self.type, type_o))
                        features.add((p, self.type, type_o))

                    ##### FEATURE: \exist p. (\exist pp. {oo}) s.t. (o pp oo) holds in Abox #####
                    for __, pp, oo in self.knowledge_base.abox(individuals=o,
                                                               class_assertions=False,
                                                               object_property_assertions=True,
                                                               data_property_assertions=False):
                        # hasChild, hassibling, max.
                        temp.setdefault(p, set()).add((pp, oo))
                        features.add((p, pp, oo))

                        ##### FEATURE: \exist r. (\exist pp. type(oo}) s.t. (o pp oo) and (oo type C) holds in Abox #####
                        for type_oo in self.knowledge_base.get_types(oo):
                            # hasChild, hassibling, Female.
                            temp.setdefault(p, set()).add((pp, type_oo))
                            features.add((p, pp, type_oo))

            hop[s] = temp
        return hop, features

    def built_sparse_training_data(self, entity_infos: Dict[OWLNamedIndividual, Dict], individuals: List[str],
                                   feature_names: List[Tuple[Union[OWLDataProperty, OWLObjectProperty, IRI], Union[str, None]]]):
        """ Construct a tabular representations from fixed features


        feature_names: A list of tuples where the first item in a tuple is an instance of OWLDataProperty or OWLObjectProperty or IRI.
        IRI denotes instance of IRI('http://www.w3.org/1999/02/22-rdf-syntax-ns#','type').
        """

        assert entity_infos is not None, "No entity_infos"
        result = []
        for i in feature_names:
            print(i)
            print(type(i))
        exit(1)
        # () Iterate over individuals.
        for s in individuals:
            # () Initialize an empty row.
            representation_of_s = [0.0 for _ in feature_names]
            # All info about s should be in the features.
            for relation, hop_info in entity_infos[s].items():
                assert isinstance(relation, str), "Relation must be string"
                for t in hop_info:
                    if isinstance(t, str):
                        if relation == self.str_type:
                            # assert t in self.owl_classes_dict
                            # Boolean feature : (type, CLASS):
                            representation_of_s[feature_names.index((relation, t))] = 1.0
                        elif relation == self.owl_object_property_dict:
                            # Boolean feature : (hasChild, Individual)
                            assert t in self.str_individuals
                            representation_of_s[feature_names.index((relation, t))] = 1.0
                        elif relation == self.owl_object_property_dict:
                            # Numerical Feature : (hasCharge, None)
                            assert t not in self.str_individuals
                            assert is_float(t)

                            print("hereee")
                            print(s, relation, t)
                            representation_of_s[feature_names.index((relation, None))] = t
                            exit(1)
                    elif isinstance(t, tuple):
                        if len(t) == 2:
                            rr, oo = t
                            if rr in self.owl_data_property_dict:
                                # Feature : hasSibling, hasCharge, NUMBER
                                assert is_float(oo)

                                representation_of_s[feature_names.index((relation, rr, None))] = eval(oo)
                            else:
                                assert rr in self.owl_object_property_dict
                                assert relation in self.owl_object_property_dict
                                assert oo in self.owl_classes_dict
                                representation_of_s[feature_names.index((relation, rr, oo))] = 1.0

                        else:
                            print(t)
                            print("ASDAD")
                            exit(1)
                            representation_of_s[feature_names.index((relation, *t))] = 1.0
                    else:
                        print("asda")
                        print(s, relation, t)
                        print(t)
                        print("BURASI")
                        exit(1)
            result.append(representation_of_s)
        result = pd.DataFrame(data=result, index=individuals, columns=feature_names)  # , dtype=np.float32)
        # result = result.loc[:, (result != False).any(axis=0)]

        return result

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

        if self.plotembeddings:
            import umap
            print("Fitting")
            reducer = umap.UMAP(random_state=1)
            embedding = reducer.fit_transform(X_train_sparse)
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        c=["r" if x == 1 else "b" for x in y_train_sparse])
            plt.grid()
            plt.gca().set_aspect('equal', 'datalim')
            plt.savefig("figure.pdf")
            plt.show()

        return X_train_sparse, y_train_sparse

    def decision_to_owl_class_exp(self, reasoning_step: dict):
        """ """
        # tail can be individual or class
        feature = reasoning_step["feature"]
        # relation, tail_info = reasoning_step["feature"]
        if len(feature) == 2:
            relation, tail_info = feature
            if relation == self.str_type:
                assert isinstance(tail_info, str), "Tail must be a string"
                # assert tail_info in self.owl_classes_dict, "a defined OWL class"
                assert reasoning_step["value"] == 0.0 or reasoning_step["value"] == 1.0
                if bool(reasoning_step["value"]):
                    owl_class = self.owl_classes_dict[tail_info]
                else:
                    owl_class = self.owl_classes_dict[tail_info].get_object_complement_of()
            elif relation in self.owl_data_property_dict:
                # To capture this ('http://dl-learner.org/mutagenesis#hasThreeOrMoreFusedRings', None)
                print("HEREEEE")
                print(relation)
                raise RuntimeError("UNCLEAR")
            else:
                rel1, tail = feature
                if rel1 in self.owl_object_property_dict:
                    owl_class = OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[rel1],
                                                        filler=self.owl_classes_dict[tail])
                else:
                    owl_class = OWLDataHasValue(property=self.owl_data_property_dict[rel1], value=OWLLiteral(tail))

                print("WHAT SHOULD BE")
                print(feature)
                print(reasoning_step["value"])
                raise RuntimeError("UNCLEAR")
        else:
            assert len(feature) == 3
            rel1, rel2, concept = feature

            if concept is None:
                assert rel2 in self.owl_data_property_dict
                assert is_float(reasoning_step["value"])
                owl_class = OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[rel1],
                                                    filler=OWLDataHasValue(property=self.owl_data_property_dict[rel2],
                                                                           value=OWLLiteral(
                                                                               float(reasoning_step["value"]))))
            elif rel2 in self.owl_object_property_dict:
                filler = OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[rel2],
                                                 filler=self.owl_classes_dict[concept])
                owl_class = OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[rel1], filler=filler)

                assert reasoning_step["value"] == 0.0 or reasoning_step["value"] == 1.0
                if bool(reasoning_step["value"]):
                    pass
                else:
                    owl_class = owl_class.get_object_complement_of()

            else:

                raise RuntimeError("UNCLEAR")
                assert rel2 in self.owl_data_property_dict
                print(reasoning_step)

                owl_class = OWLObjectSomeValuesFrom(property=self.owl_object_property_dict[rel1],
                                                    filler=OWLDataSomeValuesFrom(
                                                        property=self.owl_data_property_dict[rel2],
                                                        filler=OWLLiteral(float(reasoning_step["value"]))))

        return owl_class

    def feature_pretify(self):
        pretified_feature_names = []
        for i in self.feature_names:
            feature = ""
            for x in i:
                x = x.replace("http://www.benchmark.org/family#", "")
                x = x.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "")
                feature += x + " "
            pretified_feature_names.append(feature)
        return pretified_feature_names

    def best_hypotheses(self, n=1):
        """ Return the prediction"""
        assert n == 1, "Only one hypothesis is supported"
        return [self.disjunction_of_conjunctive_concepts]

    def predict(self, X: List[OWLNamedIndividual], proba=True) -> np.ndarray:
        """ Predict the likelihoods of individuals belonging to the classes"""
        owl_individuals = [i.get_iri().as_str() for i in X]
        hop_info, _ = self.construct_hop(owl_individuals)
        Xraw = self.built_sparse_training_data(entity_infos=hop_info,
                                               individuals=owl_individuals,
                                               feature_names=self.feature_names)
        # corrupt some infos
        Xraw_numpy = Xraw.values

        if proba:
            return self.clf.predict_proba(Xraw_numpy)
        else:
            return self.clf.predict(Xraw_numpy)

    def save_best_hypothesis(self, concepts: List[OWLClassExpression],
                             path: str = 'Predictions',
                             rdf_format: str = 'rdfxml', renderer=ManchesterOWLSyntaxOWLObjectRenderer()) -> None:
        """Serialise the best hypotheses to a file.
        @TODO: This should be a single static function We need to refactor it


        Args:
            concepts:
            path: Filename base (extension will be added automatically).
            rdf_format: Serialisation format. currently supported: "rdfxml".
            renderer: An instance of ManchesterOWLSyntaxOWLObjectRenderer
        """
        # NS: Final = 'https://dice-research.org/predictions/' + str(time.time()) + '#'
        NS: Final = 'https://dice-research.org/predictions#'
        if rdf_format != 'rdfxml':
            raise NotImplementedError(f'Format {rdf_format} not implemented.')
        # ()
        manager: OWLOntologyManager = OWLOntologyManager_Owlready2()
        # ()
        ontology: OWLOntology = manager.create_ontology(IRI.create(NS))
        # () Iterate over concepts
        for i in concepts:
            cls_a: OWLClass = OWLClass(IRI.create(NS, renderer.render(i)))
            equivalent_classes_axiom = OWLEquivalentClassesAxiom([cls_a, i])
            manager.add_axiom(ontology, equivalent_classes_axiom)

        manager.save_ontology(ontology, IRI.create('file:/' + path + '.owl'))

    def named_class_membership_feature(self, c: OWLClass):
        # For example, (hasChild Male).
        # assert o in self.owl_classes_dict
        # Mappings from string of IRI to named concepts.
        # self.owl_classes_dict = {c.get_iri().as_str(): c for c in self.knowledge_base.get_concepts()}
        str_concept = c.get_iri().as_str()
        self.owl_classes_dict.setdefault(str_concept, c)
        return str_concept

    def plot(self):
        """
        # plt.figure(figsize=(30, 30))
        # tree.plot_tree(self.clf, fontsize=10, feature_names=X.columns.to_list())
        # plt.show()

        """
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
