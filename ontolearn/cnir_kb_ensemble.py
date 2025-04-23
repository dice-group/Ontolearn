# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

""" CNIR Knowledge Base."""
from ontolearn.knowledge_base import KnowledgeBase
import time
from typing import Iterable, Optional, Callable, Union, FrozenSet, Set, Dict, cast, Generator
from owlapy.abstracts import AbstractOWLOntology, AbstractOWLReasoner
from owlapy.owl_hierarchy import ClassHierarchy, ObjectPropertyHierarchy, DatatypePropertyHierarchy
from owlapy.owl_individual import OWLNamedIndividual
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from owlapy.render import DLSyntaxObjectRenderer
from owlapy.parser import DLSyntaxParser
from owlapy.utils import iter_count

from cnir import InferenceDataset
from cnir.config import CNIRConfig
from cnir.models import CNIRComposite, CNIRTransformer, CNIRLSTM, CNIRGRU
from cnir.models.pmanet import PMAnet
from cnir.utils import str2bool, read_embs_and_apply_agg
from cnir.utils import score_all_inds, score_all_inds_composite


class CNIRKB(KnowledgeBase):
    def __init__(self,
                 path: Optional[str] = None,
                 reasoner_factory: Optional[
                     Callable[[AbstractOWLOntology], AbstractOWLReasoner]] = None,
                 ontology: Optional[AbstractOWLOntology] = None,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 class_hierarchy: Optional[ClassHierarchy] = None,
                 load_class_hierarchy: bool = True,
                 object_property_hierarchy: Optional[ObjectPropertyHierarchy] = None,
                 data_property_hierarchy: Optional[DatatypePropertyHierarchy] = None,
                 include_implicit_individuals=False,
                 dataset_dir=None, model_list=None, model_paths=None, use_pma=True,
                 pma_model_path=None,
                 tokenizer_path=None,
                 chunksize=1024, th=0.5):

        self.th = th
        self.pma_net = None
        # self.model_name = model
        self.model_list = model_list
        self.models = []
        self.chunksize = chunksize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model in model_list:
            if model.lower() == "composite" and use_pma:
                self.pma_net = PMAnet(CNIRConfig().embedding_dim, CNIRConfig().num_attention_heads,
                                      1)
                self.pma_net.load_state_dict(
                    torch.load(pma_model_path, map_location="cpu", weights_only=True)
                )
                self.pma_net.eval()

        self.kb, self.all_individuals, self.embeddings = read_embs_and_apply_agg(
            dataset_dir, nn_agg=self.pma_net, merge=True
        )
        self.dls_renderer = DLSyntaxObjectRenderer()
        self.all_individuals_set = set(self.all_individuals)
        self.all_individuals_arr = np.array(sorted(self.all_individuals), dtype=object)
        self.all_ind_embs = torch.FloatTensor(
            self.embeddings.loc[self.all_individuals_arr].values).to(self.device)

        kb_namespace = list(self.kb.ontology.classes_in_signature())[0].str
        if "#" in kb_namespace:
            self.kb_namespace = kb_namespace.split("#")[0] + "#"
        elif "/" in kb_namespace:
            self.kb_namespace = kb_namespace[:kb_namespace.rfind("/")] + "/"
        elif ":" in kb_namespace:
            self.kb_namespace = kb_namespace[:kb_namespace.rfind(":")] + ":"
        else:
            self.kb_namespace = kb_namespace
        self.expression_parser = DLSyntaxParser(self.kb_namespace)
        AutoConfig.register("cnir", CNIRConfig)

        for i in range(len(model_list)):
            model = model_list[i]
            if model.lower() == "composite":
                AutoModel.register(CNIRConfig, CNIRComposite)
            elif model.lower() == "lstm":
                AutoModel.register(CNIRConfig, CNIRLSTM)
            elif model.lower() == "gru":
                AutoModel.register(CNIRConfig, CNIRGRU)
            elif model.lower() == "transformer":
                AutoModel.register(CNIRConfig, CNIRTransformer)

            print("\n" + "\x1b[0;30;43m" + "Loading Model..." + "\x1b[0m" + "\n")
            self.models.append(AutoModel.from_pretrained(model_paths[i])) #, device_map="auto")
            if model.lower() != "composite":
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        super().__init__(path=path,
                         reasoner_factory=reasoner_factory,
                         ontology=ontology,
                         reasoner=reasoner,
                         class_hierarchy=class_hierarchy,
                         load_class_hierarchy=load_class_hierarchy,
                         object_property_hierarchy=object_property_hierarchy,
                         data_property_hierarchy=data_property_hierarchy,
                         include_implicit_individuals=include_implicit_individuals)

    def cnir_composite_predict(self, expr, model):
        start_time = time.time()
        if isinstance(expr, str):
            expr = [expr]

        data = InferenceDataset(
            data=expr,
            all_individuals=self.all_individuals_set,
            concept_to_instance_set=None,
            embeddings=self.embeddings
        )

        expr, component_embeddings_dict = next(iter(data))
        component_embeddings_dict = [
            {key: val.to(self.device) for key, val in component_embeddings_dict.items()}
        ]

        outputs = score_all_inds_composite(
            model, [expr], self.all_ind_embs, component_embeddings_dict
        ).squeeze()

        #retrieved = set(self.all_individuals_arr[np.where(outputs > self.th)[0]])
        time_taken = time.time() - start_time
        return outputs, time_taken

    def cnir_encoders_predict(self, expr, model):
        start_time = time.time()
        if isinstance(expr, str):
            expr = [expr]
        outputs = score_all_inds(model, self.tokenizer, self.all_ind_embs, expr,
                                 hidden_size=self.all_ind_embs.shape[1],
                                 chunk_size=self.chunksize).squeeze()
        #retrieved = set(self.all_individuals_arr[np.where(outputs > self.th)[0]])
        time_taken = time.time() - start_time
        return outputs, time_taken

    def cnir_predict(self, expr):
        result_dict = {}
        time_taken_dict = {}
        for model in self.model_list:
            pretrained_model = self.models[self.model_list.index(model)]
            if model.lower() == "composite":
                result_dict[model], time_taken_dict[model] = self.cnir_composite_predict(expr,
                                                                                         pretrained_model)
            else:
                result_dict[model], time_taken_dict[model] = self.cnir_encoders_predict(expr,
                                                                                        pretrained_model)
        # ensemble the models by taking mean of the values in result_dict
        mean_score = np.mean(list(result_dict.values()), axis=0)
        retrieved = set(self.all_individuals_arr[np.where(mean_score > self.th)[0]])
        return retrieved, time_taken_dict

    def individuals(self, concept=None, named_individuals=False):
        if concept:
            expr = concept if isinstance(concept, str) else self.dls_renderer.render(concept)
            individuals, _ = self.cnir_predict(expr)
            # represent the individuals in kb namespace format
            namespace = self.kb_namespace.split("/", 3)
            namespace = "/".join(namespace[:3]) + "/"
            individuals = list(map(lambda x: OWLNamedIndividual(namespace + x), individuals))

            return frozenset(individuals)

        else:
            return frozenset(self.ontology.individuals_in_signature())

        return individuals

    def __repr__(self):
        properties_count = iter_count(self.ontology.object_properties_in_signature()) + iter_count(
            self.ontology.data_properties_in_signature())
        class_count = iter_count(self.ontology.classes_in_signature())
        individuals_count = self.individuals_count()

        return f'CNIRKB(path={repr(self.path)} [model={repr(self.model_name.upper())}] <{class_count} classes, {properties_count} properties, ' \
               f'{individuals_count} individuals)'


if __name__ == "__main__":
    data_dir = "cnir/datasets/animals"
    kb = KnowledgeBase(path=f'{data_dir}/kb/ontology.owl')
    path = data_dir
    reasoner_factory = None
    ontology = kb.ontology
    reasoner = None
    class_hierarchy = None
    load_class_hierarchy = True
    object_property_hierarchy = None
    data_property_hierarchy = None
    include_implicit_individuals = False
    model_list = ["composite", "lstm", "gru", "transformer"]
    model_paths = ["cnir/pretrained_model/cnir_pretrained/CNIR_Composite_animals",
                  "cnir/pretrained_model/cnir_pretrained_encoders/CNIR_LSTM_animals",
                   "cnir/pretrained_model/cnir_pretrained_encoders/CNIR_GRU_animals",
                   "cnir/pretrained_model/cnir_pretrained_encoders/CNIR_Transformer_animals"]
    use_pma = True
    pma_model_path = "cnir/pretrained_model/pma_pretrained/PMA_animals/model.pt"
    tokenizer_path = "cnir/pretrained_model/tokenizer_pretrained/Tokenizer_animals"
    """
    def __init__(self,
                 path: Optional[str] = None,
                 reasoner_factory: Optional[
                     Callable[[AbstractOWLOntology], AbstractOWLReasoner]] = None,
                 ontology: Optional[AbstractOWLOntology] = None,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 class_hierarchy: Optional[ClassHierarchy] = None,
                 load_class_hierarchy: bool = True,
                 object_property_hierarchy: Optional[ObjectPropertyHierarchy] = None,
                 data_property_hierarchy: Optional[DatatypePropertyHierarchy] = None,
                 include_implicit_individuals=False,
                 dataset_dir=None, model_list=None, model_paths=None, use_pma=True, pma_model_path=None,
                 tokenizer_path=None,
                 chunksize=1024, th=0.5):
    """
    cnir_kb = CNIRKB(
        path=path,
        reasoner_factory=reasoner_factory,
        ontology=ontology,
        reasoner=reasoner,
        class_hierarchy=class_hierarchy,
        load_class_hierarchy=load_class_hierarchy,
        object_property_hierarchy=object_property_hierarchy,
        data_property_hierarchy=data_property_hierarchy,
        include_implicit_individuals=include_implicit_individuals,
        dataset_dir=data_dir,
        model_list=model_list,
        model_paths=model_paths,
        use_pma=use_pma,
        pma_model_path=pma_model_path,
        tokenizer_path=tokenizer_path
    )
    # kb = cnir_kb.kb
    concept = "Ostrich ⊔ Trout ⊔ (¬HasEggs)"
    # convert str to owl class expr "Ostrich ⊔ Trout ⊔ (¬HasEggs)"
    concept = cnir_kb.expression_parser.parse(concept)
    print(cnir_kb.individuals(concept=concept))
    print(cnir_kb.individuals_count(concept=concept))
