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

from .cnir import InferenceDataset
from .cnir.config import CNIRConfig
from .cnir.models import CNIRComposite, CNIRTransformer, CNIRLSTM, CNIRGRU
from .cnir.models.pmanet import PMAnet
from .cnir.utils import str2bool, read_embs_and_apply_agg
from .cnir.utils import score_all_inds, score_all_inds_composite


class CNIRKB(KnowledgeBase):
    def __init__(self, 
                 path: Optional[str] = None,
                 reasoner_factory: Optional[Callable[[AbstractOWLOntology], AbstractOWLReasoner]] = None,
                 ontology: Optional[AbstractOWLOntology] = None,
                 reasoner: Optional[AbstractOWLReasoner] = None,
                 class_hierarchy: Optional[ClassHierarchy] = None,
                 load_class_hierarchy: bool = True,
                 object_property_hierarchy: Optional[ObjectPropertyHierarchy] = None,
                 data_property_hierarchy: Optional[DatatypePropertyHierarchy] = None,
                 include_implicit_individuals=False,
                 dataset_dir=None, model=None, model_path=None, use_pma=True, pma_model_path=None, tokenizer_path=None,
                 chunksize = 1024, th=0.5):
        
        self.th = th
        self.pma_net = None
        self.model_name = model
        self.chunksize = chunksize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model.lower() == "composite" and use_pma:
            self.pma_net = PMAnet(CNIRConfig().embedding_dim, CNIRConfig().num_attention_heads, 1)
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
        self.all_ind_embs = torch.FloatTensor(self.embeddings.loc[self.all_individuals_arr].values).to(self.device)
        
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
        if model.lower() == "composite":
            AutoModel.register(CNIRConfig, CNIRComposite)
        elif model.lower() == "lstm":
            AutoModel.register(CNIRConfig, CNIRLSTM)
        elif model.lower() == "gru":
            AutoModel.register(CNIRConfig, CNIRGRU)
        elif model.lower() == "transformer":
            AutoModel.register(CNIRConfig, CNIRTransformer)

        print("\n"+"\x1b[0;30;43m"+"Loading Model..."+"\x1b[0m"+"\n")
        self.model = AutoModel.from_pretrained(model_path, device_map="auto")
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

    def cnir_predict(self, expr):
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
            self.model, [expr], self.all_ind_embs, component_embeddings_dict
        ).squeeze()

        retrieved = set(self.all_individuals_arr[np.where(outputs > self.th)[0]])
        time_taken = time.time() - start_time
        return retrieved, time_taken

    def cnir_encoders_predict(self, expr):
        start_time = time.time()
        if isinstance(expr, str):
            expr = [expr]
        outputs = score_all_inds(self.model, self.tokenizer, self.all_ind_embs, expr,
                                 hidden_size=self.all_ind_embs.shape[1],
                                 chunk_size= self.chunksize).squeeze()
        retrieved = set(self.all_individuals_arr[np.where(outputs > self.th)[0]])
        time_taken = time.time() - start_time
        return retrieved, time_taken
        
    def individuals(self, concept=None, named_individuals=False):
        if isinstance(concept, str):
            expr = concept
        elif concept is None:
            expr = '⊤'
        else:
            expr = self.dls_renderer.render(concept)
        try:
            if self.model_name.lower() == "composite":
                individuals, _ = self.cnir_predict(expr)
            else:
                individuals, _ = self.cnir_encoders_predict(expr)
        except:
            individuals, _ = self.cnir_predict('⊤')
        # represent the individuals in kb namespace format
        namespace = self.kb_namespace.split("/", 3)
        namespace = "/".join(namespace[:3]) + "/"
        individuals = frozenset(list(map(lambda x: OWLNamedIndividual(namespace+x), individuals)))
        return individuals

    def __repr__(self):
        properties_count = iter_count(self.ontology.object_properties_in_signature()) + iter_count(
            self.ontology.data_properties_in_signature())
        class_count = iter_count(self.ontology.classes_in_signature())
        individuals_count = self.individuals_count()

        return f'CNIRKB(path={repr(self.path)} [model={repr(self.model_name.upper())}] <{class_count} classes, {properties_count} properties, ' \
               f'{individuals_count} individuals)'