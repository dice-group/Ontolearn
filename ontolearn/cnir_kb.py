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

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from owlapy.render import DLSyntaxObjectRenderer
from owlapy.parser import DLSyntaxParser

from cnir import InferenceDataset
from cnir.config import CNIRConfig
from cnir.models import CNIRComposite, CNIRTransformer, CNIRLSTM, CNIRGRU
from cnir.models.pmanet import PMAnet
from cnir.utils import str2bool, read_embs_and_apply_agg
from cnir.utils import score_all_inds, score_all_inds_composite


class CNIRKB(KnowledgeBase):
    def __init__(self, dataset_dir, model, model_path, use_pma, pma_model_path, tokenizer_path=None,
                 chunksize = 1024, th=0.5):
        self.th = th
        self.pma_net = None
        self.model_name = model
        self.chunksize = chunksize

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
        self.all_ind_embs = torch.FloatTensor(self.embeddings.loc[self.all_individuals_arr].values)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        AutoConfig.register("cnir", CNIRConfig)
        if model.lower() == "composite":
            AutoModel.register(CNIRConfig, CNIRComposite)
        elif model.lower() == "lstm":
            AutoModel.register(CNIRConfig, CNIRLSTM)
        elif model.lower() == "gru":
            AutoModel.register(CNIRConfig, CNIRGRU)
        elif model.lower() == "transformer":
            AutoModel.register(CNIRConfig, CNIRTransformer)
        self.model = AutoModel.from_pretrained(model_path)
        if model.lower() != "composite":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        AutoConfig.register("cnir", CNIRConfig)
        AutoModel.register(CNIRConfig, CNIRComposite)

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
    def individuals(self, concept):
        expr = concept if isinstance(concept, str) else self.dls_renderer.render(concept)
        if self.model_name.lower() == "composite":
            individuals, _ = self.cnir_predict(expr)
        else:
            individuals, _ = self.cnir_encoders_predict(expr)
        # represent the individuals in kb namespace format
        namespace = self.kb_namespace.split("/", 3)
        namespace = "/".join(namespace[:3]) + "/"
        individuals = {namespace+i for i in individuals}

        return individuals


if __name__ == "__main__":
    data_dir = "cnir/mutagenesis"
    model = "composite"
    model_path = "cnir/pretrained_model/composite"
    use_pma = True
    pma_model_path = "cnir/pretrained_model/pma.pt"
    th = 0.5
    cnir_kb = CNIRKB(data_dir, model, model_path, use_pma, pma_model_path, th)
    print(cnir_kb.individuals("Carbon-29"))
