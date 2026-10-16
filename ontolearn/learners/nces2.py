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

"""NCES2: Neural Class Expression Synthesis in ALCHIQ(D)."""

import os
import json
import glob
import numpy as np
import torch
from datetime import datetime
from typing import List, Tuple, Iterable, Optional, Union, Set

from torch.utils.data import DataLoader
from owlapy.class_expression import OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.abstracts import AbstractScorer, AbstractNode
from ontolearn.base_nces import BaseNCES
from ontolearn.concept_abstract_syntax_tree import ConceptAbstractSyntaxTreeBuilder
from ontolearn.data_struct import TriplesData, ROCESDatasetInference
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.nces_modules import ConEx
from ontolearn.nces_architectures import SetTransformer
from ontolearn.nces_trainer import NCESTrainer, before_pad
from ontolearn.nces_utils import SimpleSolution, generate_training_data
from ontolearn.search import NCESNode
from ontolearn.utils.static_funcs import init_length_metric, compute_tp_fn_fp_tn


class NCES2(BaseNCES):
    """Neural Class Expression Synthesis in ALCHIQ(D)."""
    name = "NCES2"

    def __init__(self, knowledge_base, nces2_or_roces=True,
                 quality_func: Optional[AbstractScorer] = None, num_predictions=5,
                 path_of_trained_models=None, auto_train=True, proj_dim=128, drop_prob=0.1,
                 num_heads=4, num_seeds=1, m=[32, 64, 128], ln=False, embedding_dim=128, sampling_strategy="nces2",
                 input_dropout=0.0, feature_map_dropout=0.1, kernel_size=4, num_of_output_channels=32,
                 learning_rate=1e-4, tmax=20, eta_min=1e-5, clip_value=5.0, batch_size=256, num_workers=4,
                 max_length=48, load_pretrained=True, verbose: int = 0, data=[], enforce_validity:Optional[bool]=None):
        self.knowledge_base = knowledge_base
        super().__init__(knowledge_base, nces2_or_roces, quality_func, num_predictions, auto_train, proj_dim,
                         drop_prob, num_heads, num_seeds, m, ln, learning_rate, tmax, eta_min, clip_value, batch_size,
                         num_workers, max_length, load_pretrained, verbose)

        # Use a separate directory for triples to avoid deletion
        temp_triples_dir = os.path.abspath("temp_triples")
        if not os.path.exists(temp_triples_dir):
            os.makedirs(temp_triples_dir)
        path_temp_triples = os.path.join(temp_triples_dir, "abox.nt")
        with open(path_temp_triples, "w") as f:
            for s, p, o in self.knowledge_base.abox():
                f.write(f"<{s.str}> <{p.str}> <{o.str}> .\n")

        self.knowledge_base_path = path_temp_triples
        self.triples_data = TriplesData(self.knowledge_base_path)
        self.num_entities = len(self.triples_data.entity2idx)
        self.num_relations = len(self.triples_data.relation2idx)
        self.path_of_trained_models = path_of_trained_models
        self.embedding_dim = embedding_dim
        self.sampling_strategy = sampling_strategy
        self.input_dropout = input_dropout
        self.feature_map_dropout = feature_map_dropout
        self.kernel_size = kernel_size
        self.num_of_output_channels = num_of_output_channels
        self.num_workers = num_workers
        self.enforce_validity = enforce_validity
        self._set_prerequisites()

    def _set_prerequisites(self):
        if isinstance(self.m, int):
            self.m = [self.m]

        if self.load_pretrained and self.path_of_trained_models is None and self.auto_train:
            print(f"\n\x1b[0;30;43mPath to pretrained models is None and load_pretrained is True "
                  f"and auto_train is True. Will quickly train neural synthesizers. "
                  f"However, it is advisable that you properly train {self.name} using the "
                  f"example script in `examples/train_nces.py`.\x1b[0m\n")
            # num_workers=0: see the comment on the equivalent call in
            # NCES._set_prerequisites (nces.py) - same quick-training,
            # coverage-run-deadlock-risk rationale (#610/#622).
            self.train(epochs=5, num_workers=0)
            self.refresh(self.path_of_trained_models)
        else:
            self.model = self.get_synthesizer(self.path_of_trained_models)

    def get_synthesizer(self, path=None, verbose=True):
        if self.load_pretrained and path and glob.glob(path + "/*.pt"):
            try:
                with open(f"{path}/config.json") as f:
                    config = json.load(f)
                with open(f"{path}/vocab.json") as f:
                    vocab = json.load(f)
                inv_vocab = np.load(f"{path}/inv_vocab.npy", allow_pickle=True)
                with open(f"{path}/embedding_config.json") as f:
                    emb_config = json.load(f)
                self.max_length = config["max_length"]
                self.proj_dim = config["proj_dim"]
                self.num_heads = config["num_heads"]
                self.num_seeds = config["num_seeds"]
                self.vocab = vocab
                self.inv_vocab = inv_vocab
                self.embedding_dim = emb_config["embedding_dim"]
                self.num_entities = emb_config["num_entities"]
                self.num_relations = emb_config["num_relations"]
            except Exception as e:
                raise FileNotFoundError(f"{path} does not contain at least one of "
                                        f"`vocab.json, inv_vocab.npy or embedding_config.json`") from e
        elif self.load_pretrained and self.path_of_trained_models and glob.glob(self.path_of_trained_models + "/*.pt"):
            try:
                with open(f"{path}/config.json") as f:
                    config = json.load(f)
                with open(f"{path}/vocab.json") as f:
                    vocab = json.load(f)
                inv_vocab = np.load(f"{path}/inv_vocab.npy", allow_pickle=True)
                with open(f"{path}/embedding_config.json") as f:
                    emb_config = json.load(f)
                self.max_length = config["max_length"]
                self.proj_dim = config["proj_dim"]
                self.num_heads = config["num_heads"]
                self.num_seeds = config["num_seeds"]
                self.vocab = vocab
                self.inv_vocab = inv_vocab
                self.embedding_dim = emb_config["embedding_dim"]
                self.num_entities = emb_config["num_entities"]
                self.num_relations = emb_config["num_relations"]
            except Exception as e:
                raise FileNotFoundError(f"{self.path_of_trained_models} does not contain at least one of "
                                        f"`vocab.json, inv_vocab.npy or embedding_config.json`") from e

        Models = {str(m): {"emb_model": ConEx(self.embedding_dim, self.num_entities, self.num_relations,
                                              self.input_dropout, self.feature_map_dropout, self.kernel_size,
                                              self.num_of_output_channels),
                           "model": SetTransformer(self.vocab, self.inv_vocab,
                                                   self.max_length, self.embedding_dim, self.proj_dim, self.num_heads,
                                                   self.num_seeds, m, self.ln)} for m in self.m}

        if self.load_pretrained and path is None:
            print("\n\x1b[0;30;43mPath to pretrained models is None and load_pretrained is True. "
                  "Will return models with random weights.\x1b[0m\n")
            return Models

        elif self.load_pretrained and path and len(glob.glob(path + "/*.pt")) == 0:
            print("\n"+"\x1b[0;30;43m"+f"No pretrained model found! If {self.path_of_trained_models} "
                                       f"is empty or does not exist, set the `load_pretrained` parameter to `False` or "
                                       f"make sure `save_model` was set to `True` in the .train() "
                                       f"method."+"\x1b[0m"+"\n")
            raise FileNotFoundError(f"Path {path} does not contain any pretrained models!")

        elif self.load_pretrained and path and glob.glob(path + "/*.pt"):
            possible_checkpoints = glob.glob(path + "/*.pt")
            num_loaded_models = 0
            loaded_model_names = []
            for file_name in possible_checkpoints:
                for m in self.m:
                    if str(m) in file_name:
                        if "emb" not in file_name:
                            weights = torch.load(file_name, map_location=self.device, weights_only=True)
                            model = Models[str(m)]["model"]
                            model.load_state_dict(weights)
                            Models[str(m)]["model"] = model
                            num_loaded_models += 1
                            loaded_model_names.append(f'SetTransformer ({m} inducing points)')
                        else:
                            weights = torch.load(file_name, map_location=self.device, weights_only=True)
                            emb_model = Models[str(m)]["emb_model"]
                            emb_model.load_state_dict(weights)
                            Models[str(m)]["emb_model"] = emb_model
            if num_loaded_models == len(self.m):
                print(f"\nLoaded {self.name} weights!\n")
                return Models
            elif num_loaded_models > 0:
                models_to_remove = []
                for name in Models:
                    if not any(name in loaded_model_name for loaded_model_name in loaded_model_names):
                        models_to_remove.append(name)
                for name in models_to_remove:
                    del Models[name]
                print("\x1b[0;30;43m"+f"!!!Some pretrained weights could not be found, successfully "
                                      f"loaded models are {loaded_model_names}"+"\x1b[0m"+"\n")
                return Models
            else:
                print("\x1b[0;30;43m"+"!!!No pretrained weights were found, initializing models "
                                      "with random weights"+"\x1b[0m"+"\n")
                return Models
        else:
            if verbose:
                print(f"\nNo pretrained weights were provided, initializing models with random weights. "
                      f"You may want to first train the synthesizer using {self.name}.train()\n")
            return Models

    def refresh(self, path=None):
        if path is not None:
            self.load_pretrained = True
        self.model = self.get_synthesizer(path)

    def get_prediction(self, dataloaders, return_normalize_scores=False):
        for i, (num_ind_points, dataloader) in enumerate(zip(self.m, dataloaders)):
            x_pos, x_neg = next(iter(dataloader))
            x_pos = x_pos.squeeze().to(self.device)
            x_neg = x_neg.squeeze().to(self.device)
            if i == 0:
                _, scores = self.model[str(num_ind_points)]["model"](x_pos, x_neg)
            else:
                _, sc = self.model[str(num_ind_points)]["model"](x_pos, x_neg)
                scores = scores + sc
        scores = scores / len(self.m)
        if return_normalize_scores:
            return scores
        prediction = self.inv_vocab[scores.argmax(1).cpu()]
        return prediction

    def fit_one(self, pos: Union[List[OWLNamedIndividual], List[str]], neg: Union[List[OWLNamedIndividual], List[str]]):
        def simple_strategy(strategy: SimpleSolution, prediction: List[str]):
            return self.dl_parser.parse(strategy.predict(prediction))
        
        if isinstance(pos[0], OWLNamedIndividual):
            pos_str = [ind.str.split("/")[-1] for ind in pos]
            neg_str = [ind.str.split("/")[-1] for ind in neg]
        elif isinstance(pos[0], str):
            pos_str = pos
            neg_str = neg
        else:
            raise ValueError(f"Invalid input type, was expecting OWLNamedIndividual or str but found {type(pos[0])}")

        # dataloader objects
        dataloaders = []
        for num_ind_points in self.model:
            dataset = ROCESDatasetInference([("", pos_str, neg_str)],
                                              triples_data=self.triples_data, num_examples=self.num_examples,
                                              k=self.k if hasattr(self, "k") else None,
                                              vocab=self.vocab, inv_vocab=self.inv_vocab,
                                              max_length=self.max_length,
                                              sampling_strategy=self.sampling_strategy,
                                              num_pred_per_lp=self.num_predictions)
            dataset.load_embeddings(self.model[num_ind_points]["emb_model"])
            # num_workers=0: this builds one inference-only DataLoader per
            # architecture in self.model, in a loop - the same fresh-worker-
            # pool-per-architecture pattern already found to deadlock under
            # `coverage run` in the training path (#610). See #622.
            dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                num_workers=0, shuffle=False)
            dataloaders.append(dataloader)

        # Initialize a simple solution constructor
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions = []
        predictions_raw = self.get_prediction(dataloaders)

        if self.enforce_validity:
            concept_ast_builder = ConceptAbstractSyntaxTreeBuilder(knowledge_base=self.knowledge_base)

        for prediction in predictions_raw:
            prediction_str = "".join(before_pad(prediction.squeeze()))
            try:
                concept = self.dl_parser.parse(prediction_str)
            except Exception:
                # prediction_str is raw, unconstrained model output, not a validated DL
                # expression, so the parser can legitimately fail in many different ways -
                # this is the intended "fall back to token-level reconstruction" path, not
                # a bug being swallowed.
                concept = simple_strategy(simpleSolution, prediction_str)
                if self.enforce_validity:
                    try:
                        raw_prediction = [pred for pred in prediction if pred != 'PAD']
                        parse_concept_str, _ = concept_ast_builder.parse(token_sequence=raw_prediction, enforce_validity=True)

                        concept = self.dl_parser.parse(parse_concept_str)
                    except Exception:
                        # `concept` already holds the simple_strategy() fallback from above,
                        # so there is nothing to undo if the stricter AST-validity rebuild
                        # also fails - we just keep that fallback instead of this one.
                        pass
                elif self.verbose>0:
                    print("Prediction: ", prediction_str)
            predictions.append(concept)
        return predictions

    def fit(self, learning_problem: PosNegLPStandard, **kwargs):
        # Set models in evaluation mode
        for num_ind_points in self.model:
            for model_type in self.model[num_ind_points]:
                self.model[num_ind_points][model_type].eval()
                self.model[num_ind_points][model_type].to(self.device)

        pos = learning_problem.pos
        neg = learning_problem.neg
        if isinstance(pos, set) or isinstance(pos, frozenset):
            pos_list = list(pos)
            neg_list = list(neg)
        else:
            raise ValueError(f"Expected pos and neg to be sets, got {type(pos)} and {type(neg)}")
        predictions = self.fit_one(pos_list, neg_list)

        predictions_as_nodes = []
        for concept in predictions:
            try:
                concept_individuals_count = self.kb.individuals_count(concept)
            except AttributeError:
                concept = self.dl_parser.parse('⊤')
                concept_individuals_count = self.kb.individuals_count(concept)
            concept_length = init_length_metric().length(concept)
            concept_instances = set(self.kb.individuals(concept)) if isinstance(pos_list[0],
                OWLNamedIndividual) else set([ind.str.split("/")[-1] for ind in self.kb.individuals(concept)])
            tp, fn, fp, tn = compute_tp_fn_fp_tn(concept_instances, pos, neg)
            quality = self.quality_func.score2(tp, fn, fp, tn)[1]
            node = NCESNode(concept, length=concept_length, individuals_count=concept_individuals_count,
                            quality=quality)
            predictions_as_nodes.append(node)
        predictions_as_nodes = sorted(predictions_as_nodes, key=lambda x: -x.quality)
        self.best_predictions = predictions_as_nodes
        return self

    def best_hypotheses(self, n=1, return_node: bool = False) \
            -> Union[OWLClassExpression, Iterable[OWLClassExpression],
                     AbstractNode, Iterable[AbstractNode], None]:
        if self.best_predictions is None:
            print(f"{self.name} needs to be fitted to a problem first")
            return None
        elif len(self.best_predictions) == 1 or n == 1:
            if return_node:
                return self.best_predictions[0]
            return self.best_predictions[0].concept
        else:
            if return_node:
                return self.best_predictions
            return [best.concept for best in self.best_predictions[:n]]

    def convert_to_list_str_from_iterable(self, data):
        target_concept_str, examples = data[0], data[1:]
        pos = list(examples[0])
        neg = list(examples[1])
        if isinstance(pos[0], OWLNamedIndividual):
            pos_str = [ind.str.split("/")[-1] for ind in pos]
            neg_str = [ind.str.split("/")[-1] for ind in neg]
        elif isinstance(pos[0], str):
            pos_str, neg_str = list(pos), list(neg)
        else:
            raise ValueError(f"Invalid input type, was expecting OWLNamedIndividual or str but found {type(pos[0])}")
        return (target_concept_str, pos_str, neg_str)

    def fit_from_iterable(self, data: Union[List[Tuple[str, Set[OWLNamedIndividual], Set[OWLNamedIndividual]]],
    List[Tuple[str, Set[str], Set[str]]]], shuffle_examples=False, verbose=False, **kwargs) -> List:
        """
        - data is a list of tuples where the first items are strings corresponding to target concepts.
        
        - This function returns predictions as owl class expressions, not nodes as in fit
        """
        data = [self.convert_to_list_str_from_iterable(datapoint) for datapoint in data]
        dataloaders = []
        for num_ind_points in self.model:
            dataset = ROCESDatasetInference(data,
                                            self.triples_data, num_examples=self.num_examples,
                                            k=self.k if hasattr(self, "k") else None,
                                            vocab=self.vocab, inv_vocab=self.inv_vocab,
                                            max_length=self.max_length,
                                            sampling_strategy=self.sampling_strategy,
                                            num_pred_per_lp=self.num_predictions)
            dataset.load_embeddings(self.model[num_ind_points]["emb_model"])
            # num_workers=0: inference-only path, see the comment in fit_one above (#622).
            dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
            dataloaders.append(dataloader)
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions_as_owl_class_expressions = []
        predictions_str = []
        for dataloader in dataloaders:
            predictions = self.get_prediction(dataloader)
            per_lp_preds = []
            for prediction in predictions:
                try:
                    prediction_str = "".join(before_pad(prediction))
                    ce = self.dl_parser.parse(prediction_str)
                    predictions_str.append(prediction_str)
                except Exception:
                    # See the equivalent fallback in fit_one() above: prediction_str is raw
                    # model output, not a validated DL expression, so this is the intended
                    # "fall back to token-level reconstruction" path.
                    prediction_str = simpleSolution.predict("".join(before_pad(prediction)))
                    predictions_str.append(prediction_str)
                    ce = self.dl_parser.parse(prediction_str)
                per_lp_preds.append(ce)
            predictions_as_owl_class_expressions.append(per_lp_preds)
            if verbose:
                print("Predictions: ", predictions_str)
        return predictions_as_owl_class_expressions

    def train(self, data: Iterable[List[Tuple]] = None, epochs=50, batch_size=64, max_num_lps=1000,
              refinement_expressivity=0.2, refs_sample_size=50, learning_rate=1e-4, tmax=20, eta_min=1e-5,
              clip_value=5.0, num_workers=8, save_model=True, storage_path=None, optimizer='Adam',
              record_runtime=True, shuffle_examples=False):
        if os.cpu_count() <= num_workers:
            num_workers = max(0,os.cpu_count()-1)
        if storage_path is None:
            currentDateAndTime = datetime.now()
            storage_path = f'{self.name}-Experiment-{currentDateAndTime.strftime("%H-%M-%S")}'
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        if batch_size is None:
            batch_size = self.batch_size
        if data is None:
            data = generate_training_data(kb_path=None,kb=self.knowledge_base, max_num_lps=max_num_lps,
                                          refinement_expressivity=refinement_expressivity, beyond_alc=True,
                                          refs_sample_size=refs_sample_size, storage_path=storage_path)
        vocab_size_before = len(self.vocab)
        self.add_data_values(data) # Add data values based on training data
        self.path_of_trained_models = storage_path+"/trained_models"
        if len(self.vocab) > vocab_size_before:
            self.model = self.get_synthesizer(verbose=False)
        print(num_workers)
        trainer = NCESTrainer(self, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, tmax=tmax,
                              eta_min=eta_min, clip_value=clip_value, num_workers=num_workers,
                              storage_path=storage_path)
        trainer.train(data=data, save_model=save_model, optimizer=optimizer, record_runtime=record_runtime)
