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

"""NCES: Neural Class Expression Synthesis."""

import os
import json
import glob
import subprocess
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
from ontolearn.data_struct import NCESDatasetInference
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.nces_architectures import LSTM, GRU, SetTransformer
from ontolearn.nces_trainer import NCESTrainer, before_pad
from ontolearn.nces_utils import SimpleSolution, generate_training_data
from ontolearn.search import NCESNode
from ontolearn.utils import read_csv
from ontolearn.utils.static_funcs import init_length_metric, compute_tp_fn_fp_tn


class NCES(BaseNCES):
    """Neural Class Expression Synthesis."""

    name = "NCES"

    def __init__(self, knowledge_base, nces2_or_roces=False,
                 quality_func: Optional[AbstractScorer] = None, num_predictions=5,
                 learner_names=["SetTransformer", "LSTM", "GRU"], path_of_embeddings=None, path_temp_embeddings=None,
                 path_of_trained_models=None, auto_train=True, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4,
                 num_seeds=1, m=32, ln=False, dicee_model="DeCaL", dicee_epochs=5, dicee_lr=0.01, dicee_emb_dim=128,
                 learning_rate=1e-4, tmax=20, eta_min=1e-5, clip_value=5.0, batch_size=256, num_workers=4,
                 max_length=48, load_pretrained=True, sorted_examples=False, verbose: int = 0, 
                 enforce_validity:Optional[bool]=None):
        self.knowledge_base = knowledge_base
        super().__init__(knowledge_base=knowledge_base, nces2_or_roces=nces2_or_roces,
                         quality_func=quality_func, num_predictions=num_predictions, auto_train=auto_train,
                         proj_dim=proj_dim, drop_prob=drop_prob, num_heads=num_heads, num_seeds=num_seeds,
                         m=m, ln=ln, learning_rate=learning_rate, tmax=tmax, eta_min=eta_min, clip_value=clip_value,
                         batch_size=batch_size, num_workers=num_workers, max_length=max_length,
                         load_pretrained=load_pretrained, verbose=verbose)

        self.learner_names = learner_names
        self.path_of_embeddings = path_of_embeddings
        self.path_temp_embeddings = path_temp_embeddings
        self.path_of_trained_models = path_of_trained_models
        self.dicee_model = dicee_model
        self.dicee_emb_dim = dicee_emb_dim
        self.dicee_epochs = dicee_epochs
        self.dicee_lr = dicee_lr
        self.rnn_n_layers = rnn_n_layers
        self.sorted_examples = sorted_examples
        self.has_renamed_inds = False
        self.enforce_validity = enforce_validity
        self._set_prerequisites()

    def _rename_individuals(self, individual_name):
        if isinstance(individual_name, str) and '/' in individual_name:
            return individual_name.split('/')[-1]
        return individual_name

    def _set_prerequisites(self):
        if self.path_of_embeddings is None or (os.path.isdir(self.path_of_embeddings) and not glob.glob(
                self.path_of_embeddings + '*_entity_embeddings.csv')) or not os.path.exists(
                self.path_of_embeddings) or not self.path_of_embeddings.endswith('.csv'):
            try:
                import dicee
                print('\nĆheck packages... OK: dicee is installed.')
                del dicee
            except Exception:
                print('\x1b[0;30;43m dicee is not installed, will first install it...\x1b[0m\n')
                subprocess.run('pip install dicee==0.1.4')
            if self.auto_train:
                print("\n"+"\x1b[0;30;43m"+"Embeddings not found. Will quickly train embeddings beforehand. "
                      +"Poor performance is expected as we will also train the synthesizer for a few epochs."
                       "\nFor maximum performance, use pretrained models or train embeddings for many epochs, "
                       "and the neural synthesizer on massive amounts of data and for many epochs. "
                       "See the example script in `examples/train_nces.py` for this. "
                       "Use `python examples/train_nces.py -h` to view options.\x1b[0m"+"\n")
            try:
                path_temp_embeddings = self.path_temp_embeddings if self.path_temp_embeddings and isinstance(
                    self.path_temp_embeddings, str) else "temp_embeddings"
                if not os.path.exists(path_temp_embeddings):
                    os.makedirs(path_temp_embeddings)
                path_temp_triples = "temp_embeddings/abox.nt"
                if os.path.exists(path_temp_triples):
                    os.remove(path_temp_triples)

                with open(path_temp_triples, "a") as f:
                    for s, p, o in self.knowledge_base.abox():
                        f.write(f"<{s.str}> <{p.str}> <{o.str}> .\n")

                self.knowledge_base_path = path_temp_triples

                subprocess.run(f"dicee --path_single_kg {self.knowledge_base_path} "
                               f"--path_to_store_single_run {path_temp_embeddings} "
                               f"--backend rdflib --save_embeddings_as_csv "
                               f"--num_epochs {self.dicee_epochs} "
                               f"--lr {self.dicee_lr} "
                               f"--model {self.dicee_model} "
                               f"--embedding_dim {self.dicee_emb_dim} "
                               f"--eval_mode test",
                               shell=True)
                assert os.path.exists(f"{path_temp_embeddings}/{self.dicee_model}_entity_embeddings.csv"), \
                    (f"It seems that embeddings were not stored at the expected directory "
                     f"({path_temp_embeddings}/{self.dicee_model}_entity_embeddings.csv)")
            except Exception as e:
                print(f"Error while training embeddings: {e}")
            self.path_of_embeddings = f"{path_temp_embeddings}/{self.dicee_model}_entity_embeddings.csv"
            if self.auto_train:
                print("\n"+"\x1b[0;30;43m"+f"Will also train {self.name} for 5 epochs"+"\x1b[0m"+"\n")
            self.instance_embeddings = read_csv(self.path_of_embeddings)
            self.input_size = self.instance_embeddings.shape[1]
            self.model = self.get_synthesizer(self.path_of_trained_models)
            print(f"\nUsing embeddings at: {self.path_of_embeddings} with {self.input_size} dimensions.\n")
            if self.auto_train:
                # Train NCES for 5 epochs
                self.train(epochs=5,num_workers = self.num_workers)
                self.refresh(self.path_of_trained_models)
        else:
            self.instance_embeddings = read_csv(self.path_of_embeddings)
            self.input_size = self.instance_embeddings.shape[1]
            self.model = self.get_synthesizer(self.path_of_trained_models)

    def get_synthesizer(self, path=None):
        if self.load_pretrained and path and glob.glob(path + "/*.pt"):
            try:
                with open(f"{path}/config.json") as f:
                    config = json.load(f)
                with open(f"{path}/vocab.json") as f:
                    vocab = json.load(f)
                inv_vocab = np.load(f"{path}/inv_vocab.npy", allow_pickle=True)
                self.max_length = config["max_length"]
                self.proj_dim = config["proj_dim"]
                self.num_heads = config["num_heads"]
                self.num_seeds = config["num_seeds"]
                self.rnn_n_layers = config["rnn_n_layers"]
                self.vocab = vocab
                self.inv_vocab = inv_vocab
            except Exception as e:
                print(e,'\n')
                raise FileNotFoundError(f"{path} does not contain at least one of `vocab.json, inv_vocab.npy "
                                        f"or embedding_config.json`")
        elif self.load_pretrained and self.path_of_trained_models and glob.glob(self.path_of_trained_models + "/*.pt"):
            try:
                with open(f"{path}/config.json") as f:
                    config = json.load(f)
                with open(f"{path}/vocab.json") as f:
                    vocab = json.load(f)
                inv_vocab = np.load(f"{path}/inv_vocab.npy", allow_pickle=True)
                self.max_length = config["max_length"]
                self.proj_dim = config["proj_dim"]
                self.num_heads = config["num_heads"]
                self.num_seeds = config["num_seeds"]
                self.rnn_n_layers = config["rnn_n_layers"]
                self.vocab = vocab
                self.inv_vocab = inv_vocab
            except Exception:
                raise FileNotFoundError(f"{self.path_of_trained_models} does not contain at least one of `vocab.json, "
                                        f"inv_vocab.npy or embedding_config.json`")

        m1 = SetTransformer(self.vocab, self.inv_vocab, self.max_length,
                                   self.input_size, self.proj_dim, self.num_heads, self.num_seeds, self.m,
                                   self.ln)
        m2 = GRU(self.vocab, self.inv_vocab, self.max_length, self.input_size,
                        self.proj_dim, self.rnn_n_layers, self.drop_prob)

        m3 = LSTM(self.vocab, self.inv_vocab, self.max_length, self.input_size,
                         self.proj_dim, self.rnn_n_layers, self.drop_prob)
        Models = {"SetTransformer": {"emb_model": None, "model": m1},
                     "GRU": {"emb_model": None, "model": m2},
                     "LSTM": {"emb_model": None, "model": m3}
                    }
        models_to_remove = []
        for name in Models:
            if name not in self.learner_names:
                models_to_remove.append(name)
        for name in models_to_remove:
            del Models[name]

        if self.load_pretrained and path is None:
            print("\x1b[0;30;43mThe path to pretrained models is None and load_pretrained is True. "
                  "Will return models with random weights.\x1b[0m")
            return Models
        elif self.load_pretrained and path and glob.glob(path+"/*.pt"):
            num_loaded_models = 0
            loaded_model_names = []
            for file_name in glob.glob(path+"/*.pt"):
                for model_name in Models:
                    if model_name in file_name:
                        try:
                            model = Models[model_name]["model"]
                            model.load_state_dict(torch.load(file_name, map_location=self.device, weights_only=True))
                            Models[model_name]["model"] = model
                            num_loaded_models += 1
                            loaded_model_names.append(model_name)
                        except Exception as e:
                            print(f"Could not load pretrained weights for {model_name}. "
                                  f"Please consider training the model!")
                            print("\n", e)
                            pass
            if num_loaded_models == len(Models):
                print("\n Loaded NCES weights!\n")
                return Models
            elif num_loaded_models > 0:
                print("\n"+"\x1b[0;30;43m"+f"Some model weights could not be loaded. "
                                           f"Successful ones are: {loaded_model_names}"+"\x1b[0m"+"\n")
                return Models
            else:
                print("\n"+"\x1b[0;30;43m"+"!!!No pretrained weights were provided, "
                                           "initializing models with random weights"+"\x1b[0m"+"\n")
                return Models
        else:
            print("\nNo pretrained weights were provided, initializing models with random weights.\n")
            return Models

    def refresh(self, path=None):
        if path is not None:
            self.load_pretrained = True
        self.model = self.get_synthesizer(path)

    def get_prediction(self, x_pos, x_neg):
        models = [self.model[name]["model"] for name in self.model]
        for i, model in enumerate(models):
            model.eval()
            model.to(self.device)
            x_pos = x_pos.to(self.device)
            x_neg = x_neg.to(self.device)
            if i == 0:
                _, scores = model(x_pos, x_neg)
            else:
                _, sc = model(x_pos, x_neg)
                scores = scores + sc

        scores = scores / len(models)
        prediction = model.inv_vocab[scores.argmax(1).cpu()]
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
        Pos = np.random.choice(pos_str, size=(self.num_predictions, len(pos_str)), replace=True).tolist()
        Neg = np.random.choice(neg_str, size=(self.num_predictions, len(neg_str)), replace=True).tolist()

        dataset = NCESDatasetInference([("", Pos_str, Neg_str) for (Pos_str, Neg_str) in zip(Pos, Neg)],
                                       self.instance_embeddings, self.num_examples, self.vocab, self.inv_vocab,
                                       shuffle_examples=False, max_length=self.max_length,
                                       sorted_examples=self.sorted_examples)

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                collate_fn=self.collate_batch_inference, shuffle=False)
        x_pos, x_neg = next(iter(dataloader))
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions_raw = self.get_prediction(x_pos, x_neg)

        if self.enforce_validity:
            concept_ast_builder = ConceptAbstractSyntaxTreeBuilder(knowledge_base=self.knowledge_base)

        predictions = []
        for prediction in predictions_raw:
            prediction_str = "".join(before_pad(prediction.squeeze()))
            try:
                concept = self.dl_parser.parse(prediction_str)
            except Exception:
                concept = simple_strategy(simpleSolution, prediction_str)
                if self.enforce_validity:
                    try:
                        raw_prediction = [pred for pred in prediction if pred != 'PAD']
                        parse_concept_str, _ = concept_ast_builder.parse(token_sequence=raw_prediction, enforce_validity=True)

                        concept = self.dl_parser.parse(parse_concept_str)
                    except Exception:
                        pass
                elif self.verbose>0:
                    print("Prediction: ", prediction_str)
            predictions.append(concept)
        return predictions

    def fit(self, learning_problem: PosNegLPStandard, **kwargs):
        for model_name in self.model:
            self.model[model_name]["model"].eval()
            self.model[model_name]["model"].to(self.device)

        pos = learning_problem.pos
        neg = learning_problem.neg
        if isinstance(pos, set) or isinstance(pos, frozenset):
            pos_list = list(pos)
            neg_list = list(neg)
            if "/" not in pos_list[0].str and not self.has_renamed_inds:
                self.instance_embeddings.index = self.instance_embeddings.index.map(self._rename_individuals)
                self.has_renamed_inds = True
            if self.sorted_examples:
                pos_list = sorted(pos_list)
                neg_list = sorted(neg_list)
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
                                                                                OWLNamedIndividual) else set(
                [ind.str.split("/")[-1] for ind in self.kb.individuals(concept)])
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
            print("NCES needs to be fitted to a problem first")
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
        if self.sorted_examples:
            pos_str, neg_str = sorted(pos_str), sorted(neg_str)
        return (target_concept_str, pos_str, neg_str)

    def fit_from_iterable(self, dataset: Union[List[Tuple[str, Set[OWLNamedIndividual], Set[OWLNamedIndividual]]],
    List[Tuple[str, Set[str], Set[str]]]], shuffle_examples=False,
                          verbose=False, **kwargs) -> List:
        """
        - Dataset is a list of tuples where the first items are strings corresponding to target concepts.
        
        - This function returns predictions as owl class expressions, not nodes as in fit
        """
        dataset = [self.convert_to_list_str_from_iterable(datapoint) for datapoint in dataset]
        dataset = NCESDatasetInference(dataset, self.instance_embeddings, self.num_examples, self.vocab, self.inv_vocab,
                                       shuffle_examples, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                collate_fn=self.collate_batch_inference, shuffle=False)
        simpleSolution = SimpleSolution(list(self.vocab), self.atomic_concept_names)
        predictions_as_owl_class_expressions = []
        predictions_str = []
        for x_pos, x_neg in dataloader:
            predictions = self.get_prediction(x_pos, x_neg)
            per_lp_preds = []
            for prediction in predictions:
                try:
                    prediction_str = "".join(before_pad(prediction))
                    ce = self.dl_parser.parse(prediction_str)
                    predictions_str.append(prediction_str)
                except Exception:
                    prediction_str = simpleSolution.predict("".join(before_pad(prediction)))
                    predictions_str.append(prediction_str)
                    ce = self.dl_parser.parse(prediction_str)
                per_lp_preds.append(ce)
            predictions_as_owl_class_expressions.append(per_lp_preds)
            if verbose:
                print("Predictions: ", predictions_str)
        return predictions_as_owl_class_expressions

    def train(self, data: Iterable[List[Tuple]]=None, epochs=50, batch_size=64, max_num_lps=1000,
              refinement_expressivity=0.2, refs_sample_size=50, learning_rate=1e-4, tmax=20, eta_min=1e-5,
              clip_value=5.0, num_workers=8, save_model=True, storage_path=None, optimizer='Adam', record_runtime=True,
              example_sizes=None, shuffle_examples=False):
        if os.cpu_count() <= num_workers:
            num_workers = max(0,os.cpu_count()-1)
        if storage_path is None:
            currentDateAndTime = datetime.now()
            storage_path = f'NCES-Experiment-{currentDateAndTime.strftime("%H-%M-%S")}'
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        self.path_of_trained_models = storage_path+"/trained_models"
        if batch_size is None:
            batch_size = self.batch_size
        if data is None:
            data = generate_training_data(kb_path=None,kb=self.knowledge_base, max_num_lps=max_num_lps,
                                          refinement_expressivity=refinement_expressivity, beyond_alc=False,
                                          refs_sample_size=refs_sample_size, storage_path=storage_path)
        example_ind = data[0][-1]["positive examples"][0]
        if "/" not in example_ind and not self.has_renamed_inds:
            self.instance_embeddings.index = self.instance_embeddings.index.map(self._rename_individuals)
            self.has_renamed_inds = True
        trainer = NCESTrainer(self, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, tmax=tmax,
                              eta_min=eta_min, clip_value=clip_value, num_workers=num_workers,
                              storage_path=storage_path)
        trainer.train(data=data, save_model=save_model, optimizer=optimizer, record_runtime=record_runtime)
