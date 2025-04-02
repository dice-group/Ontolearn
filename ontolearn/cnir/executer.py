import datetime
import glob
import os
import time

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
from owlapy.render import DLSyntaxObjectRenderer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedTokenizerFast
from transformers import set_seed

from cnir.config import CNIRConfig
from cnir.models import *
from cnir.utils import read_embs, read_training_data, read_test_data, timeit, read_embs_and_apply_agg, \
    read_and_prepare_pma_data
from .models.pmanet import PMAnet
import torch
import torch.optim as optim
from .trainer import Trainer, PMATrainer


class Execute:
    """ A class for Training and Evaluating a model.

    (1) Loading & Preprocessing & Serializing input data.
    (2) Training & Validation & Save model and results
    """

    def __init__(self, args):
        set_seed(args.random_seed)
        self.args = args
        self.read_and_maybe_preprocess_data()
        self.maybe_get_tokenizer()
        self.get_model()
        assert self.model.config.embedding_dim == self.embeddings.shape[1], f"Mismatch between embedding size and d_model: {self.embeddings.shape[1]} and {self.model.config.embedding_dim}"
        self.compute_instance_retrieval_cache()
    

    def read_and_maybe_preprocess_data(self) -> None:
        if not os.path.exists(self.args.dataset_dir):
            raise FileNotFoundError(f"Path to dataset does not exist: {self.args.dataset_dir}")
        if not os.path.exists(f"{self.args.dataset_dir}/kb/ontology.owl"):
            raise FileNotFoundError(
                f"Path to dataset does not exist: {self.args.dataset_dir}/kb/ontology.owl")
        if not glob.glob(f"{self.args.dataset_dir}/embeddings/*embeddings.csv"):
            raise FileNotFoundError(f"No embeddings found at {self.args.dataset_dir}/embeddings/")
        if not glob.glob(f"{self.args.dataset_dir}/training_data/*.json"):
            raise FileNotFoundError(
                f"No training data found at {self.args.dataset_dir}/training_data/")
        if self.args.model == "composite":
            if self.args.model == "composite" and self.args.use_pma:
                self.pma_net = PMAnet(CNIRConfig().embedding_dim, CNIRConfig().num_attention_heads, 1)
                self.pma_net.load_state_dict(torch.load(self.args.pma_model_path, map_location="cpu", weights_only=True))
            self.kb, self.all_individuals, self.embeddings = read_embs_and_apply_agg(
                self.args.dataset_dir, nn_agg=self.pma_net if hasattr(self, "pma_net") else None, merge=True)
        else:
            self.embeddings = read_embs(self.args.dataset_dir, merge=False)
        remove_atomic_concepts = self.args.model.lower() == "composite"
        self.data = read_training_data(self.args.dataset_dir,
                                       remove_atomic_concepts=remove_atomic_concepts)

    @timeit
    def maybe_get_tokenizer(self):
        if not hasattr(self, "kb"):
            self.kb = KnowledgeBase(path=f"{self.args.dataset_dir}/kb/ontology.owl")
        if not hasattr(self, "all_individuals"):
            all_individuals = [ind.str.split("/")[-1] for ind in self.kb.individuals()]
            valid_inds = [ind for ind in all_individuals if ind in self.embeddings.index]
            self.all_individuals = valid_inds

        if self.args.pretrained_tokenizer_path and self.args.model.lower() != "composite":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_tokenizer_path)
                self.can_use_pretrained_model = True
            except Exception as e:
                print(e)
                print("\nSkipping...")
                if not hasattr(self, "kb"):
                    self.kb = KnowledgeBase(path=f"{self.dataset_dir}/kb/ontology.owl")
                renderer = DLSyntaxObjectRenderer()
                atomic_concept_names = frozenset(
                    [renderer.render(a) for a in self.kb.ontology.classes_in_signature()])
                role_names = frozenset([r.str.split("/")[-1].split("#")[-1] for r in
                                        self.kb.ontology.object_properties_in_signature()] +
                                       [r.str.split("/")[-1].split("#")[-1] for r in
                                        self.kb.ontology.data_properties_in_signature()])
                Vocab = ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', ')', '(', '.', '>=', '<=', 'True',
                         'False', '[', ']', '{', '}', '⁻'] + \
                        list(atomic_concept_names) + list(role_names)
                tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
                trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
                tokenizer.pre_tokenizer = WhitespaceSplit()
                tokenizer.train_from_iterator(Vocab, trainer)
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
                tokenizer.pad_token = "[PAD]"
                self.tokenizer = tokenizer
                self.can_use_pretrained_model = False
        elif self.args.model.lower() != "composite":
            renderer = DLSyntaxObjectRenderer()
            atomic_concept_names = frozenset(
                [renderer.render(a) for a in self.kb.ontology.classes_in_signature()])
            role_names = frozenset([r.str.split("/")[-1].split("#")[-1] for r in
                                    self.kb.ontology.object_properties_in_signature()] +
                                   [r.str.split("/")[-1].split("#")[-1] for r in
                                    self.kb.ontology.data_properties_in_signature()])
            Vocab = ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', ')', '(', '.', '>=', '<=', 'True', 'False',
                     '[', ']', '{', '}', '⁻'] + \
                    list(atomic_concept_names) + list(role_names)
            tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.pre_tokenizer = WhitespaceSplit()
            tokenizer.train_from_iterator(Vocab, trainer)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            tokenizer.pad_token = "[PAD]"
            tokenizer.pad_token_id = 3
            self.tokenizer = tokenizer
            self.can_use_pretrained_model = False
        else:
            self.tokenizer = None
            self.can_use_pretrained_model = True

    @timeit
    def get_model(self):
        AutoConfig.register("cnir", CNIRConfig)
        if self.args.model.lower() == "transformer":
            AutoModel.register(CNIRConfig, CNIRTransformer)
        elif self.args.model.lower() == "composite":
            AutoModel.register(CNIRConfig, CNIRComposite)
        elif self.args.model.lower() in "lstm":
            AutoModel.register(CNIRConfig, CNIRLSTM)
        elif self.args.model.lower() in "gru":
            AutoModel.register(CNIRConfig, CNIRGRU)
        if self.args.pretrained_model_path and self.can_use_pretrained_model:
            try:
                self.model = AutoModel.from_pretrained(self.args.pretrained_model_path)
                print("\n\x1b[6;30;42mSuccessfully loaded a pretrained model!\x1b[0m\n")
            except Exception as e:
                print(e)
                print("\nSkipping...")
                model_config = CNIRConfig()
                for key, val in self.args.__dict__.items():
                    if getattr(model_config, key, None) is not None and val is not None:
                        model_config.__setattr__(key, val)
                if self.args.model.lower() == "transformer":
                    model_config.__setattr__("pad_token_id", self.tokenizer.pad_token_id)
                    self.model = CNIRTransformer(model_config)
                elif self.args.model.lower() == "lstm":
                    model_config.__setattr__("pad_token_id", self.tokenizer.pad_token_id)
                    self.model = CNIRLSTM(model_config)
                elif self.args.model.lower() == "gru":
                    model_config.__setattr__("pad_token_id", self.tokenizer.pad_token_id)
                    self.model = CNIRGRU(model_config)
                elif self.args.model.lower() == "composite":
                    self.model = CNIRComposite(model_config)
                else:
                    raise ValueError(
                        f"Unknow model name {self.args.model}. Allowed models are `transformer, lstm, gru, composite`")
        else:
            model_config = CNIRConfig()
            model_config.embedding_dim = self.embeddings.shape[1]
            try:
                model_config.pad_token_id = self.tokenizer.pad_token_id
                for key, val in self.args.__dict__.items():
                    if getattr(model_config, key, None) is not None and val is not None:
                        model_config.__setattr__(key, val)
                model_config.vocab_size = max(model_config.vocab_size, self.tokenizer.vocab_size)
            except Exception as e:
                if self.args.model == "composite":
                    print("Composite model does not use a tokenizer. Therefore, the tokenizer is `None`")
                print(e)

            if self.args.model.lower() == "transformer":
                self.model = CNIRTransformer(model_config)
            elif self.args.model.lower() == "lstm":
                self.model = CNIRLSTM(model_config)
            elif self.args.model.lower() == "gru":
                self.model = CNIRGRU(model_config)
            elif self.args.model.lower() == "composite":
                self.model = CNIRComposite(model_config)
            else:
                raise ValueError(
                    f"Unknow model name {self.args.model}. Allowed models are `transformer, lstm, gru, composite`")

    @timeit
    def compute_instance_retrieval_cache(self):
        kb_namespace = list(self.kb.ontology.classes_in_signature())[0].str
        if "#" in kb_namespace:
            kb_namespace = kb_namespace.split("#")[0] + "#"
        elif "/" in kb_namespace:
            kb_namespace = kb_namespace[:kb_namespace.rfind("/")] + "/"
        elif ":" in kb_namespace:
            kb_namespace = kb_namespace[:kb_namespace.rfind(":")] + ":"
        expression_parser = DLSyntaxParser(kb_namespace)
        self.concept_to_instance_set = {expr: set(
            [ind.str.split("/")[-1] for ind in self.kb.individuals(expression_parser.parse(expr))])
            for expr in self.data}

    def start(self) -> dict:
        self.start_time = time.time()
        print("\nStarting training...")
        print(f"Start time:{datetime.datetime.now()}\n")
        self.trainer = Trainer(self.model, self.tokenizer, self.data, self.embeddings,
                               self.all_individuals, self.concept_to_instance_set,
                               self.args.num_examples, self.args.th, self.args.optimizer, self.args.pretrained_model_path,
                               self.args.output_dir, self.args.epochs, self.args.batch_size,
                               self.args.num_workers, self.args.lr, self.args.train_test_split)
        self.trainer.train()


class ExecuteTest:
    """
    A class for testing a model.
    """
    def __init__(self, args):
        #set_seed(args.random_seed)
        self.args = args
        self.read_and_maybe_preprocess_data()
        self.maybe_get_tokenizer()
        self.get_model()
        self.compute_instance_retrieval_cache()
        self.num_examples = None
        self.epochs = None
        self.train_test_split = None
    def read_and_maybe_preprocess_data(self) -> None:
        if not os.path.exists(self.args.dataset_dir):
            raise FileNotFoundError(f"Path to dataset does not exist: {self.args.dataset_dir}")
        if not os.path.exists(f"{self.args.dataset_dir}/kb/ontology.owl"):
            raise FileNotFoundError(
                f"Path to dataset does not exist: {self.args.dataset_dir}/kb/ontology.owl")
        if not glob.glob(f"{self.args.dataset_dir}/embeddings/*embeddings.csv"):
            raise FileNotFoundError(f"No embeddings found at {self.args.dataset_dir}/embeddings/")
        if not glob.glob(f"{self.args.dataset_dir}/training_data/*.json"):
            raise FileNotFoundError(
                f"No training data found at {self.args.dataset_dir}/training_data/")
        if self.args.model == "composite":
            if self.args.model == "composite" and self.args.use_pma:
                self.pma_net = PMAnet(CNIRConfig().embedding_dim, CNIRConfig().num_attention_heads, 1)
                self.pma_net.load_state_dict(torch.load(self.args.pma_model_path, map_location="cpu", weights_only=True))
            self.kb, self.all_individuals, self.embeddings = read_embs_and_apply_agg(
                self.args.dataset_dir, nn_agg=self.pma_net if hasattr(self, "pma_net") else None, merge=True)
        else:
            self.embeddings = read_embs(self.args.dataset_dir, merge=False)
        #remove_atomic_concepts = self.args.model.lower() == "composite"
        self.data = read_test_data(self.args.dataset_dir)#,
                                       #remove_atomic_concepts=remove_atomic_concepts)
    def maybe_get_tokenizer(self):
        if not hasattr(self, "kb"):
            self.kb = KnowledgeBase(path=f"{self.args.dataset_dir}/kb/ontology.owl")
        if not hasattr(self, "all_individuals"):
            all_individuals = [ind.str.split("/")[-1] for ind in self.kb.individuals()]
            valid_inds = [ind for ind in all_individuals if ind in self.embeddings.index]
            self.all_individuals = valid_inds

        if self.args.pretrained_tokenizer_path and self.args.model.lower() != "composite":
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_tokenizer_path)
            self.can_use_pretrained_model = True

        else:
            self.tokenizer = None
            self.can_use_pretrained_model = True
    def get_model(self):
        AutoConfig.register("cnir", CNIRConfig)
        if self.args.model.lower() == "transformer":
            AutoModel.register(CNIRConfig, CNIRTransformer)
        elif self.args.model.lower() == "composite":
            AutoModel.register(CNIRConfig, CNIRComposite)
        elif self.args.model.lower() in "lstm":
            AutoModel.register(CNIRConfig, CNIRLSTM)
        elif self.args.model.lower() in "gru":
            AutoModel.register(CNIRConfig, CNIRGRU)
        try:
            self.model = AutoModel.from_pretrained(self.args.pretrained_model_path)
            print("\n\x1b[6;30;42mSuccessfully loaded a pretrained model!\x1b[0m\n")
        except Exception as e:
            print(e)
    def compute_instance_retrieval_cache(self):
        kb_namespace = list(self.kb.ontology.classes_in_signature())[0].str
        if "#" in kb_namespace:
            kb_namespace = kb_namespace.split("#")[0] + "#"
        elif "/" in kb_namespace:
            kb_namespace = kb_namespace[:kb_namespace.rfind("/")] + "/"
        elif ":" in kb_namespace:
            kb_namespace = kb_namespace[:kb_namespace.rfind(":")] + ":"
        expression_parser = DLSyntaxParser(kb_namespace)
        self.concept_to_instance_set = {expr: set(
            [ind.str.split("/")[-1] for ind in self.kb.individuals(expression_parser.parse(expr))])
            for expr in self.data}
    def start(self) -> dict:
        self.start_time = time.time()
        print("\nStarting testing...")
        print(f"Start time:{datetime.datetime.now()}\n")
        self.trainer = Trainer(self.model, self.tokenizer, self.data, self.embeddings,
                               self.all_individuals, self.concept_to_instance_set,
                               self.num_examples, self.args.th, self.args.optimizer, self.args.pretrained_model_path,
                               self.args.output_dir, self.epochs, self.args.batch_size,
                               self.args.num_workers, self.args.lr, self.train_test_split)
        f1, jaccard = self.trainer.predict(self.data)
        print(f"Test F1: {f1}", f"Test Jaccard: {jaccard}")

class ExecuteTestCNIR:
    """
    A class for testing a model.
    """
    def __init__(self, args):
        self.args = args
        self.read_and_maybe_preprocess_data()
        self.get_model()
        self.compute_instance_retrieval_cache()
        self.num_examples = None
        self.epochs = None
        self.train_test_split = None
    def read_and_maybe_preprocess_data(self) -> None:
        if not os.path.exists(self.args.dataset_dir):
            raise FileNotFoundError(f"Path to dataset does not exist: {self.args.dataset_dir}")
        if not os.path.exists(f"{self.args.dataset_dir}/kb/ontology.owl"):
            raise FileNotFoundError(
                f"Path to dataset does not exist: {self.args.dataset_dir}/kb/ontology.owl")
        if not glob.glob(f"{self.args.dataset_dir}/embeddings/*embeddings.csv"):
            raise FileNotFoundError(f"No embeddings found at {self.args.dataset_dir}/embeddings/")
        if not glob.glob(f"{self.args.dataset_dir}/training_data/*.json"):
            raise FileNotFoundError(
                f"No training data found at {self.args.dataset_dir}/training_data/")

        if self.args.model == "composite" and self.args.use_pma:
            self.pma_net = PMAnet(CNIRConfig().embedding_dim, CNIRConfig().num_attention_heads, 1)
            self.pma_net.load_state_dict(torch.load(self.args.pma_model_path, map_location="cpu", weights_only=True))
        self.kb, self.all_individuals, self.embeddings = read_embs_and_apply_agg(
            self.args.dataset_dir, nn_agg=self.pma_net if hasattr(self, "pma_net") else None, merge=True)
        #remove_atomic_concepts = self.args.model.lower() == "composite"
        self.data = None
    def get_model(self):
        AutoModel.register(CNIRConfig, CNIRComposite)
        try:
            self.model = AutoModel.from_pretrained(self.args.pretrained_model_path)
            print("\n\x1b[6;30;42mSuccessfully loaded a pretrained model!\x1b[0m\n")
        except Exception as e:
            print(e)
    def compute_instance_retrieval_cache(self):
        kb_namespace = list(self.kb.ontology.classes_in_signature())[0].str
        if "#" in kb_namespace:
            kb_namespace = kb_namespace.split("#")[0] + "#"
        elif "/" in kb_namespace:
            kb_namespace = kb_namespace[:kb_namespace.rfind("/")] + "/"
        elif ":" in kb_namespace:
            kb_namespace = kb_namespace[:kb_namespace.rfind(":")] + ":"
        expression_parser = DLSyntaxParser(kb_namespace)
        self.concept_to_instance_set = {expr: set(
            [ind.str.split("/")[-1] for ind in self.kb.individuals(expression_parser.parse(expr))])
            for expr in self.data}
    def start(self) -> dict:
        self.start_time = time.time()
        print("\nStarting testing...")
        print(f"Start time:{datetime.datetime.now()}\n")
        self.trainer = Trainer(self.model, self.tokenizer, self.data, self.embeddings,
                               self.all_individuals, self.concept_to_instance_set,
                               self.num_examples, self.args.th, self.args.optimizer, self.args.pretrained_model_path,
                               self.args.output_dir, self.epochs, self.args.batch_size,
                               self.args.num_workers, self.args.lr, self.train_test_split)
        return self.trainer
class ExecutePMA:
    """
    A class for training and evaluating PMAnet model.
    """

    def __init__(self, args):
        set_seed(args.random_seed)
        self.args = args
        self.read_and_prepare_data()
        self.get_model()

    def read_and_prepare_data(self):
        self.emb, self.data, self.instances, self.valid_inds, self.class_emb, self.kb = read_and_prepare_pma_data(
            self.args.dataset_dir)
        if len(self.instances) <= self.args.num_examples:
            self.args.num_examples = len(self.instances)//2

    def get_model(self):
        self.model = PMAnet(256, self.args.num_attention_heads, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def start(self):
        self.start_time = time.time()
        print("\nStarting training...")
        print(f"Start time:{datetime.datetime.now()}\n")
        self.trainer = PMATrainer(self.model, self.optimizer, self.data, self.valid_inds,
                                  self.instances, self.emb, self.args.output_dir, self.args.th,
                                  self.args.batch_size, epochs=self.args.epochs,
                                  num_examples=self.args.num_examples)
        self.trainer.train()

