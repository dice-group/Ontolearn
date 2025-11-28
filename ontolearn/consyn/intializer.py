import os
import random
from typing import Any, Dict, List, Literal, Tuple, Union
import numpy as np
from owlapy import dl_to_owl_expression
from owlapy.owl_reasoner import StructuralReasoner
import torch


from ontolearn.consyn.grammar import ConSynGrammarParser
from ontolearn.consyn.reward import ConSynRewardFunction
from ontolearn.consyn.tokenizer import ConSynTokenizer
from ontolearn.consyn.utils import DataGenerator, DataSplitter
from ontolearn.heuristics import ConSynHeuristic
from ontolearn.knowledge_base import KnowledgeBase


class DLOWLConverter:
    def __init__(self, namespace: str, dl_to_owl_expression_func):
        self.namespace = namespace
        self.dl_to_owl_expression = dl_to_owl_expression_func

    def __call__(self, parts: Union[str, List[str]]) -> Any:
        if isinstance(parts, list):
            expr = ''.join(parts)

        expr = expr.replace('⊔', ' ⊔ ').replace('⊓', ' ⊓ ')
        return self.dl_to_owl_expression(expr, self.namespace)


class Initializer:
    def __init__(self, config: Dict[str, Any], mode: Literal["train", "fit"] = "train", verbose: bool = False):
        self.config = config
        self.mode = mode
        self.verbose = verbose
        
        self.knowledge_base: KnowledgeBase = None
        self.tokenizer: ConSynTokenizer = None
        self.reasoner: StructuralReasoner = None
        self.grammar_parser: ConSynGrammarParser = None
        self.reward: ConSynRewardFunction = None
        self.datasets: Tuple[List[Dict]] = []
        self.converter: DLOWLConverter = None
        self.heuristic: ConSynHeuristic = None

        self._set_global_seed(self.config['seed'])
        self._initialize_data_and_mappings()
        self._initialize_core_components()

    def _initialize_data_and_mappings(self):
        temp_mapping_loader = DataGenerator(
            kb_instance=None,
            json_file_path=None,
            mapping_file_path=self.config['TASK_LABEL_MAPPING_PATH']
        )
        temp_mapping_loader.load_task_label_mappings()

        try:
            self.knowledge_base = KnowledgeBase(path=self.config['KNOWLEDGE_BASE_PATH'])
        except Exception as e:
            raise RuntimeError(f"Error loading Knowledge Base from {self.config['KNOWLEDGE_BASE_PATH']}: {e}")

        data_generator = DataGenerator(
            kb_instance=self.knowledge_base,
            json_file_path=self.config['LEARNING_PROBLEM_PATH'],
            mapping_file_path=self.config['TASK_LABEL_MAPPING_PATH']
        )
        data_generator.task_label_to_atomic_id = temp_mapping_loader.task_label_to_atomic_id
        data_generator.atomic_id_to_task_label = temp_mapping_loader.atomic_id_to_task_label
        data_generator.atomic_id_counter = temp_mapping_loader.atomic_id_counter

        raw_data_original = data_generator.load_data(self.config['GENERATED_DATA_PATH'])

        if not raw_data_original:
            print("Pre-generated data not found or failed to load. Attempting to generate new data...")
            raw_data_original = data_generator.generate_data()

            if raw_data_original:
                data_generator.save_data(self.config['GENERATED_DATA_PATH'])
                data_generator.save_task_label_mappings()
            else:
                raise RuntimeError("No data generated. Please check 'lp.json' and 'benchmark-dataset.owl' content and paths.")
        else:
            if self.verbose:
                print(f"Successfully loaded {len(raw_data_original)} existing raw data entries.")

            data_generator.load_task_label_mappings()  

        if self.mode == 'train':
            splitter = DataSplitter(raw_data_original)
            self.datasets = splitter.run(save=True, prefix=self.config['EXPERIMENT_DIR'])
        elif self.mode == "fit":
            self.datasets = raw_data_original
    
    def _initialize_core_components(self):
        self.tokenizer = ConSynTokenizer(
            knowledge_base_path=self.config['KNOWLEDGE_BASE_PATH'], 
            mapping_file_path=self.config['TASK_LABEL_MAPPING_PATH']
        )

        self.grammar_parser = ConSynGrammarParser(self.tokenizer)

        ontology = self.knowledge_base.ontology
        namespace = list(ontology.classes_in_signature())[0].iri.get_namespace()

        self.converter = DLOWLConverter(namespace, dl_to_owl_expression)

        self.reasoner = StructuralReasoner(ontology, negation_default=True, sub_properties=True)
        
        self.reward = ConSynRewardFunction(self.converter, self.reasoner, self.tokenizer,
            beta=1.0, task_validity_bonus=0.0, brevity_penalty_weight=0.15, grammar_penalty_weight=1.0, 
            missing_eos_penalty_weight=0.05, diversity_weight=0.25, length_diversity_weight=0.25, retrival_bonus_weight=0.15,
            max_output_seq_len=self.config['max_output_seq_len'], device=self.config['device']
        )

        self.heuristic = ConSynHeuristic(self.knowledge_base)

    def _set_global_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_components(self) -> Dict[str, Any]:
        return {
            'converter': self.converter,
            'datasets': self.datasets,
            'grammar_parser': self.grammar_parser,
            'reasoner': self.reasoner,
            'reward': self.reward,
            'tokenizer': self.tokenizer,
            'heuristic': self.heuristic,
        }
