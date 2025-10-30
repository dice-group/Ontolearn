from collections import defaultdict
import json
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple, TypeAlias, Union
from owlapy import owl_expression_to_dl
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from ontolearn.knowledge_base import KnowledgeBase

from torch.utils.data import Dataset


from owlapy.class_expression import (
    OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLObjectExactCardinality,
    OWLDataAllValuesFrom, OWLDataSomeValuesFrom, OWLDataHasValue, OWLObjectHasValue,
    # OWLObjectComplementOf, OWLObjectCardinalityRestriction, OWLDataMinCardinality
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.class_expression import OWLClassExpression


from ontolearn.consyn.tokenizer import ConSynTokenizer
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.search import NCESNode

class DataGenerator:
    def __init__(self, kb_instance: KnowledgeBase, json_file_path: str, mapping_file_path: str, verbose: bool = False):
        self.kb_instance = kb_instance
        self.json_file_path = json_file_path
        self.mapping_file_path = mapping_file_path
        self.verbose = verbose
        self.generated_raw_data: List[Dict] = []

        self.task_label_to_atomic_id: Dict[str, str] = {}
        self.atomic_id_to_task_label: Dict[str, str] = {}
        self.atomic_id_counter: int = 0

        self.load_task_label_mappings()

        self.data_feature_domain_restriction = (
            OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLObjectExactCardinality,
            OWLDataAllValuesFrom, OWLDataSomeValuesFrom, OWLDataHasValue, OWLObjectHasValue,
        )

    def _get_or_create_atomic_id(self, complex_label_string: str) -> str:
        if complex_label_string in self.task_label_to_atomic_id:
            return self.task_label_to_atomic_id[complex_label_string]
        
        new_atomic_id = f"TASK__{self.atomic_id_counter}"
        self.atomic_id_counter += 1
        
        self.task_label_to_atomic_id[complex_label_string] = new_atomic_id
        self.atomic_id_to_task_label[new_atomic_id] = complex_label_string
        return new_atomic_id

    def _get_features_for_individuals(self, individual_iris: List[str], task_label: str, example_type: str) -> Dict[str, List[str]]:
        features_by_individual: Dict[str, List[str]] = {}
        for individual_iri in individual_iris:
            individual_owl = OWLNamedIndividual(individual_iri)
            
            features_for_individual = [
                owl_expression_to_dl(feat) for feat in self.kb_instance.abox(individual_owl, mode='expression')
                if type(feat) not in self.data_feature_domain_restriction
            ]
            if not features_for_individual:
                print(f"Warning: No relevant features found in KB for {example_type} individual '{individual_iri}' for task '{task_label}'.")
            
            features_for_individual_list = list(set(features_for_individual))
            features_by_individual[individual_iri] = random.sample(features_for_individual_list, min(len(features_for_individual_list), 85))
        return features_by_individual

    def generate_data(self, lp_data: Optional[Dict] = None) -> List[Dict]:
        if lp_data is None:
            print(f"Generating raw data from {self.json_file_path} using KnowledgeBase.")

            try:
                with open(self.json_file_path, 'r') as f:
                    lp_data = json.load(f)
            except FileNotFoundError:
                print(f"Error: JSON file not found at {self.json_file_path}")
                return []
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {self.json_file_path}")
                return []

        problems = lp_data.get("problems", {})
        if not problems:
            print("Warning: No 'problems' found in the JSON file. Generated data will be empty.")
            return []
        
        self.generated_raw_data = []
        
        for original_task_label, problem_details in problems.items():
            atomic_task_label = self._get_or_create_atomic_id(original_task_label)

            positive_example_individuals_iri = problem_details.get("positive_examples", [])
            negative_example_individuals_iri = problem_details.get("negative_examples", [])

            if not positive_example_individuals_iri and not negative_example_individuals_iri:
                print(f"Warning: Task '{original_task_label}' has no positive or negative individuals specified in {self.json_file_path}. Skipping.")
                continue

            positive_individuals_features = self._get_features_for_individuals(
                positive_example_individuals_iri, original_task_label, 'positive'
            )
            negative_individuals_features = self._get_features_for_individuals(
                negative_example_individuals_iri, original_task_label, 'negative'
            )

            if not positive_individuals_features and not negative_individuals_features:
                print(f"Warning: Task '{original_task_label}' resulted in no features after KB query and filtering for any individual. Skipping.")
                continue

            self.generated_raw_data.append({
                "task_label": atomic_task_label,
                "positive_examples": positive_individuals_features,
                "negative_examples": negative_individuals_features
            })
        
        print(f"Successfully generated {len(self.generated_raw_data)} raw data entries.")
        return self.generated_raw_data

    def save_data(self, output_file_path: str) -> None:
        if not self.generated_raw_data:
            print("Warning: No data to save. 'generated_raw_data' is empty.")
            return

        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                json.dump(self.generated_raw_data, f, indent=4)
            print(f"Generated raw data saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving data to {output_file_path}: {e}")

    def load_data(self, input_file_path: str) -> List[Dict]:
        if not os.path.exists(input_file_path):
            print(f"No pre-generated data found at {input_file_path}. Will generate new data.")
            return []
        
        try:
            with open(input_file_path, 'r') as f:
                self.generated_raw_data = json.load(f)
            
            if self.verbose:
                print(f"Successfully loaded raw data from {input_file_path}")
            return self.generated_raw_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {input_file_path}: {e}")
            return []
        except IOError as e:
            print(f"Error reading data from {input_file_path}: {e}")
            return []

    def save_task_label_mappings(self) -> None:
        if not self.task_label_to_atomic_id:
            print("Warning: No task label mappings to save.")
            return
        try:
            os.makedirs(os.path.dirname(self.mapping_file_path), exist_ok=True)
            with open(self.mapping_file_path, 'w') as f:
                json.dump({
                    "task_label_to_atomic_id": self.task_label_to_atomic_id,
                    "atomic_id_to_task_label": self.atomic_id_to_task_label,
                    "atomic_id_counter": self.atomic_id_counter
                }, f, indent=4)
            
            if self.verbose:
                print(f"Task label mappings saved to {self.mapping_file_path}")
        except IOError as e:
            print(f"Error saving task label mappings to {self.mapping_file_path}: {e}")

    def load_task_label_mappings(self) -> None:
        if not os.path.exists(self.mapping_file_path):
            print(f"No existing task label mappings found at {self.mapping_file_path}. Starting fresh.")
            return
        try:
            with open(self.mapping_file_path, 'r') as f:
                mappings_data = json.load(f)
                self.task_label_to_atomic_id = mappings_data.get("task_label_to_atomic_id", {})
                self.atomic_id_to_task_label = mappings_data.get("atomic_id_to_task_label", {})

                self.atomic_id_counter = mappings_data.get("atomic_id_counter", 0)
                
            if self.verbose:
                print(f"Task label mappings loaded from {self.mapping_file_path}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.mapping_file_path}: {e}. Starting mappings fresh.")
            self.task_label_to_atomic_id = {}
            self.atomic_id_to_task_label = {}
            self.atomic_id_counter = 0
        except IOError as e:
            print(f"Error reading task label mappings from {self.mapping_file_path}: {e}. Starting mappings fresh.")
            self.task_label_to_atomic_id = {}
            self.atomic_id_to_task_label = {}
            self.atomic_id_counter = 0

    def get_atomic_task_labels(self) -> List[str]:
        return list(self.atomic_id_to_task_label.keys())

    def get_original_task_label(self, atomic_id: str) -> str:
        return self.atomic_id_to_task_label.get(atomic_id, atomic_id)

class DataSplitter:
    REQUIRED_KEYS = {"task_label", "positive_examples", "negative_examples"}
    
    def __init__(self, task_list: List[Dict[str, Any]], train_size: float = 0.20, val_size: float = 0.10, test_size: float = 0.30, seed: int = 42):
        if not task_list:
            raise ValueError("task_list is empty.")
        self.task_list = task_list
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def robust_split_examples(self, examples_dict: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        keys = list(examples_dict.keys())
        # random.Random(self.seed).shuffle(keys)
        n = len(keys)

        if n == 0:
            return {}, {}, {}

        if n < 3:
            return examples_dict, {}, {}

        train_keys, temp_keys = train_test_split(keys, train_size=self.train_size, random_state=self.seed)
        val_ratio = self.val_size / (self.val_size + self.test_size)
        val_keys, test_keys = train_test_split(temp_keys, train_size=val_ratio, random_state=self.seed)

        return (
            {k: examples_dict[k] for k in train_keys},
            {k: examples_dict[k] for k in val_keys},
            {k: examples_dict[k] for k in test_keys},
        )

    def split_tasks(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_tasks, val_tasks, test_tasks = [], [], []

        for task in self.task_list:
            if not self.REQUIRED_KEYS.issubset(task):
                raise ValueError(f"Task missing required keys: {task}")

            label = task["task_label"]
            pos = task["positive_examples"]
            neg = task["negative_examples"]

            total = len(pos) + len(neg)

            if total < 3:
                train_tasks.append({
                    "task_label": label,
                    "positive_examples": pos,
                    "negative_examples": neg
                })
                val_tasks.append({
                    "task_label": label,
                    "positive_examples": {},
                    "negative_examples": {}
                })
                test_tasks.append({
                    "task_label": label,
                    "positive_examples": {},
                    "negative_examples": {}
                })
            else:
                pos_train, pos_val, pos_test = self.robust_split_examples(pos)
                neg_train, neg_val, neg_test = self.robust_split_examples(neg)

                train_tasks.append({
                    "task_label": label,
                    "positive_examples": pos_train,
                    "negative_examples": neg_train
                })
                val_tasks.append({
                    "task_label": label,
                    "positive_examples": pos_val,
                    "negative_examples": neg_val
                })
                test_tasks.append({
                    "task_label": label,
                    "positive_examples": pos_test,
                    "negative_examples": neg_test
                })

        return train_tasks, val_tasks, test_tasks

    def save_splits(self, train_tasks: List[Dict[str, Any]], val_tasks: List[Dict[str, Any]], test_tasks: List[Dict[str, Any]], prefix: Optional[str] = None) -> None:
        if prefix is not None:
            prefix = prefix + "/data/"
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
        else:
            prefix = ""
        
        with open(f"{prefix}train_split.json", "w") as f:
            json.dump(train_tasks, f, indent=2)
        with open(f"{prefix}val_split.json", "w") as f:
            json.dump(val_tasks, f, indent=2)
        with open(f"{prefix}test_split.json", "w") as f:
            json.dump(test_tasks, f, indent=2)

    def run(self, save: bool = False, prefix: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        train, val, test = self.split_tasks()
        if save:
            self.save_splits(train, val, test, prefix)
        return train, val, test

class ConceptLearningDataset(Dataset): # ConSynDataset
    def __init__(self, raw_data: List[Dict], tokenizer: 'ConSynTokenizer', datatype:Optional[str] = None) -> None:
        self.tokenizer = tokenizer
        # self.datatype = datatype
        self.train_individuals = set()
        self.data_processed: List[Dict] = self._process_raw_data(raw_data)

    def _process_examples(self, examples_dict: Dict[str, List[str]]) -> Tuple[List[str], Set[str]]:
        all_tokens: List[List[str]] = []
        individual_iris: Set[str] = set()

        for individual_iri, features in examples_dict.items():
            all_tokens.extend([self.tokenizer.tokenize_dl_expression(feat_str) for feat_str in features])
            clean_iri = individual_iri[3:] if "{*}" in individual_iri else individual_iri
            individual_iris.add(OWLNamedIndividual(clean_iri))

        return all_tokens, individual_iris
    
    def _process_raw_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed_items: List[Dict] = []

        for entry in raw_data:
            task_label: str = entry["task_label"]

            p_examples_tokens, p_iris = self._process_examples(entry["positive_examples"])
            n_examples_tokens, n_iris = self._process_examples(entry["negative_examples"])

            individual_iris: Dict[str, Set[str]] = {
                "pos": p_iris,
                "neg": n_iris
            }

            tokens_list, input_ids_list = self._prepare_single_input_sequence_unpadded(
                p_examples_tokens, n_examples_tokens, task_label
            )

            is_augmented = any([entry.get('__is_negated_task_origin', False), entry.get('__has_new_augmented_individuals', False)])

            processed_items.append({
                'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
                'task_label': task_label,
                'p_examples_tokens': p_examples_tokens,
                'n_examples_tokens': n_examples_tokens,
                'tokens': tokens_list,
                '__is_augmented': is_augmented,
                'individuals': individual_iris
            })
        return processed_items

    def _prepare_single_input_sequence_unpadded(self, p_examples_tokens: List[List[str]], n_examples_tokens: List[List[str]], task_label: str) -> Tuple[List[str], List[int]]:
        tokens = ['[CLS]', '[TASK_LABEL_START]']
        
        label = task_label[1:] if "¬" in task_label else task_label
        tokens.append(label)
        tokens.append('[TASK_LABEL_END]')

        tokens.append('[POS_START]')
        if p_examples_tokens:
            for p_ex_t in p_examples_tokens[:-1]:
                tokens.extend(p_ex_t)
                tokens.append('[SEP]')
            tokens.extend(p_examples_tokens[-1])
        tokens.append('[POS_END]')

        tokens.append('[NEG_START]')
        if n_examples_tokens:
            for n_ex_t in n_examples_tokens[:-1]:
                tokens.extend(n_ex_t)
                tokens.append('[SEP]')
            tokens.extend(n_examples_tokens[-1])
        tokens.append('[NEG_END]')

        tokens.append('[EOS]')

        input_ids_encoded = [self.tokenizer.vocab[tok] for tok in tokens]

        return tokens, input_ids_encoded

    def __len__(self) -> int:
        return len(self.data_processed)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List, str]]:
        return self.data_processed[idx]

def create_segment_ids(tokens: List[str]) -> List[int]:
    segment_ids = []
    current_segment = 0

    segment_map = {
        '[TASK_LABEL_START]': 1,
        '[TASK_LABEL_END]': 1,
        '[POS_START]': 2,
        '[POS_END]': 2,
        '[NEG_START]': 3,
        '[NEG_END]': 3
    }

    segment_stack = []

    for tok in tokens:
        if tok in segment_map and tok.endswith('START]'):
            current_segment = segment_map[tok]
            segment_stack.append(current_segment)
            segment_ids.append(current_segment)
        elif tok in segment_map and tok.endswith('END]'):
            segment_ids.append(segment_stack[-1] if segment_stack else 0)
            if segment_stack:
                segment_stack.pop()
            current_segment = segment_stack[-1] if segment_stack else 0
        else:
            segment_ids.append(current_segment)

    return segment_ids

def custom_collate_fn_for_dataloader(
    batch: List[Dict],
    tokenizer: ConSynTokenizer,
    max_seq_len: Optional[int] = None,
    pad_segment_id: int = 0
) -> Dict[str, Union[torch.Tensor, List]]:
    pad_token_id = tokenizer.vocab['[PAD]']

    if max_seq_len is not None:
        for item in batch:
            item['input_ids'] = item['input_ids'][:max_seq_len]
            item['tokens'] = item['tokens'][:max_seq_len]

    max_len = max(len(item['input_ids']) for item in batch)

    input_ids_batch = []
    attention_masks = []
    segment_ids_batch = []

    for item in batch:
        ids = item['input_ids']
        tokens = item['tokens']
        segment_ids = create_segment_ids(tokens)

        ids_tensor = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
        seg_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)
        attn_mask = torch.ones(len(ids_tensor), dtype=torch.bool)

        pad_len = max_len - len(ids_tensor)
        input_ids_batch.append(F.pad(ids_tensor, (0, pad_len), value=pad_token_id))
        attention_masks.append(F.pad(attn_mask, (0, pad_len), value=0))
        segment_ids_batch.append(F.pad(seg_ids_tensor, (0, pad_len), value=pad_segment_id))

    input_ids = torch.stack(input_ids_batch)
    attention_mask = torch.stack(attention_masks).unsqueeze(1).unsqueeze(1)
    segment_ids = torch.stack(segment_ids_batch)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'segment_ids': segment_ids,
        'p_examples_tokens': [item['p_examples_tokens'] for item in batch],
        'n_examples_tokens': [item['n_examples_tokens'] for item in batch],
        'task_label': [item['task_label'] for item in batch],
        'tokens': [item['tokens'] for item in batch],
        '__is_augmented': [item['__is_augmented'] for item in batch],
        'individuals': [item['individuals'] for item in batch]
    }

class DecodingStrategy:
    @staticmethod
    def select_next_token(strategy_name: str, masked_logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> Tuple[int, torch.Tensor]:
        if masked_logits.max().item() == float('-inf'):
            return -1, torch.tensor(float('-inf'), device=masked_logits.device)

        # Temperature scaling
        if strategy_name == "multinomial":
            temperature = max(1e-5, temperature)
            logits = masked_logits / temperature
        else:
            logits = masked_logits

        if strategy_name == "greedy":
            selected_token_id = torch.argmax(logits).item()
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_probs[selected_token_id]
            return selected_token_id, selected_log_prob

        elif strategy_name == "multinomial":
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            if top_k > 0:
                top_k = min(top_k, probs.size(-1))
                topk_values, topk_indices = torch.topk(probs, top_k)
                topk_probs = topk_values / topk_values.sum()
                selected_index = torch.multinomial(topk_probs, 1).item()
                selected_token_id = topk_indices[selected_index].item()
                selected_log_prob = log_probs[selected_token_id]
                return selected_token_id, selected_log_prob

            elif top_p > 0.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                cutoff = cumulative_probs > top_p
                if torch.any(cutoff):
                    last_index = torch.nonzero(cutoff, as_tuple=False)[0].item() + 1
                    sorted_probs = sorted_probs[:last_index]
                    sorted_indices = sorted_indices[:last_index]
                    sorted_probs = sorted_probs / sorted_probs.sum()
                selected_index = torch.multinomial(sorted_probs, 1).item()
                selected_token_id = sorted_indices[selected_index].item()
                selected_log_prob = log_probs[selected_token_id]
                return selected_token_id, selected_log_prob

            else:
                selected_token_id = torch.multinomial(probs, 1).item()
                selected_log_prob = log_probs[selected_token_id]
                return selected_token_id, selected_log_prob

        else:
            raise ValueError(f"Unsupported decoding strategy: {strategy_name}. Choose 'greedy' or 'multinomial'.")
        
ConceptTuple: TypeAlias = tuple[OWLClassExpression, float, float, int, int]
KeyedConcepts: TypeAlias = dict[str, list[ConceptTuple]]
ParadigmConcepts: TypeAlias = dict[str, KeyedConcepts]

def load_data(source: Union[str, Dict[str, Dict[str, List[str]]]]) -> Dict[str, Dict[str, List[str]]]:
    if isinstance(source, str):
        if not source.endswith(".json"):
            raise ValueError(f"Expected a .json file path, got: {source}")
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        with open(source, "r") as f:
            return json.load(f)
    elif isinstance(source, dict):
        return source
    else:
        raise ValueError("Source must be a dict or a .json file path.")

def extract_pos_neg_lp(examples_dict: Dict[str, List[str]], pos_key: str = 'positive_examples', neg_key: str = 'negative_examples') -> PosNegLPStandard:
    positive_examples = examples_dict.get(pos_key, [])
    negative_examples = examples_dict.get(neg_key, [])
    
    return PosNegLPStandard(
        pos={OWLNamedIndividual(i) for i in positive_examples},
        neg={OWLNamedIndividual(i) for i in negative_examples}
    )

def get_target_learning_problems(source: Union[str, Dict[str, Dict[str, List[str]]]], target_concept: str, 
                                pos_key: str = 'positive_examples', neg_key: str = 'negative_examples') -> PosNegLPStandard:
    
    data = load_data(source)['problems']
    target_concept_examples = data[target_concept]

    return extract_pos_neg_lp(target_concept_examples, pos_key=pos_key, neg_key=neg_key)

class ConSynHypothesisSpace: # ConSynPriorityQueue - move to search.py
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = defaultdict(lambda: defaultdict(list))

    def get_max_size(self) -> int:
        return self.max_size

    def get_elements(self, paradigm: str, key: str) -> list[ConceptTuple]:
        return self.data[paradigm][key]

    def get_all(self) -> ParadigmConcepts:
        return {
            paradigm: {
                key: items
                for key, items in keys.items()
            }
            for paradigm, keys in self.data.items()
        }

    def _sort_key(self, x):
        return (x[1], -x[2], x[3], x[4], x[5])

    def add(self, paradigm: str, key: str, pred_concept: OWLClassExpression, quality: float, is_negated:bool, heuristic: float, length: int, individual_count: int) -> bool:
        container = self.data[paradigm][key]
        item = (pred_concept, quality, int(is_negated), heuristic, length, individual_count)

        if item in container:
            return False

        container.append(item)

        if len(container) > self.max_size:
            container.sort(key=self._sort_key, reverse=True)
            self.data[paradigm][key] = container[:self.max_size]

        return True

    def compute(self, raw_data: Union[KeyedConcepts, list[KeyedConcepts]]) -> KeyedConcepts:
        if isinstance(raw_data, list):
            raw_data = self._merge_raw_dicts(raw_data)

        results = {}
        for key, items in raw_data.items():
            sorted_items = sorted(items, key=self._sort_key, reverse=True)
            results[key] = sorted_items[:self.max_size]
        return results

    def commit(self, paradigm: str, grouped_data: KeyedConcepts):
        for key, new_items in grouped_data.items():
            current_items = self.get_elements(paradigm, key)
            merged = current_items + new_items

            merged.sort(key=self._sort_key, reverse=True)
            top_k = merged[:self.max_size]

            self.data[paradigm][key] = []
            for pred_concept, quality, is_negated, heuristic, length, individual_count in top_k:
                self.add(paradigm, key, pred_concept, quality, is_negated, heuristic, length, individual_count)

    def display(self, paradigm: str):
        data = self.get_all().get(paradigm, {})
        if not data:
            print(f"[!] No entries found under paradigm '{paradigm}'.\n")
            return

        paradigm_titles = {
            "train": "Training",
            "val": "Validation",
            "test": "Testing",
            "fit": "Fitting"
        }

        title_paradigm = paradigm_titles.get(paradigm.lower(), paradigm.title())
        title = f"Concept Synthesis Hypothesis Space Best Explored Concepts During {title_paradigm}"
        width = len(title) + 30
        print("=" * width)
        print(title.center(width))
        print("=" * width)

        for key, items in sorted(data.items()):
            print(f"\n{key} -", end=" ")
            for pred_concept, quality, _, heuristic, length, individual_count in items:
                print(f'"{owl_expression_to_dl(pred_concept)}" F1: {quality:.2f} Heuristic: {heuristic:.2f} Length: {length} Individuals: {individual_count} |',  end=' ')
            print()
        print("\n")

    def _merge_raw_dicts(self, raw_list: list[KeyedConcepts]) -> KeyedConcepts:
        merged: defaultdict[str, list[ConceptTuple]] = defaultdict(list)
        
        for concept_map in raw_list:
            for key, items in concept_map.items():
                merged[key].extend(items)
        
        return dict(merged)

    def export_nces_nodes(self, num_nodes: int, paradigm: Optional[str] = None, key: Optional[str] = None) -> dict[str, dict[str, list['NCESNode']]]:
        if key is not None and paradigm is None:
            raise ValueError("Cannot export by key alone; 'paradigm' must be specified if 'key' is given.")
            
        num_nodes = min(num_nodes, self.max_size)
        result = {}
        all_data = self.get_all()
        paradigms_to_process = [paradigm] if paradigm else all_data.keys()

        for current_paradigm in paradigms_to_process:
            if current_paradigm not in all_data:
                continue

            keys_to_process = [key] if key else all_data[current_paradigm].keys()
            result[current_paradigm] = {}

            for current_key in keys_to_process:
                if current_key not in all_data[current_paradigm]:
                    continue

                items = all_data[current_paradigm][current_key]
                top_n = sorted(items, key=self._sort_key, reverse=True)[:num_nodes]

                nodes = []
                for pred_concept, quality, _, heuristic, length, individual_count in top_n:
                    if not isinstance(pred_concept, OWLClassExpression):
                        raise TypeError(f"Expected OWLClassExpression, got {type(pred_concept).__name__} for concept '{pred_concept}'")

                    node = NCESNode(concept=pred_concept, length=length, individuals_count=individual_count,
                                    quality=quality, heuristic=heuristic)
                    nodes.append(node)

                result[current_paradigm][current_key] = nodes
        return result

    def clear(self, paradigm: str = None, key: str = None):
        if key is not None and paradigm is None:
            raise ValueError("Cannot clear by key alone; 'paradigm' must be specified if 'key' is given.")

        if paradigm is None:
            self.data.clear()
        elif key is None:
            if paradigm in self.data:
                self.data[paradigm].clear()
        else:
            if paradigm in self.data and key in self.data[paradigm]:
                del self.data[paradigm][key]
                if not self.data[paradigm]:
                    del self.data[paradigm]

    def save(self, directory: str, filename: str = "hypothesis_space.json"):
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
    
        serializable_data = {}
        for paradigm, keys in self.data.items():
            serializable_data[paradigm] = {}
            for key, items in keys.items():
                serializable_data[paradigm][key] = [
                    {
                        "concept": str(owl_expression_to_dl(pred_concept)) if not isinstance(pred_concept, str) else pred_concept,
                        "quality": quality,
                        "is_negated": is_negated,
                        "heuristic": heuristic,
                        "length": length,
                        "individual_count": individual_count
                    }
                    for (pred_concept, quality, is_negated, heuristic, length, individual_count) in items
                ]
    
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)
    
        print(f"Hypothesis space saved to: {filepath}")
