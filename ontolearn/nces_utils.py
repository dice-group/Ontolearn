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

"""NCES utils."""
from collections import defaultdict
import os
import random
from typing import Dict, List, Optional, TypeAlias, Union
import numpy as np
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.lp_generator import LPGen

from owlapy import owl_expression_to_dl
from owlapy.class_expression import OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.search import NCESNode


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleSolution:

    def __init__(self, vocab, atomic_concept_names):
        self.name = 'SimpleSolution'
        self.atomic_concept_names = atomic_concept_names
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],show_progress=False)
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.train_from_iterator(vocab, trainer)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.tokenizer.pad_token = "[PAD]"

    def predict(self, expression: str):
        atomic_classes = [atm for atm in self.tokenizer.tokenize(expression) if atm in self.atomic_concept_names]
        if atomic_classes == []:
            # If no atomic class found, then randomly pick and use the first 3
            random.shuffle(self.atomic_concept_names)
            atomic_classes = self.atomic_concept_names[:3]
        return " ⊔ ".join(atomic_classes)


def sample_examples(pos, neg, num_ex):
    if min(len(pos), len(neg)) >= num_ex // 2:
        if len(pos) > len(neg):
            num_neg_ex = num_ex // 2
            num_pos_ex = num_ex - num_neg_ex
        else:
            num_pos_ex = num_ex // 2
            num_neg_ex = num_ex - num_pos_ex
    elif len(pos) + len(neg) >= num_ex and len(pos) > len(neg):
        num_neg_ex = len(neg)
        num_pos_ex = num_ex - num_neg_ex
    elif len(pos) + len(neg) >= num_ex and len(pos) < len(neg):
        num_pos_ex = len(pos)
        num_neg_ex = num_ex - num_pos_ex
    else:
        num_pos_ex = len(pos)
        num_neg_ex = len(neg)
    positive = np.random.choice(pos, size=min(num_pos_ex, len(pos)), replace=False)
    negative = np.random.choice(neg, size=min(num_neg_ex, len(neg)), replace=False)
    return positive.tolist(), negative.tolist()


def try_get_embs(pos, neg, embeddings, num_examples):
    """
    Depending on the KGE model, some individuals do not get assigned to any embedding during training. This function filters out such individuals from the provided positive/negative examples. It also
    """
    try:
        _ = embeddings.loc[pos]
    except Exception as e:
        # Some individuals do not appear in the embeddings
        new_pos = list(filter(lambda x: x in embeddings.index, pos))
        if new_pos and len(new_pos) >= len(pos)-len(new_pos):
            pos = new_pos + new_pos[:len(pos)-len(new_pos)]
        else:
            i = 0
            while not new_pos:
                new_pos, _ = sample_examples(pos, neg, num_examples)
                new_pos = list(filter(lambda x: x in embeddings.index, new_pos))
                i += 1
                if i > 3:
                    break
            if not new_pos:
                pos = np.random.choice(list(embeddings.index), num_examples//2).tolist()
                #if contains_prefix:
                #    pos = list(map(lambda x: x.split("/")[-1], pos))
            elif len(new_pos) > len(pos):
                pos = new_pos[:len(pos)]
            else:
                pos = new_pos + new_pos[:len(pos)-len(new_pos)]
        
    if len(pos) + len(neg) < num_examples:
        neg = neg + neg[:num_examples-len(pos)-len(neg)]
        
    elif len(pos) + len(neg) > num_examples:
        neg = neg[:num_examples-len(pos)]
        
    try:
        _ = embeddings.loc[neg]
    except Exception as e:
        # Some individuals do not appear in the embeddings
        new_neg = list(filter(lambda x: x in embeddings.index, neg))
        if new_neg and len(new_neg) >= len(neg)-len(new_neg):
            neg = new_neg + new_neg[:len(neg)-len(new_neg)]
        else:
            i = 0
            while not new_neg:
                _, new_neg = sample_examples(pos, neg, num_examples)
                new_neg = list(filter(lambda x: x in embeddings.index, new_neg))
                i += 1
                if i > 3:
                    break
            if not new_neg:
                neg = np.random.choice(list(embeddings.index), num_examples-len(pos)).tolist()
            elif len(new_neg) > len(neg):
                neg = new_neg[:len(neg)]
            else:
                neg = new_neg + new_neg[:len(neg)-len(new_neg)]

    return pos, neg


def generate_training_data(kb_path,kb, max_num_lps=1000, refinement_expressivity=0.2, refs_sample_size=50,
                           beyond_alc=True, storage_path=None):
    if storage_path is None:
        storage_path = "./Training_Data"
    lp_gen = LPGen(kb_path=kb_path, kb=kb, max_num_lps=max_num_lps, refinement_expressivity=refinement_expressivity,
                   num_sub_roots=refs_sample_size,
                   beyond_alc=beyond_alc, storage_path=storage_path)
    lp_gen.generate()
    print("Loading generated data...")
    try:
        with open(f"{storage_path}/LPs.json") as file:
            lps = json.load(file)
            if isinstance(lps, dict):
                lps = list(lps.items())
            print("Number of learning problems:", len(lps))
    except UnicodeDecodeError:
        with open(f"{storage_path}/LPs.json", encoding='utf-8') as file:
            lps = json.load(file)
            if isinstance(lps, dict):
                lps = list(lps.items())
            print("Number of learning problems:", len(lps))
    return lps

# ConSyn utils
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
