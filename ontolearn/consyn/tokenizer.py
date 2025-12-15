import json
import os
from typing import Dict, List, Optional, Set

from sortedcontainers import SortedSet
from ontolearn.knowledge_base import KnowledgeBase

class ConSynTokenizer:
    dl_core_tokens: Set[str]
    special_tokens: Set[str]
    vocab: Dict[str, int]
    id_to_token: Dict[int, str]
    vocab_size: int
    token_types: Dict[str, Set[str]]
    token_to_type: Dict[str, str]
    
    def __init__(self, knowledge_base_path: str, mapping_file_path: Optional[str] = None) -> None:
        assert isinstance(knowledge_base_path, str) and knowledge_base_path.strip(), "Knowledge base path is required"

        self.knowledge_base = KnowledgeBase(path=knowledge_base_path)

        self.dl_core_tokens = set()
        self.vocab = {}
        self.id_to_token = {}
        self.token_to_type = {}

        self.special_tokens = {'[PAD]', '[CLS]', '[EOS]', '[SEP]',
                               '[TASK_LABEL_START]', '[TASK_LABEL_END]',
                               '[POS_START]', '[POS_END]', '[NEG_START]', '[NEG_END]'}
        
        self.token_types = {
            'CONCEPT_NAME': {concept.iri.get_remainder() for concept in self.knowledge_base.get_concepts()},
            'ROLE_NAME': {relation.iri.get_remainder() for relation in self.knowledge_base.get_object_properties()},
            'INDIVIDUAL': {individual.iri.get_remainder() for individual in self.knowledge_base.individuals()},
            'OPERATOR_BINARY': {"⊓", "⊔"},
            'OPERATOR_UNARY': {"¬"},
            'OPERATOR_QUANTIFIER': {"∃", "∀"},
            'DOT': {'.'},
            'PAREN_OPEN': {"("},
            'PAREN_CLOSE': {")"},
            'CURLY_OPEN': {'{'}, 
            'CURLY_CLOSE': {'}'},
            'TOP_CONCEPT': {'⊤'},
            'BOTTOM_CONCEPT': {'⊥'},
            'TASK_LABEL_ATOMIC_ID': set(),
        }
        # print(len(self.token_types['INDIVIDUAL']))
        self.token_types['SPECIAL'] = self.special_tokens
        self.token_types['TASK_LABEL_ATOMIC_ID'] = set()

        for token_type, tokens in self.token_types.items():
            for token in tokens:
                self.token_to_type[token] = token_type
                self.dl_core_tokens.add(token)
        
        self._build_vocab()

        if mapping_file_path:
            self._load_and_add_dynamic_tokens(mapping_file_path)

    def _build_vocab(self) -> None:
        tokens_to_add: Set[str] = SortedSet(self.special_tokens.union(self.dl_core_tokens))
        for index, token in enumerate(tokens_to_add):
            if token not in self.vocab:
                self.vocab[token] = index
                self.id_to_token[index] = token
        
        self.vocab_size = len(self.vocab)
    
    def _load_and_add_dynamic_tokens(self, mapping_file_path: str) -> None:
        if not os.path.exists(mapping_file_path):
            return
        try:
            with open(mapping_file_path, 'r') as f:
                mappings_data = json.load(f)
                atomic_ids = sorted(mappings_data.get("atomic_id_to_task_label", {}).keys())
                self._add_tokens_to_vocab(atomic_ids)
        except (json.JSONDecodeError, IOError):
            pass
    
    def _add_tokens_to_vocab(self, new_tokens: List[str]) -> None:
        current_max_id = max(self.id_to_token.keys()) if self.id_to_token else -1
        for token in sorted(new_tokens):
            if token not in self.vocab:
                current_max_id += 1
                self.vocab[token] = current_max_id
                self.id_to_token[current_max_id] = token
                self.token_types['TASK_LABEL_ATOMIC_ID'].add(token) 
                self.token_to_type[token] = 'TASK_LABEL_ATOMIC_ID'
        self.vocab_size = len(self.vocab)

    def encode(self, text_tokens: List[str]) -> List[int]:
        return [self.vocab.get(token, self.vocab['[PAD]']) for token in text_tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        if skip_special_tokens:
            return [self.id_to_token.get(idx, '[UNK]') for idx in token_ids if self.id_to_token[idx] not in self.special_tokens] 
        
        return [self.id_to_token.get(idx, '[UNK]') for idx in token_ids]

    def tokenize_dl_expression(self, expr_str: str) -> List[str]:
        tokens: List[str] = []
        current_token: str = ""
        i: int = 0
        while i < len(expr_str):
            char: str = expr_str[i]
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in ['(', ')', '.', '{', '}', '⊤', '⊥', '⊓', '⊔', '¬', '∃', '∀']:
                if current_token:
                    tokens.append(current_token)
                tokens.append(char)
                current_token = ""
            else:
                current_token += char
            i += 1
        if current_token:
            tokens.append(current_token)
        
        final_tokens: List[str] = []
        for t in tokens:
            if t in self.vocab:
                final_tokens.append(t)
            else:
                split_successful = False
                for op in ['⊓', '⊔', '¬', '∃', '∀', '.', '{', '}']:
                    if op in t:
                        parts = t.split(op)
                        temp_split = []
                        for part in parts:
                            if part:
                                temp_split.append(part)
                            temp_split.append(op)
                        temp_split.pop()
                        if all(p in self.vocab for p in temp_split) and len(temp_split) > 1:
                            final_tokens.extend(temp_split)
                            split_successful = True
                            break
                if not split_successful:
                    split_by_cap: List[str] = []
                    current_word: List[str] = []
                    for char_idx, char in enumerate(t):
                        if char.isupper() and char_idx > 0 and not t[char_idx-1].isupper():
                            if current_word:
                                split_by_cap.append("".join(current_word))
                                current_word = []
                        current_word.append(char)
                    if current_word:
                        split_by_cap.append("".join(current_word))
                    
                    if all(word in self.vocab for word in split_by_cap) and len(split_by_cap) > 1:
                        final_tokens.extend(split_by_cap)
                    else:
                        final_tokens.append(t)
        return final_tokens
    
    def save(self, path: str):
        required_attrs = [
            "dl_core_tokens",
            "special_tokens",
            "vocab",
            "id_to_token",
            "vocab_size",
            "token_types",
            "token_to_type",
        ]
        for attr in required_attrs:
            assert hasattr(self, attr), f"Tokenizer is missing attribute: {attr}"

        assert isinstance(self.vocab, dict), "vocab must be a dict[str, int]"
        assert isinstance(self.id_to_token, dict), "id_to_token must be a dict[int, str]"
        assert isinstance(self.vocab_size, int), "vocab_size must be int"

        assert len(self.vocab) == len(self.id_to_token), \
            "vocab and id_to_token must have same length"

        data = {
            "dl_core_tokens": list(self.dl_core_tokens),           # sets → lists
            "special_tokens": list(self.special_tokens),
            "vocab": self.vocab,                                   # str → int OK
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()}, # int keys → str
            "vocab_size": self.vocab_size,
            "token_types": {k: list(v) for k, v in self.token_types.items()},
            "token_to_type": self.token_to_type,
        }

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)