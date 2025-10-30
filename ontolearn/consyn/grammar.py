from typing import Dict, List, Optional, Set, Tuple
from .tokenizer import ConSynTokenizer

class ConSynGrammarParser:
    def __init__(self, tokenizer: ConSynTokenizer) -> None:
        self.tokenizer = tokenizer

        self.special_token_ids = {
            self.tokenizer.vocab.get(t, -1) for t in self.tokenizer.special_tokens if t in self.tokenizer.vocab
        }
        self.operator_binary_ids = self._get_ids_by_type('OPERATOR_BINARY')
        self.operator_unary_ids = self._get_ids_by_type('OPERATOR_UNARY')
        self.operator_quantifier_ids = self._get_ids_by_type('OPERATOR_QUANTIFIER')
        self.concept_name_ids = self._get_ids_by_type('CONCEPT_NAME')
        self.role_name_ids = self._get_ids_by_type('ROLE_NAME')
        self.individual_ids = self._get_ids_by_type('INDIVIDUAL')
        self.top_concept_ids = self._get_ids_by_type('TOP_CONCEPT')
        self.bottom_concept_ids = self._get_ids_by_type('BOTTOM_CONCEPT')
        
        vocab = self.tokenizer.vocab
        self.dot_id = vocab.get('.', -1)
        self.paren_open_id = vocab.get('(', -1)
        self.paren_close_id = vocab.get(')', -1)
        self.curly_open_id = vocab.get('{', -1)
        self.curly_close_id = vocab.get('}', -1)
        
        self.eos_id = vocab.get('[EOS]', -1)
        self.cls_id = vocab.get('[CLS]', -1)
        self.sep_id = vocab.get('[SEP]', -1)
        self.pad_id = vocab.get('[PAD]', -1)
        self.unk_id = vocab.get('[UNK]', -1) 
        self.task_label_atomic_id = vocab.get('[TASK_LABEL_ATOMIC_ID]', -1)

        self.valid_transitions: Dict[int, Set[int]] = self._build_valid_transitions()

    def _get_ids_by_type(self, token_type: str) -> Set[int]:
        return {self.tokenizer.vocab[t] for t in self.tokenizer.token_types.get(token_type, []) if t in self.tokenizer.vocab}

    def _get_token_type_by_id(self, token_id: int) -> Optional[str]:
        token_str = self.tokenizer.id_to_token.get(token_id)
        if token_str:
            return self.tokenizer.token_to_type.get(token_str)
        return None

    def _compute_balances(self, token_ids: List[int]) -> Tuple[int, int]:
        paren, curly = 0, 0
        for t in token_ids:
            if t == self.paren_open_id: paren += 1
            elif t == self.paren_close_id: paren -= 1
            elif t == self.curly_open_id: curly += 1
            elif t == self.curly_close_id: curly -= 1
        return paren, curly

    def _build_valid_transitions(self) -> Dict[int, Set[int]]:
        transitions = {}

        concept_start_tokens = (
            self.concept_name_ids |
            self.operator_unary_ids |
            self.operator_quantifier_ids | 
            # self.top_concept_ids |
            # self.bottom_concept_ids |
            {self.paren_open_id} 
        )
        
        concept_end_tokens = (
            self.concept_name_ids |
            # self.top_concept_ids |
            # self.bottom_concept_ids |
            {self.paren_close_id, self.curly_close_id}
        )

        transitions[self.cls_id] = concept_start_tokens.copy()
        transitions[self.cls_id].add(self.eos_id) 

        for token_id in (concept_end_tokens | self.individual_ids): 
            transitions[token_id] = self.operator_binary_ids.copy()
            transitions[token_id].add(self.paren_close_id)
            transitions[token_id].add(self.curly_close_id)
            transitions[token_id].add(self.eos_id)

        for op_id in self.operator_binary_ids:
            transitions[op_id] = concept_start_tokens.copy()

        for op_id in self.operator_unary_ids:
            transitions[op_id] = concept_start_tokens.copy().difference(self.operator_unary_ids)

        for q_id in self.operator_quantifier_ids:
            transitions[q_id] = self.role_name_ids.copy()

        for r_id in self.role_name_ids:
            transitions[r_id] = {self.dot_id}

        transitions[self.dot_id] = (
            self.concept_name_ids |         
            # self.top_concept_ids |          
            # self.bottom_concept_ids | 
            self.operator_quantifier_ids |      
            self.operator_unary_ids |       
            {self.paren_open_id} |          
            {self.curly_open_id}            
        )

        transitions[self.paren_open_id] = concept_start_tokens.copy()
        transitions[self.curly_open_id] = self.individual_ids.copy()

        for ind_id in self.individual_ids:
            transitions[ind_id] = {self.curly_close_id}

        return transitions

    def _filter_base_valid_ids(self, is_initial: bool) -> Set[int]:
        valid_next = set(self.tokenizer.vocab.values())

        valid_next.discard(self.unk_id)
        valid_next.discard(self.pad_id)
        valid_next.discard(self.cls_id) 

        if not is_initial:
            for special_id in self.special_token_ids:
                if special_id not in {self.eos_id, self.cls_id, self.pad_id, self.unk_id}:
                    valid_next.discard(special_id)
        
        return valid_next

    def _get_valid_ids_after_token(self, last_token_id: int) -> Set[int]:
        if last_token_id in self.valid_transitions:
            return self.valid_transitions[last_token_id]
        
        print(f"Warning: Unexpected last token ID {last_token_id} (token: {self.tokenizer.id_to_token.get(last_token_id, 'UNKNOWN')}) encountered by grammar parser. Forcing EOS/PAD.")
        return {self.eos_id, self.pad_id}

    def _final_filter(self, valid_ids: Set[int], last_token_id: int, paren: int, curly: int) -> Set[int]:
        if paren < 0 or curly < 0:
            return {self.eos_id, self.pad_id} 
        
        if paren <= 0:
            valid_ids.discard(self.paren_close_id)
        if curly <= 0:
            valid_ids.discard(self.curly_close_id)

        if paren > 0 or curly > 0:
            valid_ids.discard(self.eos_id)
            
        if last_token_id in (
            self.operator_binary_ids |
            self.operator_unary_ids |
            self.operator_quantifier_ids |
            self.role_name_ids | 
            {self.dot_id, self.paren_open_id, self.curly_open_id}
        ):
            valid_ids.discard(self.eos_id)
            
        return valid_ids

    def get_valid_next_token_ids(self, partial_token_ids: List[int]) -> Set[int]:
        is_initial_generation_step = not partial_token_ids or (len(partial_token_ids) == 1 and partial_token_ids[0] == self.cls_id)
        last_token_id = partial_token_ids[-1] if partial_token_ids else self.cls_id

        paren, curly = self._compute_balances(partial_token_ids)

        valid_ids = self._filter_base_valid_ids(is_initial_generation_step)
        valid_ids = valid_ids.intersection(self._get_valid_ids_after_token(last_token_id))
        valid_ids = self._final_filter(valid_ids, last_token_id, paren, curly)

        if len(partial_token_ids) >= 2:
            prev_token_id = partial_token_ids[-2]
            
            if last_token_id in self.operator_binary_ids:
                if prev_token_id in self.concept_name_ids:
                    if prev_token_id in valid_ids:
                        valid_ids.discard(prev_token_id)
        
        if not valid_ids:
            return {self.eos_id, self.pad_id}

        return valid_ids

    def is_expression_grammatical_complete(self, current_sequence: List[int]) -> bool:
        if not current_sequence:
            return False 

        last_token_id = current_sequence[-1]
        paren, curly = self._compute_balances(current_sequence)
        
        if paren != 0 or curly != 0:
            return False
            
        if last_token_id in (
            self.operator_binary_ids |
            self.operator_unary_ids |
            self.operator_quantifier_ids |
            self.role_name_ids |
            {self.dot_id, self.paren_open_id, self.curly_open_id}
        ):
            return False
            
        return True 

    def get_valid_next_token_ids_batch(
        self, batch_partial_token_ids: List[List[int]]
    ) -> List[Set[int]]:
        return [self.get_valid_next_token_ids(seq) for seq in batch_partial_token_ids]