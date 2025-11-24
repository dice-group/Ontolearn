from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, List, Optional, Tuple
from owlapy import owl_expression_to_dl
import torch
import torch.nn as nn

from owlapy.owl_reasoner import StructuralReasoner
from owlapy.class_expression import OWLClassExpression

from ontolearn.consyn.configs import CONFIG
from ontolearn.consyn.tokenizer import ConSynTokenizer
from ontolearn.consyn.utils import ConSynHypothesisSpace, extract_pos_neg_lp, get_target_learning_problems, load_data
from ontolearn.heuristics import  ConSynHeuristic
from ontolearn.utils.static_funcs import compute_f1_score, init_length_metric


if TYPE_CHECKING:
    from ontolearn.consyn.intializer import DLOWLConverter

AcceptedType = Tuple[OWLClassExpression, float, float]


class ConSynRewardFunction(nn.Module):
    def __init__(self, converter:'DLOWLConverter', reasoner: StructuralReasoner, tokenizer:ConSynTokenizer, beta: float, task_validity_bonus: float, 
                brevity_penalty_weight: float, grammar_penalty_weight: float, missing_eos_penalty_weight: float, diversity_weight: float, 
                retrival_bonus_weight: float, max_output_seq_len, length_diversity_weight: float, device: torch.device = None):
        super().__init__()
        self.semantic_tokens_dl_owl_converter = converter
        self.reasoner = reasoner
        self.tokenizer = tokenizer
        self.beta = beta
        self.task_validity_bonus = task_validity_bonus
        self.brevity_penalty_weight = brevity_penalty_weight
        self.grammar_penalty_weight = grammar_penalty_weight
        self.missing_eos_penalty_weight = missing_eos_penalty_weight
        self.diversity_weight = diversity_weight
        self.retrival_bonus_weight = retrival_bonus_weight
        self.length_diversity_weight = length_diversity_weight
        self.max_output_seq_len = max_output_seq_len
        self.device = device if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Diversity within num of predictions for a given input
        self._concept_type_mapping_bonus = {
            'quantifier_logical':('4', 0.35), 
            'quantifiers':('3', 0.25),       
            'logicals':('2', 0.25),          
            'atomic':('1', 0.15),                
        }

        self.full_learning_problem = load_data(CONFIG['LEARNING_PROBLEM_PATH'])
        
    def _get_concept_type_with_bonus(self, concept_tokens: List[str]) -> Tuple[str, int]:
        if not concept_tokens:
            return self._concept_type_mapping_bonus['atomic']

        has_quantifier = ('∃' in concept_tokens) or ('∀' in concept_tokens)
        has_logical_op = ('⊓' in concept_tokens) or ('⊔' in concept_tokens)
        
        if has_quantifier and has_logical_op:
            return self._concept_type_mapping_bonus['quantifier_logical']
        elif has_logical_op:
            return self._concept_type_mapping_bonus['logicals']
        elif has_quantifier:
            return self._concept_type_mapping_bonus['quantifiers']
        
        elif len(concept_tokens) == 1 and self.tokenizer.token_to_type[concept_tokens[0]] == 'CONCEPT_NAME':
            return self._concept_type_mapping_bonus['atomic']

        return self._concept_type_mapping_bonus['atomic']

    def forward(self, semantic_tokens_grouped: List[List[List[int]]], batch: Any, is_grammatically_invalid_grouped: List[List[bool]], 
                has_explicit_eos_grouped: List[List[bool]], hypothesis_threshold_score: float, heuristic_function: ConSynHeuristic, 
                cshs: ConSynHypothesisSpace, fit_mode:Optional[bool]=None, verbose:bool = False) -> torch.Tensor:

        batch_size = len(semantic_tokens_grouped)
        num_k_predictions = len(semantic_tokens_grouped[0])

        flatten_rewards_container = []
        batched_hypothesis_task_container: DefaultDict[str, List[AcceptedType]] = defaultdict(list)
        all_lengths_for_batch = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            rewards_batched = []
            concept_types_batched = []
            task_label_batched = batch['task_label'][i]

            semantic_tokens_batched = semantic_tokens_grouped[i]
            is_grammatically_invalid_batched = is_grammatically_invalid_grouped[i]
            has_explicit_eos_batched = has_explicit_eos_grouped[i]

            if fit_mode:
                task_label_path = CONFIG['FIT_PATH']['TASK_LABEL_MAPPING_PATH']
            else:
                task_label_path = CONFIG['TASK_LABEL_MAPPING_PATH']

            targeted_concept = load_data(task_label_path)['atomic_id_to_task_label'][task_label_batched]
            full_task_examples = self.full_learning_problem['problems'][targeted_concept]
            full_task_examples_lp = extract_pos_neg_lp(full_task_examples)

            for j in range(num_k_predictions):
                concept_tokens = self.tokenizer.decode(semantic_tokens_batched[j], True)
                is_invalid = is_grammatically_invalid_batched[j]
                has_eos = has_explicit_eos_batched[j]

                current_reward = 0.0
                if not concept_tokens:
                    current_reward -= (
                        self.retrival_bonus_weight + self.grammar_penalty_weight 
                        + self.brevity_penalty_weight + int(has_eos)
                        )
                else:
                    if not is_invalid:
                        is_negated = False
                        concept_owl_expr = self.semantic_tokens_dl_owl_converter(concept_tokens)
                        genereated_concept_inidividuals = frozenset(self.reasoner.instances(concept_owl_expr))

                        if len(genereated_concept_inidividuals):
                            current_reward += self.retrival_bonus_weight
                        else:
                            concept_owl_expr = concept_owl_expr.get_object_complement_of()
                            genereated_concept_inidividuals = frozenset(self.reasoner.instances(concept_owl_expr))
                            current_reward -= self.retrival_bonus_weight
                            is_negated = True

                        f1_score = compute_f1_score(genereated_concept_inidividuals, frozenset(full_task_examples_lp.pos), frozenset(full_task_examples_lp.neg))
                        current_reward += f1_score
                        concept_owl_expr_len = init_length_metric().length(concept_owl_expr)

                        if current_reward > (self.retrival_bonus_weight) and (f1_score >= hypothesis_threshold_score): # 
                            targeted_lp = get_target_learning_problems(self.full_learning_problem, targeted_concept)
                            heuristic_score = heuristic_function.apply(concept_owl_expr, targeted_lp)
                            batched_hypothesis_task_container[targeted_concept].append((concept_owl_expr, round(f1_score, 4), is_negated, heuristic_score, concept_owl_expr_len, len(genereated_concept_inidividuals)))

                            if verbose:
                                print(f"{targeted_concept} | {owl_expression_to_dl(concept_owl_expr)} {f1_score:.2f} {heuristic_score:.2f}")
                            
                        normalized_concept_length = concept_owl_expr_len/self.max_output_seq_len
                        current_reward -= self.brevity_penalty_weight * normalized_concept_length

                    if is_invalid:
                        current_reward -= (self.grammar_penalty_weight + self.brevity_penalty_weight)

                    if not has_eos:
                        current_reward -= self.missing_eos_penalty_weight

                all_lengths_for_batch[i].append(len(concept_tokens))

                rewards_batched.append(current_reward)
                concept_type, type_bonus =  self._get_concept_type_with_bonus(concept_tokens)
                current_reward += type_bonus
                concept_types_batched.append(concept_type)
                
            concept_type_counts_hashed = Counter(concept_types_batched)
            for j in range(num_k_predictions):
                concept_tokens = self.tokenizer.decode(semantic_tokens_batched[j], True)
                is_invalid = is_grammatically_invalid_batched[j]

                if len(concept_tokens) and not is_invalid:
                    concept_type = concept_types_batched[j]
                    frequency = concept_type_counts_hashed[concept_type]
                    
                    diversity_bonus = self.diversity_weight * (1.0 / frequency)
                    rewards_batched[j] += diversity_bonus

            lengths_tensor = torch.tensor(all_lengths_for_batch[i], dtype=torch.float, device=self.device)
            if num_k_predictions > 1:
                mean_length = lengths_tensor.mean()
                variance = ((lengths_tensor - mean_length) ** 2).mean()

                normalized_variance = 0.0
                if mean_length > 0:
                    normalized_variance = variance / (mean_length ** 2)

                length_diversity_reward = self.length_diversity_weight * min(normalized_variance, 1.0)
            else:
                length_diversity_reward = 0.0

            rewards_batched = [r + length_diversity_reward for r in rewards_batched]
            flatten_rewards_container.extend(rewards_batched)
        
        return torch.tensor(flatten_rewards_container, dtype=torch.float, device=self.device), cshs.compute(batched_hypothesis_task_container)