from functools import partial
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Union
import torch 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from owlapy.class_expression import OWLClassExpression

from ontolearn.consyn.grammar import ConSynGrammarParser
from ontolearn.consyn.inference import ConSynInference
from ontolearn.consyn.model.model import ConSynGeneratorModel
from ontolearn.consyn.reward import ConSynRewardFunction

from ontolearn.consyn.tokenizer import ConSynTokenizer
from ontolearn.consyn.utils import ConSynHypothesisSpace, ConceptLearningDataset, DataGenerator, custom_collate_fn_for_dataloader
from ontolearn.heuristics import ConSynHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.search import NCESNode


class ConSynTrainer:
    def __init__(self, model: ConSynGeneratorModel, tokenizer: ConSynTokenizer, grammar_parser: ConSynGrammarParser,
        reward_function: ConSynRewardFunction, heuristic_function: ConSynHeuristic, device: torch.device, optimizer: optim.Optimizer, lr: float = 0.001, lr_scheduler: Any = None, 
        max_gen_length: int = 50, gradient_clip_norm: float = 1.0, eval_interval: int = 0, patience: int = 10, expr_save_path: str = 'expriments',
        initial_baseline: float = 0.0, baseline_alpha: float = 0.99, num_k_predictions: int = 1, decoding_strategy: str = "multinomial", temperature: float = 1.0,
        top_k: int = 0, top_p: float = 0.0, triplet_loss_weight: float = 0.3, diversity_loss_weight: float = 0.3, length_diversity_loss_weight: float = 0.3, hypothesis_threshold_score: float = 0.70, verbose: bool = False):

        self.model = model
        self.tokenizer = tokenizer
        self.grammar_parser = grammar_parser
        self.reward_function = reward_function
        self.heuristic_function = heuristic_function
        self.optimizer = optimizer
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.max_gen_length = max_gen_length
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.verbose = verbose
        self.eval_interval = eval_interval
        self.patience = patience
        self.expr_save_path = expr_save_path
        self.model_save_path = expr_save_path + '/model/best_model.pth'
        self.num_k_predictions = num_k_predictions

        self.decoding_strategy = decoding_strategy
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.triplet_loss_weight = triplet_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.length_diversity_loss_weight = length_diversity_loss_weight
        self.reward_baseline = initial_baseline
        self.baseline_alpha = baseline_alpha

        self.model.to(self.device)

        # TODO: setting device to cpu
        self.rl_inference = ConSynInference(self.model, self.tokenizer, self.grammar_parser, self.reward_function, self.max_gen_length, self.device, self.verbose)

        self.best_val_reward = -float('inf')
        self.epochs_no_improve = 0
        self.history: Dict[str, List[float]] = {
            'train_total_loss': [], 'train_rl_loss': [], 'train_triplet_loss': [], 'train_diversity_loss': [],
            'train_length_diversity_loss': [], 'train_reward': [], 'val_reward': [], 'epoch': [], 'baseline': []
        }

        self.hypothesis_threshold_score = hypothesis_threshold_score
        self.cshs =  ConSynHypothesisSpace(max_size=50)
        self.device_val_test = torch.device('cpu')

        self._checkpoint_loaded = False
        self._fit_inference_initialized = False
        self.fit_task_label_mapping = set()
        self.fit_prediction_nodes:List[NCESNode] = None

    def _compute_length_diversity_loss(self, semantic_tokens_grouped: List[List[List[int]]], batch_size: int) -> torch.Tensor:
        if self.num_k_predictions <= 1:
            return torch.tensor(0.0, device=self.device)
            
        batch_length_diversity_loss = 0.0
        for i in range(batch_size):
            lengths = torch.tensor([len(seq) for seq in semantic_tokens_grouped[i]], dtype=torch.float, device=self.device)
            mean_length = lengths.mean()
            variance = ((lengths - mean_length) ** 2).mean()
            batch_length_diversity_loss += variance
        
        return -(batch_length_diversity_loss / batch_size)

    def _compute_diversity_loss(self, generated_ids_flat: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.num_k_predictions <= 1: # new
            return torch.tensor(0.0, device=self.device)
        
        batch_diversity_loss = 0.0
        for i in range(batch_size):
            group_start = i * self.num_k_predictions
            group_end = group_start + self.num_k_predictions
            generated_ids_for_triple = generated_ids_flat[group_start:group_end]
            
            grouped_embeddings = self.model.get_sequence_embeddings(generated_ids_for_triple)
            
            similarity_matrix = F.cosine_similarity(
                grouped_embeddings.unsqueeze(1), 
                grouped_embeddings.unsqueeze(0), 
                dim=2
            )
            
            off_diag_sum = similarity_matrix.sum() - similarity_matrix.diag().sum()

            batch_diversity_loss += off_diag_sum.mean()

        return batch_diversity_loss / batch_size

    def _compute_policy_loss(self, batch_log_probs: List[List[torch.Tensor]], rewards: torch.Tensor) -> torch.Tensor:
        flat_log_probs_summed = []
        for i in range(len(batch_log_probs)):
            for j in range(len(batch_log_probs[i])):
                sequence_log_probs_tensors = batch_log_probs[i][j]
                if sequence_log_probs_tensors:
                    flat_log_probs_summed.append(torch.stack(sequence_log_probs_tensors).sum())
                else:
                    flat_log_probs_summed.append(torch.tensor(0.0, device=self.device))

        log_prob_tensor = torch.stack(flat_log_probs_summed)

        if rewards.dim() == 1 and rewards.size(0) == (log_prob_tensor.size(0) // self.num_k_predictions):
            rewards = rewards.repeat_interleave(self.num_k_predictions)
    
        advantage = (rewards - self.reward_baseline).detach()      
        policy_loss = -(log_prob_tensor * advantage).mean()

        return policy_loss

    def _save_checkpoint(self, epoch: int, val_reward: float):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        torch.save({
            'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_reward': self.best_val_reward, 'epochs_no_improve': self.epochs_no_improve, 'reward_baseline': self.reward_baseline
        }, self.model_save_path)
        
        print(f"Model saved to {self.model_save_path} with val reward: {val_reward:.4f}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at '{path}'. Cannot load model state.")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'reward_baseline' in checkpoint:
            self.reward_baseline = checkpoint['reward_baseline']

            if self.verbose:
                print(f"Loaded reward baseline: {self.reward_baseline:.4f}")

        if 'best_val_reward' in checkpoint:
            self.best_val_reward = checkpoint['best_val_reward']

            if self.verbose:
                print(f"Loaded best validation reward: {self.best_val_reward:.4f}")

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']

            if self.verbose:
                print(f"Resuming training from epoch {start_epoch + 1}")
        
        self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        self.best_val_reward = checkpoint.get('best_val_reward', -float('inf'))
        
        if self.verbose:
            print(f"Checkpoint loaded successfully from '{path}'.")

    def _save_history(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
        json_path = os.path.join(output_dir, "history.json")
    
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"History space saved to: {json_path}")

    def train(self, num_epochs: int, train_dataloader: Optional[Any] = None, val_dataloader: Optional[Any] = None, test_dataloader: Optional[Any] = None):
        if train_dataloader:
            for epoch in range(num_epochs):
                self.model.train()
                total_train_loss = 0.0
                total_rl_loss = 0.0
                total_triplet_loss = 0.0
                total_diversity_loss = 0.0
                total_length_diversity_loss = 0.0
                total_train_rewards = 0.0
                num_train_batches = 0
                
    
                epoch_batched_hypothesis = []
    
                print("="*150)
                # print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
                for batch_idx, batch in enumerate(train_dataloader):
                    input_ids = batch['input_ids'].to(self.device)
                    segment_ids = batch['segment_ids'].to(self.device)
                    enc_mask = batch['attention_mask'].to(self.device)
                    individuals_original_grouped = batch['individuals']
                    task_label_original_grouped = batch['task_label']
                    
                    batch_size = input_ids.size(0)
    
                    self.optimizer.zero_grad()
    
                     # --- Compute Triplet Loss from the encoder embeddings ---
                    _, triplet_loss = self.model.forward(input_ids=input_ids, segment_ids=segment_ids,
                                                        input_attention_mask=enc_mask, use_triplet_loss=True)

                    # with torch.no_grad():
                    generated_ids_flat, per_token_log_probs_flat, \
                    is_grammatically_invalid_flat, semantic_tokens_flat, \
                    has_explicit_eos_flat = self.model.generate_for_rl( input_ids, segment_ids, enc_mask,
                        self.grammar_parser, self.max_gen_length, k=self.num_k_predictions, decoding_strategy=self.decoding_strategy,
                        temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)
                    
                    # Process and group outputs using ConSynInference
                    (generated_ids_grouped, decoded_concepts_grouped, 
                     per_token_log_probs_grouped, is_grammatically_invalid_grouped, 
                     semantic_tokens_grouped, has_explicit_eos_grouped) = \
                        self.rl_inference.process_generated_output(
                            generated_ids_flat, per_token_log_probs_flat, 
                            is_grammatically_invalid_flat, semantic_tokens_flat, 
                            has_explicit_eos_flat, batch_size, self.num_k_predictions
                        )
    
                    rewards, batched_hypothesis = self.reward_function.forward(
                        semantic_tokens_grouped,
                        # individuals_original_grouped,
                        batch,
                        is_grammatically_invalid_grouped,
                        has_explicit_eos_grouped,
                        self.hypothesis_threshold_score,
                        self.heuristic_function,
                        self.cshs,
                        verbose=self.verbose
                    )
    
                    if batched_hypothesis:
                        epoch_batched_hypothesis.append(batched_hypothesis)
    
                    self.reward_baseline = self.baseline_alpha * self.reward_baseline + (1 - self.baseline_alpha) * rewards.mean().item()
                    # self.history['baseline'].append(self.reward_baseline)
                    
                    rl_loss = self._compute_policy_loss(per_token_log_probs_grouped, rewards)
                    
                    avg_diversity_loss = self._compute_diversity_loss(generated_ids_flat, batch_size)
    
                    avg_length_diversity_loss = self._compute_length_diversity_loss(semantic_tokens_grouped, batch_size)
    
                    total_loss = rl_loss + (self.triplet_loss_weight * triplet_loss) + \
                                 (self.diversity_loss_weight * avg_diversity_loss) + \
                                 (self.length_diversity_loss_weight * avg_length_diversity_loss) 
                    
                    total_loss.backward()
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
    
                    total_train_loss += total_loss.item()
                    total_rl_loss += rl_loss.item()
                    total_triplet_loss += triplet_loss.item()
                    total_diversity_loss += avg_diversity_loss.item()
                    total_length_diversity_loss += avg_length_diversity_loss.item()
                    total_train_rewards += rewards.mean().item()
                    num_train_batches += 1
    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Batch {batch_idx + 1}/{len(train_dataloader)} | Total Loss: {(total_train_loss / num_train_batches):.4f} Reward Loss: {(total_rl_loss / num_train_batches):.4f}, Triplet Loss: {(total_triplet_loss / num_train_batches):.4f}, Diversity Loss: {total_diversity_loss / num_train_batches:.4f}, Length Diversity Loss: {total_length_diversity_loss / num_train_batches:.4f} | Train Reward: {(total_train_rewards / num_train_batches):.4f} | Baseline: {self.reward_baseline:.4f}")
                        print()

                    # Optional cleanup
                    # del input_ids, segment_ids, enc_mask
                    # del generated_ids_flat, per_token_log_probs_flat, is_grammatically_invalid_flat
                    # del semantic_tokens_flat, has_explicit_eos_flat
                    # torch.cuda.empty_cache()
    
                    if val_dataloader is not None and self.eval_interval > 0 and (batch_idx + 1) % self.eval_interval == 0:
                        val_reward, _ = self.rl_inference.evaluate(val_dataloader, heuristic_function=self.heuristic_function, decoding_strategy=self.decoding_strategy,
                                                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, num_k_predictions=self.num_k_predictions, 
                                                hypothesis_threshold_score=self.hypothesis_threshold_score, cshs=self.cshs, device = self.device_val_test)
                        print(f"  Batch {batch_idx+1}/{len(train_dataloader)} | Val Reward: {val_reward:.4f}")
                        
                        self.model.train()
    
                        if val_reward >= self.best_val_reward:
                            self.best_val_reward = val_reward
                            self.epochs_no_improve = 0
                            self._save_checkpoint(epoch, val_reward)
                        else:
                            self.epochs_no_improve += 1
                            print(f"  Validation reward did not improve. Epochs without improvement: {self.epochs_no_improve}")
    
                avg_train_loss = total_train_loss / num_train_batches
                avg_rl_loss = total_rl_loss / num_train_batches
                avg_triplet_loss = total_triplet_loss / num_train_batches
                avg_diversity_loss = total_diversity_loss / num_train_batches
                avg_length_diversity_loss = total_length_diversity_loss / num_train_batches
                avg_train_reward = total_train_rewards / num_train_batches
                
                self.history['train_total_loss'].append(avg_train_loss)
                self.history['train_rl_loss'].append(avg_rl_loss)
                self.history['train_triplet_loss'].append(avg_triplet_loss)
                self.history['train_diversity_loss'].append(avg_diversity_loss)
                self.history['train_length_diversity_loss'].append(avg_length_diversity_loss) # Store new loss
                self.history['train_reward'].append(avg_train_reward)
                self.history['baseline'].append(self.reward_baseline)
                self.history['epoch'].append(epoch + 1)
    
                self.cshs.commit('train', self.cshs.compute(epoch_batched_hypothesis))
    
                print("="*150)
                print(f"Epoch {epoch + 1} Summary | Total Loss: {avg_train_loss:.4f} Reward Loss: {avg_rl_loss:.4f}, Triplet Loss: {avg_triplet_loss:.4f}, Diversity Loss: {avg_diversity_loss:.4f}, Length Diversity Loss: {avg_length_diversity_loss:.4f} | Train Reward: {avg_train_reward:.4f} | Baseline: {self.reward_baseline:.4f}")
                print()
    
                if val_dataloader:
                    val_reward_at_epoch_end, val_epoch_batched_hypothesis = self.rl_inference.evaluate(val_dataloader, decoding_strategy=self.decoding_strategy, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                                            num_k_predictions=self.num_k_predictions, hypothesis_threshold_score=self.hypothesis_threshold_score, heuristic_function=self.heuristic_function, cshs=self.cshs, device = self.device_val_test)
                    
                    self.history['val_reward'].append(val_reward_at_epoch_end)
                    self.cshs.commit('val', self.cshs.compute(val_epoch_batched_hypothesis))
                    print("="*150)
                    print(f"Epoch {epoch+1} Validation Reward: {val_reward_at_epoch_end:.4f}")
    
                    if val_reward_at_epoch_end > self.best_val_reward:
                        self.best_val_reward = val_reward_at_epoch_end
                        self.epochs_no_improve = 0
                        self._save_checkpoint(epoch, val_reward_at_epoch_end)
                    else:
                        self.epochs_no_improve += 1
                        print(f"Validation reward did not improve. Epochs without improvement: {self.epochs_no_improve}")
    
                    if self.epochs_no_improve >= self.patience:
                        print(f"Early stopping triggered after {epoch+1} epochs due to no improvement for {self.patience} epochs.")
                        break
                    print()
                
                if self.lr_scheduler:
                    self.lr_scheduler.step()
    
                self.model.to(self.device)
            print()
            print("\nTraining complete.")
            print(f"Best validation reward achieved: {self.best_val_reward:.4f}")
            print()
            print()

        if test_dataloader:
            print("\n--- Starting final test evaluation ---")

            if os.path.exists(self.model_save_path):
                print(f"Loading best model from {self.model_save_path} for final test evaluation.")
                self.load_checkpoint(self.model_save_path)
            else:
                print(f"Warning: Best model checkpoint not found at {self.model_save_path}. Evaluating the final trained model.")

            test_reward, test_batched_hypothesis = self.rl_inference.evaluate(test_dataloader, decoding_strategy=self.decoding_strategy, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                            num_k_predictions=self.num_k_predictions, heuristic_function=self.heuristic_function, cshs=self.cshs, device = self.device_val_test)
            
            self.history['test_reward'] = test_reward
            self.cshs.commit('test', self.cshs.compute(test_batched_hypothesis))
            print(f"Final Test Reward: {test_reward:.4f}")
            print()

        if train_dataloader:
            self.cshs.display(paradigm="train")
            
        if val_dataloader:
            self.cshs.display(paradigm="val")
            
        if test_dataloader:
            self.cshs.display(paradigm="test")

        self.history['patience'] = self.patience

        self._save_history(self.expr_save_path)
        self.cshs.save(self.expr_save_path)

    def prepare_for_fit(self, verbose:bool = False):
        self.device = torch.device("cpu")
        self.model.to(self.device)

        if not self._checkpoint_loaded:
            self.load_checkpoint(self.model_save_path)
            self._checkpoint_loaded = True

            if self.verbose:
                print("Checkpoint loaded and ready for fit.")

        if not self._fit_inference_initialized:
            self.rl_inference = ConSynInference(
                self.model, self.tokenizer, self.grammar_parser,
                self.reward_function, self.max_gen_length, self.device, self.verbose
            )
            self._fit_inference_initialized = True

            if self.verbose:
                print("Inference initialized and ready for fit.")
        print()
    
    def fit(self, knowledge_base: KnowledgeBase, target_concept:str, target_concept_lp: PosNegLPStandard, path:dict, num_predictions: Optional[int] = None, use_sample_ratio: Optional[int] = 50, verbose:bool = False):
        self.device = torch.device("cpu")
        self.model.to(self.device)

        if num_predictions is None:
            num_predictions = self.num_k_predictions

        assert isinstance(target_concept_lp, PosNegLPStandard), f"Learning problem must be a non-None instance of PosNegLPStandard"

        # Ensure checkpoint is loaded and inference is prepared
        if not getattr(self, "_checkpoint_loaded", False) or not getattr(self, "_fit_inference_initialized", False):
            self.prepare_for_fit(verbose=verbose)
        
        def sample_k_percent(data, k=50):
            data_list = list(data)
            n = int(len(data_list) * k / 100)
            return frozenset(random.sample(data_list, n))
        
        if use_sample_ratio is not None:
            positive_examples = [i.str for i in sample_k_percent(target_concept_lp.pos, use_sample_ratio)]
            negative_examples = [i.str for i in sample_k_percent(target_concept_lp.neg, use_sample_ratio)]
        else:
            positive_examples = [i.str for i in target_concept_lp.pos]
            negative_examples = [i.str for i in target_concept_lp.neg]

        data = {'problems': {target_concept: {'positive_examples':positive_examples, 'negative_examples':negative_examples}}}
        
        target_data_generator = DataGenerator(kb_instance=knowledge_base, json_file_path=None, mapping_file_path=path['TASK_LABEL_MAPPING_PATH'])
        target_raw_data =  target_data_generator.generate_data(lp_data=data)
        
        if target_concept not in self.fit_task_label_mapping:
            target_data_generator.save_task_label_mappings()
            self.fit_task_label_mapping.add(target_concept)

        target_raw_dataset = ConceptLearningDataset(target_raw_data, self.tokenizer)
        collate_fn_fit = partial(custom_collate_fn_for_dataloader, tokenizer=self.tokenizer)

        target_dataloader = DataLoader(
            target_raw_dataset, 
            batch_size=1, 
            shuffle=False, 
            # num_workers=config['num_dataloader_workers'],
            collate_fn=collate_fn_fit,
            pin_memory=True
        )

        paradigm = "fit"
        fit_reward, fit_epoch_batched_hypothesis = self.rl_inference.evaluate(target_dataloader, decoding_strategy=self.decoding_strategy, temperature=self.temperature,
                                    top_k=self.top_k, top_p=self.top_p, num_k_predictions=num_predictions, heuristic_function=self.heuristic_function, cshs=self.cshs, fit_mode=True)
        self.cshs.commit(paradigm, self.cshs.compute(fit_epoch_batched_hypothesis))
        self.fit_prediction_nodes = self.cshs.export_nces_nodes(num_nodes=3, paradigm=paradigm, key=target_concept)[paradigm][target_concept]
        return self

    def best_hypotheses(self, n=1, return_node: bool = False) -> Union[OWLClassExpression, Iterable[OWLClassExpression], None]:
        if self.fit_prediction_nodes is None:
            print("Use ConSynTrainer(..).fit() for the given learning problem")
            return None
        elif len(self.fit_prediction_nodes) == 1 or n == 1:
            best_pred = self.fit_prediction_nodes[0]
            if return_node:
                return best_pred
            return best_pred.concept
        else:
            if return_node:
                return self.fit_prediction_nodes
            return [best_pred.concept for best_pred in self.fit_prediction_nodes[:n]]