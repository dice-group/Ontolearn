"""
DrillV Variants - From Simple to Complex
Test each variant to see which approach actually works!
"""

from typing import Optional, Iterable, Tuple, Set, List, Union
import torch
import torch.nn as nn
import random
from ontolearn.learners.drill import Drill
from ontolearn.heuristics import CeloeBasedReward
from ontolearn.data_struct import Experience
from ontolearn.abstracts import AbstractNode
from ontolearn.search import RL_State
from owlapy.class_expression import OWLClassExpression
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_dl


# ============================================================================
# KB-Driven Optimization Mixin
# ============================================================================
class KBDrivenOptimizationMixin:
    """
    Mixin class providing KB-driven optimization methods.
    These can be used by any DrillV variant.
    """
    
    def extract_kb_driven_starting_concepts(self, pos_examples: Set[OWLNamedIndividual], 
                                           neg_examples: Set[OWLNamedIndividual],
                                           max_concepts: int = 10) -> List[OWLClassExpression]:
        """
        Extract relevant starting concepts from the KB based on positive and negative examples.
        
        Strategy:
        1. Find classes that positive examples belong to
        2. Filter out classes that negative examples also belong to
        3. Prioritize classes with high positive coverage and low negative coverage
        """
        from collections import defaultdict
        
        class_pos_count = defaultdict(int)
        class_neg_count = defaultdict(int)
        
        for ind in pos_examples:
            for owl_class in self.kb.get_types(ind, direct=False):
                class_pos_count[owl_class] += 1
        
        for ind in neg_examples:
            for owl_class in self.kb.get_types(ind, direct=False):
                class_neg_count[owl_class] += 1
        
        class_scores = []
        for owl_class, pos_count in class_pos_count.items():
            neg_count = class_neg_count.get(owl_class, 0)
            
            if neg_count >= pos_count:
                continue
                
            pos_coverage = pos_count / len(pos_examples) if pos_examples else 0
            neg_penalty = neg_count / len(neg_examples) if neg_examples else 0
            score = pos_coverage - neg_penalty
            
            if score > 0:
                class_scores.append((owl_class, score, pos_count, neg_count))
        
        class_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        starting_concepts = [owl_class for owl_class, score, pos_count, neg_count 
                           in class_scores[:max_concepts]]
        
        if hasattr(self, 'verbose') and self.verbose > 0 and starting_concepts:
            print(f"\n🎯 KB-Driven: Found {len(starting_concepts)} relevant starting concepts:")
            for i, (owl_class, score, pos_count, neg_count) in enumerate(class_scores[:min(5, max_concepts)]):
                print(f"  {i+1}. {owl_expression_to_dl(owl_class)}: "
                      f"Score={score:.3f}, Pos={pos_count}/{len(pos_examples)}, "
                      f"Neg={neg_count}/{len(neg_examples)}")
        
        return starting_concepts
    
    def should_prune_refinement(self, concept: OWLClassExpression, 
                               pos_examples: Set[OWLNamedIndividual],
                               neg_examples: Set[OWLNamedIndividual],
                               min_coverage_threshold: float = 0.1) -> bool:
        """
        Early pruning: Check if a concept is worth exploring based on KB coverage.
        Avoids computing quality for concepts that clearly won't work.
        """
        try:
            instances = set(self.kb.individuals(concept))
            pos_covered = len(instances.intersection(pos_examples))
            neg_covered = len(instances.intersection(neg_examples))
            
            if pos_covered < len(pos_examples) * min_coverage_threshold:
                return True
            
            if neg_covered == len(neg_examples) and neg_covered > 0:
                return True
                
            return False
            
        except Exception:
            return False


# ============================================================================
# VARIANT 1: BASELINE - No RL, just use quality as heuristic
# ============================================================================
class DrillV_Baseline(Drill):
    """
    Baseline: No neural network, just use concept quality as heuristic.
    This is our control - shows what happens with no learning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DrillV_Baseline"
        self.heuristic_func = CeloeBasedReward()
    
    def train(self, *args, **kwargs):
        print("Baseline: No training needed")
        return self.terminate_training()


# ============================================================================
# VARIANT 2: MINIMAL - Simplest possible neural network
# ============================================================================
class DrillVNet_Minimal(nn.Module):
    """Dead simple: Input -> Hidden -> Output. That's it."""
    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Just 2 layers, no fancy stuff
        self.fc1 = nn.Linear(embedding_dim, 128, device=device)
        self.fc2 = nn.Linear(128, 1, device=device)
        self.loss = nn.MSELoss()
    
    def forward(self, X):
        """
        X: Can be (batch_size, 4, embedding_dim) or (batch_size, embedding_dim)
        """
        # Flatten to (batch_size, -1) then take only embedding_dim features
        if X.dim() == 3:
            batch_size = X.shape[0]
            X = X.view(batch_size, -1)  # Flatten
            # If we have more features than embedding_dim, average or take first chunk
            if X.shape[1] > self.embedding_dim:
                # Average across the feature dimension
                X = X.view(batch_size, -1, self.embedding_dim).mean(dim=1)
        
        X = torch.relu(self.fc1(X))
        return self.fc2(X).flatten()


class DrillV_Minimal(Drill):
    """
    Minimal: Simplest possible V-learning.
    - Tiny network (2 layers)
    - No dropout, no normalization
    - Basic replay memory
    - High learning rate for fast learning
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DrillV_Minimal"
        
        if self.df_embeddings is not None:
            from ontolearn.learners.drill import DrillVHeuristic
            # Override with minimal network
            self.heuristic_func = DrillVHeuristic(
                model=DrillVNet_Minimal(self.embedding_dim, self.device),
                mode="averaging",
                device=self.device
            )
            self.heuristic_func.name = "DrillV_Minimal"  # Add name attribute
            self.experiences = Experience(maxlen=self.max_len_replay_memory)
            if self.learning_rate:
                self.optimizer = torch.optim.Adam(
                    self.heuristic_func.net.parameters(), 
                    lr=0.01  # High LR
                )
        else:
            self.heuristic_func = CeloeBasedReward()
    
    def form_experiences(self, state_pairs, rewards):
        """Simple experience storage"""
        for (e, e_next), reward in zip(state_pairs, rewards):
            self.experiences.append((e, e_next, reward))
    
    def learn_from_replay_memory(self, gamma=0.9):
        """Dead simple V-learning"""
        if isinstance(self.heuristic_func, CeloeBasedReward):
            return
        
        result = self.experiences.retrieve()
        if len(result) != 3:
            return
        current_states, next_states, rewards = result
        
        N = len(rewards)
        if N == 0:
            return
        
        # Use all data, no sampling
        batch_size = min(128, N)
        indices = random.sample(range(N), batch_size)
        
        current_batch = torch.cat([current_states[i] for i in indices], 0).to(self.device)
        next_batch = torch.cat([next_states[i] for i in indices], 0).to(self.device)
        reward_batch = torch.tensor([rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        
        self.heuristic_func.net.train()
        
        # Single epoch
        v_current = self.heuristic_func.net(current_batch)
        with torch.no_grad():
            v_next = self.heuristic_func.net(next_batch)
        
        target = reward_batch + gamma * v_next
        loss = self.heuristic_func.net.loss(v_current, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.heuristic_func.net.eval()


# ============================================================================
# VARIANT 3: STANDARD - Moderate complexity
# ============================================================================
class DrillVNet_Standard(nn.Module):
    """Moderate network: 3 layers with LayerNorm and light dropout"""
    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        hidden = embedding_dim
        
        self.fc1 = nn.Linear(embedding_dim, hidden, device=device)
        self.ln1 = nn.LayerNorm(hidden, device=device)
        self.dropout = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden, hidden // 2, device=device)
        self.ln2 = nn.LayerNorm(hidden // 2, device=device)
        
        self.fc_out = nn.Linear(hidden // 2, 1, device=device)
        self.loss = nn.MSELoss()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, X):
        """
        X: Can be (batch_size, 4, embedding_dim) or (batch_size, embedding_dim)
        """
        # Flatten to (batch_size, -1) then take only embedding_dim features
        if X.dim() == 3:
            batch_size = X.shape[0]
            X = X.view(batch_size, -1)  # Flatten
            # If we have more features than embedding_dim, average or take first chunk
            if X.shape[1] > self.embedding_dim:
                # Average across the feature dimension
                X = X.view(batch_size, -1, self.embedding_dim).mean(dim=1)
        
        X = self.fc1(X)
        X = self.ln1(X)
        X = torch.relu(X)
        X = self.dropout(X)
        
        X = self.fc2(X)
        X = self.ln2(X)
        X = torch.relu(X)
        
        return self.fc_out(X).flatten()


class DrillV_Standard(Drill):
    """
    Standard: Balanced approach.
    - Moderate network (3 layers, LayerNorm, light dropout)
    - Standard learning rate
    - Multiple epochs per replay
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DrillV_Standard"
        
        if self.df_embeddings is not None:
            from ontolearn.learners.drill import DrillVHeuristic
            self.heuristic_func = DrillVHeuristic(
                model=DrillVNet_Standard(self.embedding_dim, self.device),
                mode="averaging",
                device=self.device
            )
            self.heuristic_func.name = "DrillV_Standard"  # Add name attribute
            self.experiences = Experience(maxlen=self.max_len_replay_memory)
            if self.learning_rate:
                self.optimizer = torch.optim.Adam(
                    self.heuristic_func.net.parameters(), 
                    lr=self.learning_rate
                )
        else:
            self.heuristic_func = CeloeBasedReward()
    
    def form_experiences(self, state_pairs, rewards):
        for (e, e_next), reward in zip(state_pairs, rewards):
            self.experiences.append((e, e_next, reward))
    
    def learn_from_replay_memory(self, gamma=0.95):
        if isinstance(self.heuristic_func, CeloeBasedReward):
            return
        
        result = self.experiences.retrieve()
        if len(result) != 3:
            return
        current_states, next_states, rewards = result
        
        N = len(rewards)
        if N == 0:
            return
        
        batch_size = min(256, N)
        indices = random.sample(range(N), batch_size)
        
        current_batch = torch.cat([current_states[i] for i in indices], 0).to(self.device)
        next_batch = torch.cat([next_states[i] for i in indices], 0).to(self.device)
        reward_batch = torch.tensor([rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        
        self.heuristic_func.net.train()
        total_loss = 0
        
        # Multiple epochs
        for _ in range(self.num_epochs_per_replay):
            v_current = self.heuristic_func.net(current_batch)
            with torch.no_grad():
                v_next = self.heuristic_func.net(next_batch)
            
            target = reward_batch + gamma * v_next
            loss = self.heuristic_func.net.loss(v_current, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.heuristic_func.net.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        if self.verbose > 1:
            print(f"Loss: {total_loss / self.num_epochs_per_replay:.4f}")
        
        self.heuristic_func.net.eval()


# ============================================================================
# VARIANT 4: ENHANCED - Add smart features
# ============================================================================
class DrillV_Enhanced(DrillV_Standard):
    """
    Enhanced: Add proven techniques.
    - Curriculum learning (easy -> hard)
    - Intrinsic curiosity (explore novel states)
    - Lower gamma (focus on immediate rewards)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DrillV_Enhanced"
        self.state_visits = {}
    
    def form_experiences(self, state_pairs, rewards):
        """Add curiosity bonus"""
        for (e, e_next), base_reward in zip(state_pairs, rewards):
            # Curiosity: reward visiting new states
            state_key = str(e_next.concept) if hasattr(e_next, 'concept') else str(e_next)
            visits = self.state_visits.get(state_key, 0)
            self.state_visits[state_key] = visits + 1
            
            # Bonus decreases with visits
            curiosity = 0.1 / ((visits + 1) ** 0.5)
            reward = base_reward + curiosity
            
            self.experiences.append((e, e_next, reward))
    
    def generate_learning_problems_curriculum(self, num_target, num_problems):
        """Sort problems by difficulty"""
        problems = super().generate_learning_problems(num_target, num_problems)
        
        if len(problems) <= 10:
            return problems
        
        # Sort by coverage (easier = more examples)
        def difficulty(prob):
            _, pos, neg = prob
            return -(len(pos) + len(neg))  # Negative for ascending
        
        return sorted(problems, key=difficulty)
    
    def train(self, dataset=None, num_of_target_concepts=1, num_learning_problems=1):
        if isinstance(self.heuristic_func, CeloeBasedReward):
            print("No training...")
            return self.terminate_training()
        
        # Use curriculum
        training_data = self.generate_learning_problems_curriculum(
            num_of_target_concepts, num_learning_problems
        )
        
        for (target, pos, neg) in training_data:
            self.rl_learning_loop(pos_uri=frozenset(pos), neg_uri=frozenset(neg))
            
            self.seen_examples.setdefault(len(self.seen_examples), dict()).update({
                'Concept': target,
                'Positives': [i.str for i in pos],
                'Negatives': [i.str for i in neg]
            })
        
        return self.terminate_training()
    
    def learn_from_replay_memory(self, gamma=0.90):  # Lower gamma
        super().learn_from_replay_memory(gamma=gamma)


# ============================================================================
# VARIANT 5: COMPLEX - All the bells and whistles
# ============================================================================
class DrillVNet_Complex(nn.Module):
    """Complex network: Wider, deeper, with residual connections"""
    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        hidden = embedding_dim * 2  # 2x wider
        
        self.fc1 = nn.Linear(embedding_dim, hidden, device=device)
        self.ln1 = nn.LayerNorm(hidden, device=device)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden, hidden, device=device)
        self.ln2 = nn.LayerNorm(hidden, device=device)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden, hidden // 2, device=device)
        self.ln3 = nn.LayerNorm(hidden // 2, device=device)
        
        self.fc_out = nn.Linear(hidden // 2, 1, device=device)
        
        # Residual projection
        self.residual_proj = nn.Linear(embedding_dim, hidden // 2, device=device)
        
        self.loss = nn.SmoothL1Loss()  # Huber loss
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, X):
        """
        X: Can be (batch_size, 4, embedding_dim) or (batch_size, embedding_dim)
        """
        # Flatten to (batch_size, -1) then take only embedding_dim features
        if X.dim() == 3:
            batch_size = X.shape[0]
            X = X.view(batch_size, -1)  # Flatten
            # If we have more features than embedding_dim, average or take first chunk
            if X.shape[1] > self.embedding_dim:
                # Average across the feature dimension
                X = X.view(batch_size, -1, self.embedding_dim).mean(dim=1)
        
        identity = X
        
        # Deep network with residual
        X = torch.relu(self.ln1(self.fc1(X)))
        X = self.dropout1(X)
        
        X = torch.relu(self.ln2(self.fc2(X)))
        X = self.dropout2(X)
        
        X = torch.relu(self.ln3(self.fc3(X)))
        
        # Add residual connection
        residual = self.residual_proj(identity)
        X = X + residual
        
        return self.fc_out(X).flatten()


class DrillV_Complex(KBDrivenOptimizationMixin, DrillV_Enhanced):
    """
    Complex: Everything including the kitchen sink.
    - Complex network (4 layers, residual, heavy dropout)
    - Target network for stability
    - Learning rate scheduling
    - All smart features from Enhanced
    - **NEW: Intelligent RL-based termination (agent decides when to stop)**
    - **NEW: KB-driven initialization and early pruning (optional)**
    """
    def __init__(self, *args, termination_epsilon=0.3, enable_pruning=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DrillV_Complex"
        self.enable_pruning = enable_pruning  # Flag to control KB-driven pruning
        # Import and initialize V-learning termination agent
        from rl_termination_module import IntelligentTerminationAgent
        self.termination_agent = IntelligentTerminationAgent(
            learning_rate=0.001,            # V-network learning rate
            gamma=0.95,                     # Discount for future improvements
            epsilon=termination_epsilon,    # Exploration rate (user configurable!)
            min_quality_threshold=0.75,     # Minimum acceptable quality
            min_concepts_explored=1,      # Safety minimum
            max_concepts_explored=self.max_num_of_concepts_tested+1 if hasattr(self, 'max_num_of_concepts_tested') else 15000,     # Safety maximum
            memory_path='termination_agent_memory.pkl'  # Persistent memory file
        )
        
        print(f"Termination epsilon: {termination_epsilon} "
              f"(0=always exploit, 1=always explore)")
        
        if self.df_embeddings is not None:
            from ontolearn.learners.drill import DrillVHeuristic
            self.heuristic_func = DrillVHeuristic(
                model=DrillVNet_Complex(self.embedding_dim, self.device),
                mode="averaging",
                device=self.device
            )
            self.heuristic_func.name = "DrillV_Complex"  # Add name attribute
            
            if self.learning_rate:
                self.optimizer = torch.optim.AdamW(
                    self.heuristic_func.net.parameters(),
                    lr=self.learning_rate,
                    weight_decay=1e-5
                )
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5
                )
            
            # Target network
            self.target_net = DrillVNet_Complex(self.embedding_dim, self.device)
            self.target_net.load_state_dict(self.heuristic_func.net.state_dict())
            self.target_net.eval()
            self.update_counter = 0
    
    def reset_for_new_lp(self):
        """
        Reset agent for a new learning problem.
        Deletes memory file so each LP starts fresh.
        """
        import os
        memory_file = 'termination_agent_memory.pkl'
        if os.path.exists(memory_file):
            os.remove(memory_file)
            print(f"Deleted memory file for new LP")
        
        # Reset agent's internal state
        self.termination_agent.total_runs = 0
        self.termination_agent.best_ever_quality = 0.0
    
    def is_simple_named_concept(self, concept: OWLClassExpression) -> bool:
        """
        Check if a concept is just a simple named class (e.g., Female, Male)
        vs a complex concept (e.g., Female ⊓ hasSibling(Parent)).
        
        Simple concepts should only be accepted if they have F1=1.0,
        otherwise we should continue refining.
        """
        from owlapy.class_expression import OWLClass
        return isinstance(concept, OWLClass)
    
    def should_accept_best_concept(self, best_quality: float) -> bool:
        """
        Decide if we should accept the current best concept or keep refining.
        
        Logic:
        - If best is a simple named class (e.g., Female) and F1 < 1.0 → Keep refining
        - If best is a complex concept (e.g., Female ⊓ ...) → Can accept if quality is good
        - If any concept has F1 = 1.0 → Accept immediately
        """
        if best_quality >= 1.0:
            return True  # Perfect solution, accept
        
        # Check if current best is a simple named concept
        try:
            best_node = self.best_hypotheses(n=1, return_node=True)
            if best_node and self.is_simple_named_concept(best_node.concept):
                # Simple concept with F1 < 1.0 → Don't accept, keep refining
                if self.verbose > 1:
                    print(f"   Best is simple concept '{owl_expression_to_dl(best_node.concept)}' "
                          f"with F1={best_quality:.3f} < 1.0 → Continue refining")
                return False
        except:
            # If we can't get best node, use normal termination logic
            pass
        
        # Complex concept or couldn't determine → Use normal termination logic
        return True
    
    def compute_quality_of_class_expression(self, state):
        """Override to feed info to V-learning termination agent"""
        super().compute_quality_of_class_expression(state)
        
        # Let V-learning agent observe the quality trajectory
        self.termination_agent.observe_quality(state.quality)
        
        # Debug: Print progress periodically
        if self.verbose > 0 and self.number_of_tested_concepts % 50 == 0:
            stats = self.termination_agent.get_statistics()
            print(f"   Progress: {self.number_of_tested_concepts} concepts | Best: {stats['best_quality']:.3f} | Run #{stats['total_runs'] + 1}")
    
    def fit(self, learning_problem, max_runtime=None):
        """Override fit to use V-learning termination agent"""
        import time
        from collections import Counter
        from itertools import chain
        from ontolearn.utils.static_funcs import make_iterable_verbose
        from owlapy import owl_expression_to_dl
        
        if max_runtime:
            assert isinstance(max_runtime, float) or isinstance(max_runtime, int)
            self.max_runtime = max_runtime
        
        # Reset agent for NEW episode (keeps learned V-function!)
        self.termination_agent.reset_for_new_episode()
        
        # Standard initialization
        self.clean()
        self.start_time = time.time()
        
        # KB-driven initialization (optional - controlled by enable_pruning flag)
        if self.enable_pruning:
            # NEW APPROACH: KB-driven starting concepts
            kb_driven_concepts = self.extract_kb_driven_starting_concepts(
                pos_examples=learning_problem.pos,
                neg_examples=learning_problem.neg,
                max_concepts=20
            )
        else:
            # FALLBACK: Use type bias (old approach)
            pos_type_counts = Counter(
                [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.pos))])
            neg_type_counts = Counter(
                [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.neg))])
            type_bias = pos_type_counts - neg_type_counts
            kb_driven_concepts = list(type_bias.keys())[:20]
        
        root_state = self.initialize_training_class_expression_learning_problem(
            pos=learning_problem.pos, neg=learning_problem.neg)
        self.operator.set_input_examples(pos=learning_problem.pos, neg=learning_problem.neg)
        assert root_state.quality > 0, f"Root state {root_state} must have quality > 0"
        
        root_state.heuristic = root_state.quality
        self.search_tree.add(root_state)
        best_found_quality = 0
        
        # Add KB-driven concepts to search tree
        for x in (self.create_rl_state(i, parent_node=root_state) for i in kb_driven_concepts):
            self.compute_quality_of_class_expression(x)
            x.heuristic = x.quality
            if x.quality > best_found_quality:
                best_found_quality = x.quality
                self.search_tree.add(x)
        
        if self.verbose > 0:
            init_method = "KB-driven" if self.enable_pruning else "type bias"
            print(f"✓ Initialized with {len(kb_driven_concepts)} {init_method} concepts "
                  f"(best quality: {best_found_quality:.3f})")
        
        # Main loop with intelligent agent-based termination
        for iteration in make_iterable_verbose(range(0, self.iter_bound),
                                              verbose=self.verbose,
                                              desc=f"DrillV Complex with Intelligent Termination"):
            assert len(self.search_tree) > 0, "Search Tree cannot be empty!"
            self.search_tree.show_current_search_tree()
            
            # AGENT DECISION: Should I stop exploring?
            # BUT: Enforce minimum exploration using actual concepts tested counter
            if self.number_of_tested_concepts < self.termination_agent.min_concepts_explored:
                # Force exploration - don't let agent stop yet
                pass
            else:
                # NEW CHECK: Don't stop if best is a simple named concept with F1 < 1.0
                if not self.should_accept_best_concept(best_found_quality):
                    if self.verbose > 1:
                        print("   Forcing continued refinement of simple concept...")
                else:
                    should_stop, reason, confidence = self.termination_agent.should_stop_exploring(verbose=self.verbose)
                    if should_stop:
                        if self.verbose > 0:
                            print(f"\n🤖 Agent decided to stop: {reason} (confidence: {confidence:.2f})")
                            stats = self.termination_agent.get_statistics()
                            print(f"   Best quality found: {stats['best_quality']:.4f}")
                            print(f"   Concepts explored: {self.number_of_tested_concepts}")
                        # CRITICAL: Learn from episode before terminating!
                        self.termination_agent.learn_from_episode()
                        return self.terminate()
            
            # Standard time check (safety fallback)
            if time.time() - self.start_time > self.max_runtime:
                if self.verbose > 0:
                    print(f"\n⏱ Time limit reached ({self.max_runtime}s)")
                # CRITICAL: Learn from episode even if time limit reached!
                self.termination_agent.learn_from_episode()
                return self.terminate()
            
            if self.max_num_of_concepts_tested is not None and self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
            
            # Get most promising state
            most_promising = self.next_node_to_expand()
            next_possible_states = []
            
            # Refine and evaluate
            for ref in make_iterable_verbose(self.apply_refinement(most_promising),
                                            verbose=self.verbose,
                                            position=0, leave=True):
                # Check agent decision during refinement (but respect minimum exploration)
                if self.number_of_tested_concepts >= self.termination_agent.min_concepts_explored:
                    # NEW: Only check termination if we should accept current best
                    if self.should_accept_best_concept(best_found_quality):
                        should_stop, _, _ = self.termination_agent.should_stop_exploring(verbose=0)
                        if should_stop:
                            break
                
                # Time check
                if time.time() - self.start_time > self.max_runtime:
                    break
                
                # KB-DRIVEN OPTIMIZATION: Early pruning of clearly irrelevant concepts (optional)
                if self.enable_pruning:
                    if self.should_prune_refinement(ref.concept, learning_problem.pos, learning_problem.neg):
                        if self.verbose > 1:
                            print(f"Pruned: {owl_expression_to_dl(ref.concept)} (low KB coverage)")
                        continue
                
                self.compute_quality_of_class_expression(ref)
                if ref.quality == 0:
                    continue
                
                if ref.quality > best_found_quality:
                    if self.verbose > 0:
                        print(f"\n✨ Best: {owl_expression_to_dl(ref.concept)} | Q: {ref.quality:.4f}")
                    best_found_quality = ref.quality
                
                next_possible_states.append(ref)
                
                if self.stop_at_goal and ref.quality == 1.0:
                    break
            
            if not next_possible_states:
                continue
            
            # Predict values and update search tree
            if self.df_embeddings is not None:
                preds = self.predict_values(current_state=most_promising,
                                          next_states=next_possible_states)
            else:
                preds = None
            
            self.goal_found = self.update_search(next_possible_states, preds)
            if self.goal_found and self.stop_at_goal:
                if self.terminate_on_goal:
                    # IMPORTANT: Learn from this episode before terminating!
                    self.termination_agent.learn_from_episode()
                    return self.terminate()
        
        # IMPORTANT: Learn from this complete episode!
        # This is where V-learning happens - agent gets smarter for next run
        self.termination_agent.learn_from_episode()
        
        return self.terminate()
    
    def learn_from_replay_memory(self, gamma=0.95):
        """V-learning with target network"""
        if isinstance(self.heuristic_func, CeloeBasedReward):
            return
        
        result = self.experiences.retrieve()
        if len(result) != 3:
            return
        current_states, next_states, rewards = result
        
        N = len(rewards)
        if N == 0:
            return
        
        batch_size = min(256, N)
        indices = random.sample(range(N), batch_size)
        
        current_batch = torch.cat([current_states[i] for i in indices], 0).to(self.device)
        next_batch = torch.cat([next_states[i] for i in indices], 0).to(self.device)
        reward_batch = torch.tensor([rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        
        self.heuristic_func.net.train()
        total_loss = 0
        
        for _ in range(self.num_epochs_per_replay):
            v_current = self.heuristic_func.net(current_batch)
            
            # Use target network
            with torch.no_grad():
                v_next = self.target_net(next_batch)
            
            target = reward_batch + gamma * v_next
            loss = self.heuristic_func.net.loss(v_current, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.heuristic_func.net.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_net.load_state_dict(self.heuristic_func.net.state_dict())
        
        # LR scheduling
        avg_loss = total_loss / self.num_epochs_per_replay
        self.scheduler.step(avg_loss)
        
        if self.verbose > 1:
            print(f"Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self.heuristic_func.net.eval()


# ============================================================================
# SUMMARY
# ============================================================================
DRILLV_VARIANTS = {
    'baseline': DrillV_Baseline,
    'minimal': DrillV_Minimal,
    'standard': DrillV_Standard,
    'enhanced': DrillV_Enhanced,
    'complex': DrillV_Complex,
}

VARIANT_DESCRIPTIONS = {
    'baseline': 'No RL - just use quality as heuristic',
    'minimal': 'Simplest NN (2 layers, high LR, 1 epoch)',
    'standard': 'Balanced (3 layers, LayerNorm, multi-epoch)',
    'enhanced': 'Standard + curriculum + curiosity',
    'complex': 'Everything (deep network, target net, scheduling) + INTELLIGENT TERMINATION (agent decides when to stop)',
}
