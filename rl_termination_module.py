"""
TRUE V-LEARNING for Intelligent Termination

The agent learns a VALUE FUNCTION that predicts:
"What's the expected improvement if I continue exploring?"

Key idea:
- V(state) = Expected future quality gain from continuing
- Agent learns from experience: "When I was in state S and continued, 
  did I find better solutions?"
- Gets SMARTER with each run on the same LP
- Uses learned V-function to decide when to stop

This is TRUE RL V-Learning!
"""

from time import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import pickle
import os


class TerminationVNet(nn.Module):
    """
    Neural network that learns to predict the value of continuing exploration.
    
    Input: Current search state features
    Output: V(continue) - expected quality improvement if we keep searching
    
    IMPORTANT: Initialized with POSITIVE BIAS to be optimistic by default!
    (Untrained network should say "continue exploring" not "stop")
    """
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
        # OPTIMISTIC INITIALIZATION: Bias the output to predict positive values
        # (Untrained network should encourage exploration, not stop immediately)
        with torch.no_grad():
            self.fc3.bias.fill_(0.1)  # Start with prediction: "continuing gives +0.1 improvement"
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class IntelligentTerminationAgent:
    """
    V-Learning agent that learns WHEN to stop exploring.
    
    Gets smarter with each run on the same LP:
    - Run 1: May not find optimal, but learns the trajectory
    - Run 2: Uses learned V-function to make better decisions
    - Run 3+: Increasingly efficient at finding good solutions quickly
    """
    
    def __init__(self, 
                 learning_rate=0.001,
                 gamma=0.95,                      # Discount factor for future improvements
                 epsilon=0.3,                     # Exploration rate (allow some risky decisions)
                 min_quality_threshold=0.75,      # Minimum acceptable quality
                 min_concepts_explored=10,       # Safety minimum
                 max_concepts_explored=5000,      # Safety maximum
                 memory_path='termination_agent_memory.pkl'):
        
        # V-Learning network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.v_net = TerminationVNet(input_dim=10).to(self.device)
        self.optimizer = torch.optim.Adam(self.v_net.parameters(), lr=learning_rate)
        
        # RL hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_quality_threshold = min_quality_threshold
        self.min_concepts_explored = min_concepts_explored
        self.max_concepts_explored = max_concepts_explored
        
        # Experience replay for this LP
        self.experience_buffer = []
        
        # Current episode tracking
        self.quality_history = []
        self.concepts_explored_count = 0
        self.best_quality = 0.0
        self.iterations_without_improvement = 0
        
        # Persistent memory across runs on SAME LP
        self.memory_path = memory_path
        self.total_runs = 0  # How many times have we run on this LP?
        self.best_ever_quality = 0.0  # Best we've ever achieved on this LP
        
        # Training metrics tracking
        self.training_losses = []  # Track V-network training loss
        self.episode_qualities = []  # Track quality per episode
        self.episode_concepts = []  # Track concepts explored per episode
        self.v_predictions = []  # Track V-network predictions over time
        
        # Epsilon-greedy: Decide ONCE per episode, not per check!
        self.episode_explore_mode = False
        
        # Load previous experience if exists
        self._load_memory()
        
        self.termination_reason = None
    
    def _load_memory(self):
        """Load agent's memory from previous runs on this LP"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'rb') as f:
                    memory = pickle.load(f)
                    self.total_runs = memory.get('total_runs', 0)
                    self.best_ever_quality = memory.get('best_ever_quality', 0.0)
                    
                    # Load V-network weights if available
                    if 'v_net_state' in memory:
                        self.v_net.load_state_dict(memory['v_net_state'])
                    
                    print(f"Loaded agent memory: {self.total_runs} previous runs, best quality: {self.best_ever_quality:.4f}")
            except:
                print("Could not load agent memory, starting fresh")
    
    def _save_memory(self):
        """Save agent's learned knowledge for next run"""
        memory = {
            'total_runs': self.total_runs,
            'best_ever_quality': self.best_ever_quality,
            'v_net_state': self.v_net.state_dict()
        }
        with open(self.memory_path, 'wb') as f:
            pickle.dump(memory, f)
    
    def _extract_state_features(self):
        """
        Extract features that describe current search state.
        
        These features should generalize across runs!
        Not absolute values, but PATTERNS.
        """
        if len(self.quality_history) < 10:
            # Not enough data yet
            return torch.zeros(10, device=self.device)
        
        recent = self.quality_history[-50:] if len(self.quality_history) >= 50 else self.quality_history
        
        # Use max(current quality, best_ever) to avoid division issues
        reference_quality = max(self.best_ever_quality, max(recent), 0.01)
        
        features = [
            # Quality patterns (relative to reference)
            max(recent) / reference_quality,  # Current relative to best known
            np.mean(recent),
            np.std(recent),
            (max(recent) - min(recent)),  # Quality range
            
            # Improvement patterns
            self.iterations_without_improvement / 100.0,
            (self.best_quality - np.mean(recent[-10:])) if len(recent) >= 10 else 0,
            
            # Exploration efficiency
            min(self.concepts_explored_count / 1000.0, 1.0),  # Normalized, capped at 1.0
            self.best_quality / (self.concepts_explored_count + 1),  # Quality per concept
            
            # Trend
            np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 5 else 0,  # Slope
            
            # Experience (normalize better)
            min(self.total_runs / 10.0, 1.0)  # How experienced are we with this LP?
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def observe_quality(self, quality):
        """Agent observes the quality of current concept"""
        self.quality_history.append(quality)
        self.concepts_explored_count += 1
        
        if quality > self.best_quality:
            self.best_quality = quality
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
    
    def should_stop_exploring(self, verbose=0):
        """
        V-Learning Decision: Should I continue exploring?
        
        Uses learned V-function to estimate: "If I continue, will I find better?"
        
        Returns:
            (should_stop: bool, reason: str, confidence: float)
        """
        # CRITICAL: First run(s) should explore fully to gather training data!
        # Don't trust untrained V-network
        if self.total_runs == 0:
            # First run ever - explore fully until max or time limit
            if self.concepts_explored_count >= self.max_concepts_explored:
                if verbose > 0:
                    print(f"\n First run: Maximum exploration reached ({self.max_concepts_explored})")
                return True, "First run: max concepts", 1.0
            if verbose > 1 and self.concepts_explored_count % 100 == 0:
                print(f"   First run: exploring... ({self.concepts_explored_count} concepts so far)")
            return False, "First run: gathering training data", 0.0
        
        # Safety: Enforce minimum exploration
        if self.concepts_explored_count < self.min_concepts_explored:
            if verbose > 0 and self.concepts_explored_count % 50 == 0:
                print(f"   Agent: Minimum exploration ({self.concepts_explored_count}/{self.min_concepts_explored})")
            return False, "Minimum exploration", 0.0
        
        # Safety: Maximum exploration (give up)
        if self.concepts_explored_count >= self.max_concepts_explored:
            if verbose > 0:
                print(f"\n Agent: Maximum exploration reached")
            return True, "Maximum concepts", 1.0
        
        # Minimum data for V-function
        if len(self.quality_history) < 20:
            return False, "Gathering data", 0.0
        
        # Extract current state features
        state_features = self._extract_state_features().unsqueeze(0)
        
        # V-Learning decision: Predict value of continuing
        with torch.no_grad():
            v_continue = self.v_net(state_features).item()
        
        if verbose > 1:
            print(f" V-network prediction: {v_continue:.4f} (based on {self.total_runs} previous runs)")
        
        # Epsilon-greedy: Use episode-level exploration mode (decided once at start)
        explore = self.episode_explore_mode
        
        # Decision threshold - adapt based on experience
        # After first run, we have real data about what "good" looks like
        if self.total_runs >= 2:
            # We have experience - be more aggressive
            # Allow NEGATIVE predictions to stop (continuing is worse than stopping)
            threshold = 0.0  # Stop if V(continue) < 0
        elif self.best_ever_quality >= 0.85:
            # First experience run found great solution - be moderately strict
            threshold = 0.01
        else:
            # First experience run, solution was mediocre - be conservative
            threshold = 0.02
        
        # Additional safety: Don't stop if quality is still low
        if self.best_quality < self.min_quality_threshold:
            if verbose > 1:
                print(f"   Quality {self.best_quality:.4f} < threshold {self.min_quality_threshold:.4f} → Continue")
            return False, "Quality below threshold", 0.0
        
        # CRITICAL: NEVER stop if quality hasn't matched best_ever.
        # This guarantees RL termination never degrades solution quality.
        if self.best_ever_quality > 0:
            if self.best_quality < self.best_ever_quality:
                if verbose > 1:
                    print(f"   Quality {self.best_quality:.4f} < best_ever {self.best_ever_quality:.4f} → Continue")
                return False, "Quality hasn't matched best_ever yet", 0.0
        
        # Log V-network prediction periodically (every 100 concepts) for debugging
        if verbose > 1 and self.concepts_explored_count % 100 == 0:
            print(f"   [Concept {self.concepts_explored_count}] V(continue)={v_continue:.4f}, threshold={threshold:.4f}, explore={explore}")
        
        # Decision logic
        if v_continue < threshold and not explore:
            # V-function says: "Not worth continuing"
            self.termination_reason = "V-function: Low expected improvement"
            if verbose > 0:
                print(f"\n V-Learning Decision: STOP")
                print(f"   V(continue) = {v_continue:.4f} < threshold {threshold:.4f}")
                print(f"   Run #{self.total_runs + 1}, Best so far: {self.best_quality:.4f}")
            return True, self.termination_reason, 1.0 - v_continue
        
        # Also stop if quality is really good and not improving
        if self.best_quality >= 0.9999 and self.iterations_without_improvement > 100:
            if verbose > 0:
                print(f"\n Agent: Excellent quality reached ({self.best_quality:.4f})")
            return True, "Excellent quality found", 0.9999
        
        # Continue exploring
        if verbose > 1 and self.concepts_explored_count % 50 == 0:
            print(f"   V(continue)={v_continue:.4f}, threshold={threshold:.4f} → Continue")
        return False, "V-function: Expected improvement", v_continue
    
    def learn_from_episode(self):
        """
        After episode ends, learn from the trajectory.
        
        This is where V-Learning happens!
        Agent learns: "When I was in state S and continued, 
                      how much did my quality actually improve?"
        """
        print(f"\n learn_from_episode called! Quality history length: {len(self.quality_history)}")
        
        if len(self.quality_history) < 20:
            print(f"Not enough data to learn (need 20, have {len(self.quality_history)})")
            return  # Not enough data
        
        # Compute actual returns with efficiency reward
        # Key insight: We want to maximize quality WHILE minimizing concepts explored
        # Find the "sweet spot" where we achieved good quality with minimal exploration
        returns = []
        final_quality = self.quality_history[-1]
        total_concepts = len(self.quality_history)
        
        # Find optimal stopping point: first time we reach within 1% of final quality
        optimal_stop = total_concepts
        for t in range(len(self.quality_history)):
            if self.quality_history[t] >= final_quality - 0.01:
                optimal_stop = t
                break
        
        for t in range(len(self.quality_history)):
            # Quality improvement if we continue from here
            future_best = max(self.quality_history[t:])
            current = self.quality_history[t]
            quality_gain = future_best - current
            
            # Efficiency reward: Penalize exploring beyond optimal stopping point
            if t < optimal_stop:
                # Before optimal: reward = quality gained
                value = quality_gain
            elif quality_gain < 0.001:
                # After optimal with no improvement: STRONG penalty for wasting time
                wasteful_concepts = t - optimal_stop
                penalty = -0.2 * (wasteful_concepts / total_concepts)
                value = penalty
            else:
                # After optimal but still improving: mild penalty
                value = quality_gain - 0.05
            
            returns.append(value)
        
        # Train V-network on this experience
        self.v_net.train()
        episode_losses = []  # Track losses for this episode
        
        for t in range(0, len(self.quality_history) - 1, 10):  # Sample every 10 steps
            # Get state at time t
            old_history = self.quality_history[:t+1]
            if len(old_history) < 10:
                continue
            
            # Temporarily set history to compute features
            temp_history = self.quality_history
            temp_count = self.concepts_explored_count
            temp_best = self.best_quality
            temp_iters = self.iterations_without_improvement
            
            self.quality_history = old_history
            self.concepts_explored_count = t + 1
            self.best_quality = max(old_history)
            self.iterations_without_improvement = t - old_history.index(max(old_history))
            
            state_features = self._extract_state_features().unsqueeze(0)
            
            # Restore
            self.quality_history = temp_history
            self.concepts_explored_count = temp_count
            self.best_quality = temp_best
            self.iterations_without_improvement = temp_iters
            
            # Actual return
            actual_return = torch.tensor([returns[t]], dtype=torch.float32, device=self.device)
            
            # Predicted V-value
            predicted_v = self.v_net(state_features)
            
            # V-Learning update (TD learning)
            loss = nn.MSELoss()(predicted_v, actual_return)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            episode_losses.append(loss.item())
        
        self.v_net.eval()
        
        # Store metrics for this episode
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            self.training_losses.append(avg_loss)
            print(f"   V-network training loss: {avg_loss:.6f}")
        
        self.episode_qualities.append(self.best_quality)
        self.episode_concepts.append(self.concepts_explored_count)
        
        # Update memory BEFORE incrementing total_runs
        if self.best_quality > self.best_ever_quality:
            self.best_ever_quality = self.best_quality
            print(f"New best quality! {self.best_ever_quality:.4f}")
        
        self.total_runs += 1
        
        print(f"Saving memory to: {self.memory_path}")
        self._save_memory()
        
        import os
        if os.path.exists(self.memory_path):
            size = os.path.getsize(self.memory_path)
            print(f"Memory file saved successfully! Size: {size} bytes")
        else:
            print(f"WARNING: Memory file was NOT created!")
        
        print(f"V-Learning complete! Run #{self.total_runs}, Best this run: {self.best_quality:.4f}, Best ever: {self.best_ever_quality:.4f}")
    
    def reset_for_new_episode(self):
        """Reset episode-specific state (but keep learned V-function!)"""
        self.quality_history = []
        self.concepts_explored_count = 0
        self.best_quality = 0.0
        self.iterations_without_improvement = 0
        self.termination_reason = None
        
        # Epsilon-greedy: Decide exploration mode ONCE per episode
        # Use the user-configured epsilon (self.epsilon) with optional decay
        # Decay formula: epsilon * (decay_rate ** total_runs)
        # decay_rate = 1.0 means NO decay (constant epsilon)
        # decay_rate < 1.0 means epsilon decreases with experience
        decay_rate = 0.95  # Mild decay: 1.0 -> 0.95 -> 0.90 -> 0.86...
        current_epsilon = self.epsilon #max(0.001, self.epsilon * (decay_rate ** self.total_runs))
       
        self.episode_explore_mode = np.random.random() < current_epsilon
        if self.episode_explore_mode:
            print(f"Episode mode: EXPLORE (ε={current_epsilon:.3f}, base={self.epsilon:.2f}, runs={self.total_runs})")
        else:
            print(f"Episode mode: EXPLOIT V-network (ε={current_epsilon:.3f}, base={self.epsilon:.2f}, runs={self.total_runs})")
    
    def get_statistics(self):
        """Get agent's current state for debugging"""
        return {
            'best_quality': self.best_quality,
            'iterations_without_improvement': self.iterations_without_improvement,
            'concepts_explored': self.concepts_explored_count,
            'quality_history_len': len(self.quality_history),
            'termination_reason': self.termination_reason,
            'total_runs': self.total_runs,
            'best_ever_quality': self.best_ever_quality
        }
    
    def get_training_metrics(self):
        """Get training metrics for visualization"""
        return {
            'training_losses': self.training_losses,
            'episode_qualities': self.episode_qualities,
            'episode_concepts': self.episode_concepts,
            'v_predictions': self.v_predictions,
            'total_runs': self.total_runs
        }
    
    def save_training_metrics(self, filepath='training_metrics.json'):
        """Save training metrics to file"""
        import json
        metrics = self.get_training_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Training metrics saved to: {filepath}")
    
    def plot_training_progress(self, save_path='training_progress.png'):
        """
        Visualize training progress of the V-network.
        
        Shows:
        1. V-network training loss over episodes
        2. Quality achieved per episode
        3. Concepts explored per episode
        4. Learning efficiency (quality/concepts)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_losses:
                print("No training data to plot yet!")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('DrillV Agent Training Progress', fontsize=16, fontweight='bold')
            
            episodes = list(range(1, len(self.training_losses) + 1))
            
            # Plot 1: V-Network Training Loss
            axes[0, 0].plot(episodes, self.training_losses, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0, 0].set_xlabel('Episode', fontsize=12)
            axes[0, 0].set_ylabel('V-Network Loss (MSE)', fontsize=12)
            axes[0, 0].set_title('V-Network Learning Curve', fontsize=13, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')  # Log scale to see improvements better
            
            # Add trend line
            if len(episodes) > 1:
                z = np.polyfit(episodes, self.training_losses, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=1.5, label='Trend')
                axes[0, 0].legend()
            
            # Plot 2: Quality Achieved per Episode
            axes[0, 1].plot(episodes, self.episode_qualities, 'g-', linewidth=2, marker='s', markersize=4)
            axes[0, 1].set_xlabel('Episode', fontsize=12)
            axes[0, 1].set_ylabel('Best Quality (F1)', fontsize=12)
            axes[0, 1].set_title('Quality per Episode', fontsize=13, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=np.mean(self.episode_qualities), color='r', linestyle='--', 
                              alpha=0.7, label=f'Mean: {np.mean(self.episode_qualities):.3f}')
            axes[0, 1].legend()
            axes[0, 1].set_ylim([0, 1.0])
            
            # Plot 3: Concepts Explored per Episode
            axes[1, 0].plot(episodes, self.episode_concepts, 'orange', linewidth=2, marker='^', markersize=4)
            axes[1, 0].set_xlabel('Episode', fontsize=12)
            axes[1, 0].set_ylabel('Concepts Explored', fontsize=12)
            axes[1, 0].set_title('Exploration Efficiency', fontsize=13, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=np.mean(self.episode_concepts), color='r', linestyle='--',
                              alpha=0.7, label=f'Mean: {np.mean(self.episode_concepts):.0f}')
            axes[1, 0].legend()
            
            # Plot 4: Learning Efficiency (Quality per Concept)
            if self.episode_concepts:
                efficiency = [q / max(c, 1) for q, c in zip(self.episode_qualities, self.episode_concepts)]
                axes[1, 1].plot(episodes, efficiency, 'purple', linewidth=2, marker='D', markersize=4)
                axes[1, 1].set_xlabel('Episode', fontsize=12)
                axes[1, 1].set_ylabel('Quality / Concepts', fontsize=12)
                axes[1, 1].set_title('Learning Efficiency', fontsize=13, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add trend line
                if len(episodes) > 1:
                    z = np.polyfit(episodes, efficiency, 1)
                    p = np.poly1d(z)
                    trend = p(episodes)
                    axes[1, 1].plot(episodes, trend, "r--", alpha=0.8, linewidth=1.5, label='Trend')
                    
                    # Show if efficiency is improving
                    if z[0] > 0:
                        axes[1, 1].text(0.05, 0.95, '↗ Improving', transform=axes[1, 1].transAxes,
                                       fontsize=11, verticalalignment='top', color='green', fontweight='bold')
                    else:
                        axes[1, 1].text(0.05, 0.95, '↘ Declining', transform=axes[1, 1].transAxes,
                                       fontsize=11, verticalalignment='top', color='red', fontweight='bold')
                    axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Training progress plot saved to: {save_path}")
            
            return fig
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error creating plot: {e}")
