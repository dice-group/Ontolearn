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

"""
NERO - Neural Class Expression Learning with Reinforcement.

This module implements NERO, a neural-symbolic concept learner that combines
neural networks with symbolic reasoning for OWL class expression learning.
"""

from typing import Dict, List, Set, Tuple, Optional
import time
import torch
from owlapy import dl_to_owl_expression

from owlapy.class_expression import OWLThing
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.nero_architectures import DeepSet, SetTransformerNet
from ontolearn.nero_utils import SearchTree, TargetClassExpression
from ontolearn.refinement_operators import NERORefinement
from ontolearn.utils.static_funcs import compute_f1_score

# =============================================================================
# NERO Main Class
# =============================================================================

class NERO:
    """
    NERO - Neural Class Expression Learning with Reinforcement.

    NERO combines neural networks with symbolic reasoning for learning OWL class expressions.
    It uses set-based neural architectures (DeepSet or SetTransformer) to predict quality scores
    for candidate class expressions.

    Args:
        knowledge_base: The knowledge base to learn from
        num_embedding_dim: Dimensionality of entity embeddings (default: 50)
        neural_architecture: Neural architecture to use ('DeepSet' or 'SetTransformer', default: 'DeepSet')
        learning_rate: Learning rate for training (default: 0.001)
        num_epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training (default: 32)
        num_workers: Number of workers for data loading (default: 4)
        quality_func: Quality function for evaluating expressions (default: F1-score)
        max_runtime: Maximum runtime in seconds (default: None)
        verbose: Verbosity level (default: 0)
    """

    name = 'NERO'

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 namespace = None,
                 num_embedding_dim: int = 50,
                 neural_architecture: str = 'DeepSet',
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 quality_func=None,
                 max_runtime: Optional[int] = 10,
                 verbose: int = 0):

        self.kb = knowledge_base
        self.ns = namespace
        self.num_embedding_dim = num_embedding_dim
        self.neural_architecture = neural_architecture
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_runtime = max_runtime
        self.verbose = verbose
        self.search_tree = SearchTree()
        self.refinement_op = None

        # Quality function
        if quality_func is None:
            self.quality_func = compute_f1_score
        else:
            self.quality_func = quality_func

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model components (initialized during training)
        self.model = None
        self.instance_idx_mapping = None
        self.idx_to_instance_mapping = None
        self.target_class_expressions = None
        self.expression = {}
        self._best_predictions = None

        # Training state
        self._is_trained = False

        if self.verbose > 0:
            print(f"NERO initialized with {self.neural_architecture} architecture")
            print(f"Device: {self.device}")

    def _initialize_instance_mapping(self):
        """Initialize mapping from individuals to indices."""
        self.instance_idx_mapping = {
            ind.str: i
            for i, ind in enumerate(self.kb.individuals())
        }
        self.idx_to_instance_mapping = {
            v: k for k, v in self.instance_idx_mapping.items()
        }

    def _initialize_refinement_operator(self):
        """Initialize the refinement operator."""
        if self.refinement_op is None:
            if self.verbose > 0:
                print("Initializing refinement operator...")
            self.refinement_op = NERORefinement(self.kb)
            self.expression.update(self.refinement_op.expressions)

    def _extract_target_expressions(self) -> List[TargetClassExpression]:
        """Extract target class expressions from the knowledge base."""
        renderer = DLSyntaxObjectRenderer()
        target_expressions = []

        # Get all named classes
        for idx, owl_class in enumerate(self.kb.ontology.classes_in_signature()):
            individuals = set(
                ind.str
                for ind in self.kb.individuals(owl_class)
            )
            idx_individuals = set(
                self.instance_idx_mapping[iri]
                for iri in individuals
            )

            target_exp = TargetClassExpression(
                label_id=idx,
                name=renderer.render(owl_class),
                str_individuals=individuals,
                idx_individuals=idx_individuals,
                expression_chain=[renderer.render(OWLThing)],
                length=1,
                _type='atomic_expression'
            )
            target_expressions.append(target_exp)

        return target_expressions

    def _create_model(self, num_outputs: int) -> torch.nn.Module:
        """Create the neural model based on architecture choice."""
        num_instances = len(self.instance_idx_mapping)

        if self.neural_architecture == 'DeepSet':
            model = DeepSet(
                num_instances=num_instances,
                num_embedding_dim=self.num_embedding_dim,
                num_outputs=num_outputs
            )
        elif self.neural_architecture == 'SetTransformer':
            model = SetTransformerNet(
                num_instances=num_instances,
                num_embedding_dim=self.num_embedding_dim,
                num_outputs=num_outputs
            )
        else:
            raise ValueError(f"Unknown architecture: {self.neural_architecture}")

        return model

    def train(self, learning_problems: List[Tuple[List[str], List[str]]]):
        """
        Train the NERO model on learning problems.

        Args:
            learning_problems: List of (positive_examples, negative_examples) tuples
        """
        if self.verbose > 0:
            print("Training NERO model...")

        start_time = time.time()

        # Initialize mappings
        self._initialize_instance_mapping()

        # Extract target expressions
        self.target_class_expressions = self._extract_target_expressions()

        if len(self.target_class_expressions) == 0:
            raise ValueError("No target class expressions found in knowledge base")

        # Create model
        num_outputs = len(self.target_class_expressions)
        self.model = self._create_model(num_outputs)
        self.model.to(self.device)
        self.model.train()

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_func = torch.nn.MSELoss()

        # Convert learning problems to tensors
        X_pos_list, X_neg_list, Y_list = [], [], []

        for pos_examples, neg_examples in learning_problems:
            pos_idx = [self.instance_idx_mapping[uri] for uri in pos_examples]
            neg_idx = [self.instance_idx_mapping[uri] for uri in neg_examples]

            # Compute labels (F1 scores for each target expression)
            labels = []
            for target_exp in self.target_class_expressions:
                f1 = self.quality_func(
                    individuals=target_exp.str_individuals,
                    pos=set(pos_examples),
                    neg=set(neg_examples)
                )
                labels.append(f1)

            X_pos_list.append(pos_idx)
            X_neg_list.append(neg_idx)
            Y_list.append(labels)

        # Pad sequences to same length
        max_pos_len = max(len(x) for x in X_pos_list) if X_pos_list else 0
        max_neg_len = max(len(x) for x in X_neg_list) if X_neg_list else 0

        X_pos_padded = torch.zeros(len(X_pos_list), max_pos_len, dtype=torch.long)
        X_neg_padded = torch.zeros(len(X_neg_list), max_neg_len, dtype=torch.long)

        for i, (pos, neg) in enumerate(zip(X_pos_list, X_neg_list)):
            X_pos_padded[i, :len(pos)] = torch.tensor(pos)
            X_neg_padded[i, :len(neg)] = torch.tensor(neg)

        Y = torch.tensor(Y_list, dtype=torch.float32)

        # Training loop
        dataset = torch.utils.data.TensorDataset(X_pos_padded, X_neg_padded, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for xpos, xneg, y in dataloader:
                xpos = xpos.to(self.device)
                xneg = xneg.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(xpos, xneg)
                loss = loss_func(predictions, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

        self.model.eval()
        self._is_trained = True

        if self.verbose > 0:
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")

    def search(self, pos: Set[str], neg: Set[str], top_k: int = 10, max_child_length: int = 10, max_queue_size: int = 10000) -> Dict:
        """
        Perform reinforcement learning-based search for complex class expressions.
        Uses neural predictions to initialize and guide the search.
        """
        if self.verbose > 0:
            print("Starting reinforcement learning search...")

        start_time = time.time()
        self.search_tree = SearchTree()

        # Initialize with neural predictions OR top refinements
        if self._is_trained and len(self.target_class_expressions) > 0:
            idx_pos = torch.LongTensor([[self.instance_idx_mapping[i] for i in pos]])
            idx_neg = torch.LongTensor([[self.instance_idx_mapping[i] for i in neg]])

            with torch.no_grad():
                sort_scores, sort_idxs = torch.topk(
                    self.model(idx_pos, idx_neg).flatten(),
                    min(top_k, len(self.target_class_expressions)),
                    largest=True
                )

            # Initialize queue with top neural predictions
            for idx_target in sort_idxs.detach().numpy():
                target_ce = self.target_class_expressions[idx_target]
                try:
                    target_ce.quality = self.quality_func(
                        individuals=target_ce.str_individuals, pos=pos, neg=neg
                    )
                    self.search_tree.put(target_ce)
                except (AttributeError, KeyError):
                    continue

        # Always add top refinements to initial queue for diversity
        # This ensures we explore complex expressions, not just atomic ones
        if self.refinement_op and self.refinement_op.top_refinements:
            for length, refinement_set in self.refinement_op.top_refinements.items():
                for concept in refinement_set:
                    if concept.name not in self.search_tree.gate:
                        try:
                            concept.quality = self.quality_func(
                                individuals=concept.str_individuals, pos=pos, neg=neg
                            )
                            self.search_tree.put(concept)
                        except (AttributeError, KeyError):
                            continue

        if len(self.search_tree.gate) == 0:
            raise ValueError("Failed to initialize search tree with any concepts")

        best_hypothesis = None
        best_quality = -1.0

        # Find initial best
        for name, expr in list(self.search_tree.gate.items()):
            if expr.quality > best_quality:
                best_quality = expr.quality
                best_hypothesis = expr

        if self.verbose > 0:
            print(f"Initial best hypothesis: {best_hypothesis.name} (Quality: {best_quality:.3f})")
            print(f"Initial queue size: {len(self.search_tree.gate)}")

        # Iterative refinement-based search
        iteration = 0
        max_iterations = 100000
        num_explored = len(self.search_tree.gate)
        # More generous exploration: explore up to 50% more concepts, not just 10%
        exploration_limit = num_explored + max(num_explored // 2, 100)

        while (not self.search_tree.items_in_queue.empty() and
               (time.time() - start_time) < self.max_runtime and
               iteration < max_iterations and
               num_explored < exploration_limit):
            iteration += 1

            queue_size = self.search_tree.items_in_queue.qsize()
            if queue_size > max_queue_size:
                if self.verbose > 0:
                    print(f"WARNING: Queue size ({queue_size}) exceeded max. Stopping search.")
                break

            try:
                current_concept = self.search_tree.get()
            except Exception:
                break

            if current_concept.length > max_child_length:
                continue

            # Generate refinements
            refined_concepts = self.refinement_op.refine(current_concept)

            if not refined_concepts:
                continue

            # Limit refinements
            if len(refined_concepts) > 100:
                refined_concepts = refined_concepts[:100]

            # Evaluate refinements
            for next_concept in refined_concepts:
                if next_concept.name not in self.search_tree.gate:
                    try:
                        quality = self.quality_func(
                            individuals=next_concept.str_individuals, pos=pos, neg=neg
                        )
                        next_concept.quality = quality

                        if next_concept.length <= max_child_length and queue_size < max_queue_size:
                            self.search_tree.put(next_concept)
                            num_explored += 1

                        if quality > best_quality:
                            best_quality = quality
                            best_hypothesis = next_concept
                            if self.verbose > 1:
                                print(f"New best at iter {iteration}: {best_hypothesis.name} (Quality: {best_quality:.3f})")

                        # Early stopping if perfect solution found
                        if quality >= 1.0:
                            if self.verbose > 0:
                                print(f"Perfect solution found at iteration {iteration}")
                            runtime = time.time() - start_time
                            return {
                                'Prediction': best_hypothesis.name,
                                'F-measure': best_quality,
                                'Runtime': runtime,
                                'Quality': best_quality
                            }
                    except (AttributeError, KeyError, Exception) as e:
                        # Skip concepts that cause errors during evaluation
                        if self.verbose > 1:
                            print(f"Skipping concept due to error: {e}")
                        continue

            if iteration % 10 == 0:
                time.sleep(0.001)

            if self.verbose > 0 and iteration % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration}, Queue: {queue_size}, Best: {best_quality:.3f}, Time: {elapsed:.1f}s")

        runtime = time.time() - start_time
        if self.verbose > 0:
            print(f"Search finished in {runtime:.2f}s after {iteration} iterations.")
            print(f"Best: {best_hypothesis.name} (Quality: {best_quality:.3f})")

        return {
            'Prediction': best_hypothesis.name,
            'F-measure': best_quality,
            'Runtime': runtime,
            'Quality': best_quality
        }

    def search_with_smart_init(self, pos: Set[str], neg: Set[str], top_k: int = 10) -> Dict:
        """
        Search with smart initialization from neural predictions (model.py compatible).
        This uses neural model predictions to guide the symbolic refinement search.
        """
        if not self._is_trained:
            return self.search(pos, neg, top_k=top_k)

        start_time = time.time()

        # Get neural predictions
        idx_pos = torch.LongTensor([[self.instance_idx_mapping[i] for i in pos]])
        idx_neg = torch.LongTensor([[self.instance_idx_mapping[i] for i in neg]])

        with torch.no_grad():
            sort_scores, sort_idxs = torch.topk(
                self.model(idx_pos, idx_neg).flatten(),
                min(top_k, len(self.target_class_expressions)),
                largest=True
            )

        # Initialize priority queue with top predictions
        top_predictions = SearchTree()

        for idx_target in sort_idxs.detach().numpy():
            target_ce = self.target_class_expressions[idx_target]
            target_ce.quality = self.quality_func(
                individuals=target_ce.str_individuals, pos=pos, neg=neg
            )
            top_predictions.put(target_ce)

        # Refine top predictions
        n = len(top_predictions)
        exploration_budget = n  # Explore same amount as initial predictions
        refinements_explored = SearchTree()

        while len(top_predictions) > (n * 0.99) and exploration_budget > 0:
            current = top_predictions.get()

            for refined in self.refinement_op.refine(current):
                if refined.name in refinements_explored.gate or refined.name in top_predictions.gate:
                    continue

                refined.quality = self.quality_func(
                    individuals=refined.str_individuals, pos=pos, neg=neg
                )

                exploration_budget -= 1
                refinements_explored.put(refined)
                top_predictions.put(refined)

                if exploration_budget <= 0:
                    break

            if exploration_budget <= 0:
                break

        best_pred = top_predictions.get()
        runtime = time.time() - start_time

        return {
            'Prediction': best_pred.name,
            'F-measure': best_pred.quality,
            'Runtime': runtime,
            'Quality': best_pred.quality
        }

    def fit(self, learning_problem: PosNegLPStandard, max_runtime: Optional[int] = None):
        """
        Fit the model to a learning problem (Ontolearn-compatible interface).
        This now includes training the neural model and performing the search.
        """
        if max_runtime:
            self.max_runtime = max_runtime

        # 1. Initialize components
        self._initialize_instance_mapping()
        self._initialize_refinement_operator()

        # 2. Train the neural model (as a policy guide, though not fully used in this search impl)
        pos_examples = [ind.str for ind in learning_problem.pos]
        neg_examples = [ind.str for ind in learning_problem.neg]
        self.train([(pos_examples, neg_examples)])

        # 3. Perform the search to find the best complex expression
        self._best_predictions = self.search(pos=set(pos_examples), neg=set(neg_examples))

        return self

    def best_hypothesis(self) -> Optional[str]:
        """
        Return the best hypothesis (Ontolearn-compatible interface).

        Returns:
            The best predicted class expression as a string
        """
        if not self._is_trained or self._best_predictions is None:
            return None
        assert self.ns is not None, "Namespace must be set for OWL expression conversion"

        return dl_to_owl_expression(self._best_predictions['Prediction'], self.ns)

    def best_hypothesis_quality(self) -> float:
        """
        Return the quality of the best hypothesis.

        Returns:
            The F-measure/quality of the best prediction
        """
        if not self._is_trained or self._best_predictions is None:
            return 0.0
        return self._best_predictions['Quality']

    def forward(self, xpos: torch.Tensor, xneg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural model.

        Args:
            xpos: Tensor of positive example indices
            xneg: Tensor of negative example indices

        Returns:
            Predictions for target class expressions
        """
        return self.model(xpos, xneg)

    def positive_expression_embeddings(self, individuals: List[str]) -> torch.Tensor:
        """
        Get embeddings for positive individuals.

        Args:
            individuals: List of individual URIs

        Returns:
            Tensor of embeddings
        """
        indices = torch.LongTensor([[self.instance_idx_mapping[i] for i in individuals]])
        return self.model.positive_expression_embeddings(indices)

    def negative_expression_embeddings(self, individuals: List[str]) -> torch.Tensor:
        """
        Get embeddings for negative individuals.

        Args:
            individuals: List of individual URIs

        Returns:
            Tensor of embeddings
        """
        indices = torch.LongTensor([[self.instance_idx_mapping[i] for i in individuals]])
        return self.model.negative_expression_embeddings(indices)

    def downward_refine(self, expression, max_length: Optional[int] = None) -> Set:
        """
        Top-down/downward refinement operator from original NERO.

        This implements the refinement logic from model.py:
        ∀s ∈ StateSpace : ρ(s) ⊆ {s^i ∈ StateSpace | s^i ⊑ s}

        Args:
            expression: Expression to refine
            max_length: Maximum length constraint for refinements

        Returns:
            Set of refined expressions
        """
        if isinstance(expression, str):
            if expression == '⊤':
                return set()  # Can return top_refinements if needed
            return set()

        refinements = set()

        # Get expression from refinement operator's expressions dict
        if hasattr(expression, 'name') and expression.name in self.refinement_op.expressions:
            expression = self.refinement_op.expressions[expression.name]

        # Get all refinements from the refinement operator
        refined_list = self.refinement_op.refine(expression)

        for ref in refined_list:
            if max_length is None or ref.length <= max_length:
                refinements.add(ref)

        return refinements

    def upward_refine(self, expression) -> Set:
        """
        Bottom-up/upward refinement operator from original NERO.

        This implements the generalization logic:
        ∀s ∈ StateSpace : ρ(s) ⊆ {s^i ∈ StateSpace | s ⊑ s^i}

        Args:
            expression: Expression to generalize

        Returns:
            Set of generalized expressions
        """
        # Note: Upward refinement is less commonly used in NERO's main search
        # It's included for completeness but the main search uses downward refinement
        refinements = set()

        # This would require implementing upward refinement in the refinement operator
        # For now, return empty set as it's not critical for the main search functionality
        return refinements

    def search_with_init(self, top_prediction_queue: SearchTree, set_pos: Set[str], set_neg: Set[str]) -> SearchTree:
        """
        Standard search with smart initialization (from original model.py).

        This is the key search method that combines neural predictions with symbolic refinement.

        Args:
            top_prediction_queue: Priority queue initialized with neural predictions
            set_pos: Set of positive examples
            set_neg: Set of negative examples

        Returns:
            SearchTree with explored and refined expressions
        """
        # Initialize final predictions
        top_predictions = SearchTree()
        top_predictions.extend_queue(top_prediction_queue)
        refinements_of_top_predictions = SearchTree()

        n = len(top_prediction_queue)
        exploration_counter = n

        # Iterate over advantageous states
        while len(top_prediction_queue) > (n * 0.99):  # explore only 1 percent
            # Get top ranked Description Logic Expression
            c = top_prediction_queue.get()

            # Refine with length constraint
            for a in self.downward_refine(c, max_length=c.length + 3):
                if a.name in refinements_of_top_predictions.gate or a.name in top_predictions.gate:
                    # Already seen
                    continue
                else:
                    a.quality = self.quality_func(
                        individuals=a.str_individuals,
                        pos=set_pos,
                        neg=set_neg
                    )
                    # Add refinements
                    exploration_counter -= 1
                    refinements_of_top_predictions.put(a, key=-a.quality)
                    top_predictions.put(a, key=-a.quality)

                    if exploration_counter == 0:
                        break

            if exploration_counter == 0:
                break

        return top_predictions

    def fit_from_iterable(self, pos: List[str], neg: List[str], top_k: int = 10, use_search: str = 'SmartInit') -> Dict:
        """
        Fit method compatible with original NERO's model.py interface.

        This implements the complete prediction pipeline from the original NERO:
        1. Neural prediction to get top-k candidates
        2. Quality evaluation
        3. Optional symbolic search for refinement

        Args:
            pos: List of positive example URIs
            neg: List of negative example URIs
            top_k: Number of top neural predictions to consider
            use_search: Search strategy ('SmartInit', 'None', or None)

        Returns:
            Dictionary with prediction results
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before calling fit_from_iterable")

        start_time = time.time()
        set_pos, set_neg = set(pos), set(neg)

        # Get neural predictions
        idx_pos = torch.LongTensor([[self.instance_idx_mapping[i] for i in pos]])
        idx_neg = torch.LongTensor([[self.instance_idx_mapping[i] for i in neg]])

        goal_found = False
        top_prediction_queue = SearchTree()

        # Get top-k predictions from neural model
        with torch.no_grad():
            sort_scores, sort_idxs = torch.topk(
                self.forward(xpos=idx_pos, xneg=idx_neg).flatten(),
                min(top_k, len(self.target_class_expressions)),
                largest=True
            )

        # Evaluate predictions
        for idx_target in sort_idxs.detach().numpy():
            target_ce = self.target_class_expressions[idx_target]
            target_ce.quality = self.quality_func(
                individuals=target_ce.str_individuals,
                pos=set_pos,
                neg=set_neg
            )
            top_prediction_queue.put(target_ce, key=-target_ce.quality)

            if target_ce.quality == 1.0:
                goal_found = True
                break

        assert len(top_prediction_queue) > 0

        # If goal not found, perform search
        if not goal_found:
            if use_search == 'SmartInit':
                best_pred = top_prediction_queue.get()
                top_prediction_queue.put(best_pred, key=-best_pred.quality)
                top_prediction_queue = self.search_with_init(
                    top_prediction_queue, set_pos, set_neg
                )
                best_constructed_expression = top_prediction_queue.get()

                if best_constructed_expression.quality > best_pred.quality:
                    best_pred = best_constructed_expression
            elif use_search == 'None' or use_search is None:
                best_pred = top_prediction_queue.get()
            else:
                raise ValueError(f"Unknown search strategy: {use_search}")
        else:
            best_pred = top_prediction_queue.get()

        runtime = time.time() - start_time

        report = {
            'Prediction': best_pred.name,
            'Instances': best_pred.str_individuals,
            'F-measure': round(best_pred.quality, 3),
            'Runtime': round(runtime, 3),
            'Quality': best_pred.quality
        }

        return report

    def predict(self, pos: Set[OWLNamedIndividual], neg: Set[OWLNamedIndividual], top_k: int = 10) -> Dict:
        """
        Predict class expressions for given positive and negative examples.
        This now uses the search mechanism.
        """
        if not self._is_trained:
            self.fit(PosNegLPStandard(pos=pos, neg=neg))

        return self._best_predictions


    def __str__(self):
        return f"NERO(architecture={self.neural_architecture}, embedding_dim={self.num_embedding_dim})"

    def __repr__(self):
        return self.__str__()
