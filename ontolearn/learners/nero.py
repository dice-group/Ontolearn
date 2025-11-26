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

from typing import Dict, List, Set, Tuple, Optional, Iterable
import time
import torch
from torch import nn
import math
import torch.nn.functional as F
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from queue import PriorityQueue
from collections import deque
from multiprocessing import Pool

from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.abstracts import AbstractKnowledgeBase, BaseRefinement
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils.static_funcs import compute_f1_score


# =============================================================================
# Expression Classes
# =============================================================================

class Role:
    """Represents an OWL object property/role."""
    def __init__(self, *, name: str):
        assert isinstance(name, str)
        self.name = name

    def __str__(self):
        return f'Role at {hex(id(self))} | {self.name}'

    def __repr__(self):
        return self.__str__()


class TargetClassExpression:
    """Represents a target class expression for neural training."""
    def __init__(self, *, label_id, name: str, idx_individuals: Set = None,
                 expression_chain: List = None, length: int = None,
                 str_individuals: Set = None, type=None):
        self.label_id = label_id
        self.name = name
        self.idx_individuals = idx_individuals
        self.str_individuals = str_individuals
        self.type = type
        self.expression_chain = expression_chain
        self.num_individuals = len(self.str_individuals) if self.str_individuals else 0
        self.length = length
        self.quality = None

    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality

    def __str__(self):
        return f'TargetClassExpression | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality}'

    def __repr__(self):
        return self.__str__()


class ClassExpression(ABC):
    """Base class for class expressions."""
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List,
                 owl_class=None, quality=None, length=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, (list, tuple))
        self.name = name
        self.str_individuals = str_individuals
        self.expression_chain = expression_chain
        self.num_individuals = len(self.str_individuals)
        self.quality = quality if quality is not None else -1.0
        self.owl_class = owl_class
        self.length = length if length is not None else len(self.name.split())

    def __str__(self):
        return f'{self.type} | {self.name} | Indv:{self.num_individuals} | Quality:{self.quality:.3f}'

    def __repr__(self):
        return self.__str__()

    @property
    def size(self):
        return self.num_individuals

    def __lt__(self, other):
        return self.quality < other.quality


class AtomicExpression(ClassExpression):
    """Represents an atomic class expression."""
    def __init__(self, *, name: str, str_individuals: Set, expression_chain: List,
                 owl_class=None, quality=None, label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        self.length = 1
        self.type = 'atomic_expression'
        self.idx_individuals = idx_individuals
        self.label_id = label_id


class ComplementOfAtomicExpression(ClassExpression):
    """Represents a negated atomic class expression."""
    def __init__(self, *, name: str, atomic_expression, str_individuals: Set,
                 expression_chain: List, quality=None, owl_class=None,
                 label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        self.atomic_expression = atomic_expression
        self.length = 2
        self.type = 'negated_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals


class UniversalQuantifierExpression(ClassExpression):
    """Represents a universal quantifier expression (∀)."""
    def __init__(self, *, name: str, role=None, filler=None, label_id=None,
                 idx_individuals=None, str_individuals: Set, expression_chain: List, quality=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, (list, tuple))
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "universal_quantifier_expression"
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.length = 3


class ExistentialQuantifierExpression(ClassExpression):
    """Represents an existential quantifier expression (∃)."""
    def __init__(self, *, name: str, role=None, filler=None, str_individuals: Set,
                 expression_chain: List, quality=None, label_id=None, idx_individuals=None):
        assert isinstance(name, str)
        assert isinstance(str_individuals, set)
        assert isinstance(expression_chain, (list, tuple))
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality)
        self.role = role
        self.filler = filler
        self.type = "existantial_quantifier_expression"
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.length = 3


class IntersectionClassExpression(ClassExpression):
    """Represents an intersection of class expressions."""
    def __init__(self, *, name: str, length: int, str_individuals: Set,
                 expression_chain: List, owl_class=None, quality=None,
                 label_id=None, concepts=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        assert length >= 3
        self.length = length
        self.type = 'intersection_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.concepts = concepts


class UnionClassExpression(ClassExpression):
    """Represents a union of class expressions."""
    def __init__(self, *, name: str, length: int, str_individuals: Set,
                 expression_chain: List, owl_class=None, concepts=None,
                 quality=None, label_id=None, idx_individuals=None):
        super().__init__(name=name, str_individuals=str_individuals,
                         expression_chain=expression_chain, quality=quality, owl_class=owl_class)
        assert length >= 3
        self.length = length
        self.type = 'union_expression'
        self.label_id = label_id
        self.idx_individuals = idx_individuals
        self.concepts = concepts


# =============================================================================
# Data Structures
# =============================================================================

class SearchTree:
    """Priority queue for managing search states."""
    def __init__(self, maxsize=0):
        self.items_in_queue = PriorityQueue(maxsize)
        self.gate = dict()

    def __contains__(self, key):
        return key in self.gate

    def put(self, expression, key=None, condition=None):
        if condition is None:
            if expression.name not in self.gate:
                if key is None:
                    key = -expression.quality
                self.items_in_queue.put((key, expression))
                self.gate[expression.name] = expression
        else:
            raise ValueError('Define the condition')

    def get(self):
        _, expression = self.items_in_queue.get(timeout=1)
        del self.gate[expression.name]
        return expression

    def get_all(self):
        return list(self.gate.values())

    def __len__(self):
        return len(self.gate)

    def __iter__(self):
        return (exp for q, exp in self.items_in_queue.queue)


# =============================================================================
# Neural Architectures
# =============================================================================

class MAB(nn.Module):
    """Multi-head Attention Block."""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Self-Attention Block."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """Induced Self-Attention Block."""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by Multihead Attention."""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    """Set Transformer architecture."""
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32,
                 dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class DeepSet(torch.nn.Module):
    """DeepSet neural architecture for set-based learning."""
    def __init__(self, num_instances: int, num_embedding_dim: int, num_outputs: int):
        super(DeepSet, self).__init__()
        self.name = 'DeepSet'
        self.num_instances = num_instances
        self.num_embedding_dim = num_embedding_dim
        self.num_outputs = num_outputs

        self.embeddings = torch.nn.Embedding(self.num_instances, self.num_embedding_dim)
        self.fc0 = nn.Sequential(
            nn.BatchNorm1d(self.num_embedding_dim),
            nn.Linear(in_features=self.num_embedding_dim, out_features=self.num_outputs))
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_embedding_dim),
            nn.Linear(in_features=self.num_embedding_dim, out_features=self.num_outputs))

    def forward(self, xpos, xneg):
        xpos_score = self.fc0(torch.sum(self.embeddings(xpos), 1))
        xneg_score = self.fc1(torch.sum(self.embeddings(xneg), 1))
        return torch.sigmoid(xpos_score - xneg_score)

    def positive_expression_embeddings(self, tensor_idx_individuals: torch.LongTensor):
        return self.fc0(torch.sum(self.embeddings(tensor_idx_individuals), 1))

    def negative_expression_embeddings(self, tensor_idx_individuals: torch.LongTensor):
        return self.fc1(torch.sum(self.embeddings(tensor_idx_individuals), 1))


class SetTransformerNet(torch.nn.Module):
    """Set Transformer based architecture."""
    def __init__(self, num_instances: int, num_embedding_dim: int, num_outputs: int):
        super(SetTransformerNet, self).__init__()
        self.name = 'ST'
        self.num_instances = num_instances
        self.num_embedding_dim = num_embedding_dim
        self.num_outputs = num_outputs

        self.embeddings = torch.nn.Embedding(self.num_instances, self.num_embedding_dim)
        self.set_transformer_negative = SetTransformer(
            dim_input=self.num_embedding_dim, num_outputs=self.num_outputs,
            dim_output=1, num_inds=4, dim_hidden=4, num_heads=4, ln=False)
        self.set_transformer_positive = SetTransformer(
            dim_input=self.num_embedding_dim, num_outputs=self.num_outputs,
            dim_output=1, num_inds=4, dim_hidden=4, num_heads=4, ln=False)

    def forward(self, xpos, xneg):
        xpos_score = torch.squeeze(self.set_transformer_positive(self.embeddings(xpos)), dim=2)
        xneg_score = torch.squeeze(self.set_transformer_negative(self.embeddings(xneg)), dim=2)
        return torch.sigmoid(xpos_score - xneg_score)


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
                 num_embedding_dim: int = 50,
                 neural_architecture: str = 'DeepSet',
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 quality_func=None,
                 max_runtime: Optional[int] = None,
                 verbose: int = 0):

        self.kb = knowledge_base
        self.num_embedding_dim = num_embedding_dim
        self.neural_architecture = neural_architecture
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_runtime = max_runtime
        self.verbose = verbose

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
                type='atomic_expression'
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
        max_pos_len = max(len(x) for x in X_pos_list)
        max_neg_len = max(len(x) for x in X_neg_list)

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

    def fit(self, learning_problem: PosNegLPStandard, max_runtime: Optional[int] = None):
        """
        Fit the model to a learning problem (Ontolearn-compatible interface).

        Args:
            learning_problem: A PosNegLPStandard learning problem
            max_runtime: Maximum runtime in seconds
        """
        pos_examples = [ind.str for ind in learning_problem.pos]
        neg_examples = [ind.str for ind in learning_problem.neg]

        # Train on this single learning problem
        self.train([(pos_examples, neg_examples)])

        # Return self for chaining
        return self

    def predict(self, pos: Set[OWLNamedIndividual], neg: Set[OWLNamedIndividual], top_k: int = 10) -> Dict:
        """
        Predict class expressions for given positive and negative examples.

        Args:
            pos: List of positive example IRIs
            neg: List of negative example IRIs
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        start_time = time.time()

        pos_uris = [ind.str for ind in pos]
        neg_uris = [ind.str for ind in neg]

        # Convert to indices
        pos_idx = torch.LongTensor([[self.instance_idx_mapping[uri] for uri in pos_uris]])
        neg_idx = torch.LongTensor([[self.instance_idx_mapping[uri] for uri in neg_uris]])

        # Get predictions
        with torch.no_grad():
            pos_idx = pos_idx.to(self.device)
            neg_idx = neg_idx.to(self.device)
            scores = self.model(pos_idx, neg_idx).flatten()

        # Get top k predictions
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)), largest=True)

        # Find best prediction
        best_quality = -1.0
        best_prediction = None
        set_pos, set_neg = set(pos), set(neg)

        for idx in top_indices.cpu().numpy():
            target_exp = self.target_class_expressions[idx]
            quality = self.quality_func(
                individuals=target_exp.str_individuals,
                pos=set_pos,
                neg=set_neg
            )

            if quality > best_quality:
                best_quality = quality
                best_prediction = target_exp

        runtime = time.time() - start_time

        return {
            'Prediction': best_prediction.name if best_prediction else "⊤",
            'F-measure': best_quality,
            'Runtime': runtime,
            'Quality': best_quality
        }

    def __str__(self):
        return f"NERO(architecture={self.neural_architecture}, embedding_dim={self.num_embedding_dim})"

    def __repr__(self):
        return self.__str__()
