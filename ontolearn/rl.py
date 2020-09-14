from abc import ABCMeta

from .base_rl_agent import BaseRLTrainer
from .util import get_full_iri, balanced_sets
from .search import Node
import random
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import functools
from torch.functional import F
from typing import List, Any
from collections import namedtuple
from .abstracts import AbstractScorer
from typing import Set, AnyStr, Tuple
from torch.nn.init import xavier_normal_
from itertools import chain
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR


class DrillHeuristic(AbstractScorer):
    """
    Use pretrained agent as heuristic func.
    """

    def __init__(self, pos=None, neg=None, model=None):
        super().__init__(pos, neg, unlabelled=None)
        self.name = 'DrillHeuristic'
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.model.eval()

    def score(self, node, parent_node=None):
        """ Compute and return predicted Q-value"""
        if parent_node is None and node.is_root:
            return torch.FloatTensor([.0001]).squeeze()
        with torch.no_grad():
            return self.model.forward(parent_node.concept.idx_instances, node.concept.idx_instances, self.pos,
                                      self.neg).squeeze()

    def apply(self, node, parent_node=None):
        """ Assign predicted Q-value to node object."""
        predicted_q_val = self.score(node, parent_node)
        node.heuristic = predicted_q_val


class Drill(nn.Module, metaclass=ABCMeta):
    """
    A neural model for Deep Q-Learning.

    An input Drill has the following form
            1. indexes of individuals belonging to current state (s).
            2. indexes of individuals belonging to next state state (s_prime).
            3. indexes of individuals provided as positive examples.
            4. indexes of individuals provided as negative examples.

    Given such input, we from a sparse 3D Tensor where  each slice is a **** N *** by ***D***
    where N is the number of individuals and D is the number of dimension of embeddings.
    Given that N on the current benchmark datasets < 10^3, we can get away with this computation. By doing so
    we do not need to subsample from given inputs.

    """

    def __init__(self):
        super(Drill, self).__init__()
        self.loss = nn.MSELoss()

        self.conv1 = nn.Conv1d(in_channels=4,
                               out_channels=32,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=True)
        self.pool = nn.MaxPool2d(kernel_size=3,
                                 padding=1,
                                 stride=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=True)
        self.bn2 = nn.BatchNorm1d(16)

        self.conv3 = nn.Conv1d(in_channels=16,
                               out_channels=8,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=True)

        self.fc1 = nn.Linear(in_features=800, out_features=200)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(in_features=200, out_features=50)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(in_features=50, out_features=1)
        self.bn5 = nn.BatchNorm1d(50)

    def init(self):
        xavier_normal_(self.fc1.weight.data)
        xavier_normal_(self.fc2.weight.data)
        xavier_normal_(self.fc3.weight.data)

        xavier_normal_(self.conv1.weight.data)
        xavier_normal_(self.conv2.weight.data)
        xavier_normal_(self.conv3.weight.data)

    def forward(self, X: torch.FloatTensor):
        # X => torch.Size([batchsize, 4, dim])
        X = self.bn1(self.pool(F.relu(self.conv1(X))))

        # X => torch.Size([batchsize, 32, dim]) # 32 kernels.
        X = self.bn2(self.pool(F.relu(self.conv2(X))))
        # X => torch.Size([batchsize, 16, dim]) # 16 kernels.
        X = self.bn3(self.pool(F.relu(self.conv3(X))))
        # X => torch.Size([batchsize, 8, dim]) # 16 kernels.
        X = X.view(-1, X.shape[1] * X.shape[2])

        X = self.bn4(F.relu(self.fc1(X)))
        X = self.bn5(F.relu(self.fc2(X)))
        return self.fc3(X)


class DrillTrainer(BaseRLTrainer):
    """
    A training for DQL agent.
    GOAL:

    1) Drill must be able to learn embeddings of individuals if no embeddings provided.
    2) If embeddings of individuals provided then we can do any of the followings
        2.1) Given that number of indiv in states differ and sizes of pos and neg are not same.
            We must sample from them to create input tensor in our manuscript the input tensor is visualized.
            where sample size =10.
        2.2) Given that current benchmark datasets are small, i.e. number of instances < 10^3.
            Input tensor can have be consider  numberofinstances x 4 x embedding_dim denoted by X
            where slices represent current state, next state, pos and neg
            and rows of each slice is mask by 0 if the corresponding entity is not given.



    """

    def __init__(self, *,
                 knowledge_base,
                 refinement_operator,
                 quality_func,
                 search_tree,
                 reward_func,
                 train_data, pre_trained_drill, relearn_rate_per_problem, learning_problem_generator,
                 instance_embeddings
                 ):
        super().__init__(kb=knowledge_base,
                         rho=refinement_operator,
                         quality_func=quality_func,
                         search_tree=search_tree,
                         reward_func=reward_func,
                         train_data=train_data)

        if pre_trained_drill:
            self.model = pre_trained_drill
        else:
            self.model = Drill()
            self.model.init()

        self.lp_gen = learning_problem_generator
        self.instance_embeddings = instance_embeddings
        # Normalize input
        self.instance_embeddings = (self.instance_embeddings - self.instance_embeddings.min()) / (
                self.instance_embeddings.max() - self.instance_embeddings.min())

        self.relearn_rate_per_problem = relearn_rate_per_problem
        self.decay_rate = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=.0001)
        self.scheduler = ExponentialLR(self.optimizer, self.decay_rate)
        self.iter_bound = 100
        self.learn_per_modulo_iter = 10  # 50%10==0 > 5 times.
        self.num_epoch_per_replay = 3
        self.batch_size = 1024
        self.min_length = 2
        self.size_of_path = 4
        self.epsilon, self.epsilon_decay, self.epsilon_min = 1, 0.99, 0.01
        self.experiences = []

        self.pos, self.neg = None, None
        self.emb_pos, self.emb_neg = None, None

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)
        self.search_tree.add_root(root)

    def apply_rho(self, node: Node) -> List:
        assert isinstance(node, Node)
        return [self.rho.getNode(i, parent_node=node) for i in
                self.rho.refine(node, maxlength=len(node) + self.min_length)]

    def next_node_to_expand(self, current_state, next_states) -> Node:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        """
        if np.random.random() < self.epsilon:  # Exploration vs Exploitation
            next_state = random.choice(next_states)
            self.assign_embeddings(next_state)
            return next_state
        else:
            self.assign_embeddings(current_state)
            with torch.no_grad():
                self.model.eval()
                # create batch batch.
                next_state_batch = []
                for _ in next_states:
                    self.assign_embeddings(_)
                    next_state_batch.append(_.concept.embeddings)
                next_state_batch = torch.cat(next_state_batch, dim=0)

                ds = PrepareBatchOfPrediction(current_state.concept.embeddings,
                                              next_state_batch,
                                              self.emb_pos,
                                              self.emb_neg)
                predictions = self.model.forward(ds.get_all())
                next_state = next_states[torch.argmax(predictions)]
            return next_state

    def assign_embeddings(self, node: Node):
        if node.concept.embeddings is None:
            str_idx = [get_full_iri(i).replace('\n', '') for i in node.concept.instances]
            emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
            emb = torch.mean(emb, dim=0)
            emb = emb.view(1, 1, emb.shape[0])
            node.concept.embeddings = emb

    def sequence_of_actions(self, root):
        current_state = root
        path_of_concepts = []
        rewards = []

        for _ in range(self.size_of_path):
            next_states = self.apply_rho(current_state)
            if len(next_states) == 0:  # DEAD END, what to do ?
                break
            next_state = self.next_node_to_expand(current_state, next_states)
            assert next_state
            assert current_state
            if next_state.concept.str == 'Nothing':  # Dead END
                break
            path_of_concepts.append((current_state, next_state))
            rewards.append(self.reward_func.calculate(current_state, next_state))
            current_state = next_state

        return path_of_concepts, rewards

    def form_experiences(self, state_pairs: List, rewards: List) -> None:
        """
        Form experiences from a sequence of concepts and corresponding rewards.

        state_pairs - a list of tuples containing two consecutive states
        reward      - a list of reward.

        Return
        X - a list of embeddings of current concept, next concept, positive examples, negative examples
        y - argmax Q value.
        """

        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    def learn_from_replay_memory(self) -> None:

        current_state_batch = []
        next_state_batch = []
        q_values = []

        for experience in self.experiences:
            s, s_prime, q = experience
            current_state_batch.append(s.concept.embeddings)
            next_state_batch.append(s_prime.concept.embeddings)
            q_values.append(q)

        current_state_batch = torch.cat(current_state_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)
        q_values = torch.Tensor(q_values)

        ds = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                    next_state_batch=next_state_batch,
                                    p=self.emb_pos, n=self.emb_neg, q=q_values)
        self.model.train()
        for m in range(self.num_epoch_per_replay):
            total_loss = 0
            for X, y in DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=1):
                if torch.isnan(X).any() or torch.isinf(X).any():
                    print('invalid input detected')
                    exit(1)

                if torch.isnan(y).any() or torch.isinf(y).any():
                    print('invalid q val detected')
                    exit(1)

                self.optimizer.zero_grad()  # zero the gradient buffers

                # forward
                predicted_q = self.model.forward(X)
                # loss
                loss = self.model.loss(predicted_q, y)
                total_loss += loss.item()
                # compute the derivative of the loss w.r.t. the parameters using backpropagation
                loss.backward()
                # clip gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
            print(total_loss)

    def start(self):
        """
        Training RL agent on given standard supervised learning problems.
        For each supervised learning problem
        1) Train RL agent.
        2) Use RL agent as Heuristic function.
        """

        for example_node in self.lp_gen:
            # string_pos
            string_all_pos = {get_full_iri(i) for i in example_node.concept.instances}
            string_all_neg = {get_full_iri(i) for i in
                              self.kb.get_all_individuals().difference(example_node.concept.instances)}
            print('Target Concept:', example_node.concept.str, '\t |E+|', len(string_all_pos), '\t |E-|',
                  len(string_all_neg))
            # create balanced setting
            string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
            print('|Pos. Ex|:{0}\t|Neg. Ex|:{1}:'.format(len(string_balanced_pos), len(string_balanced_neg)))

            # string to owlready2 object conversion
            self.pos = {self.kb.str_to_instance_obj[i] for i in string_balanced_pos}
            self.neg = {self.kb.str_to_instance_obj[i] for i in string_balanced_neg}
            self.reward_func.pos = self.pos
            self.reward_func.neg = self.neg

            self.emb_pos = torch.tensor(self.instance_embeddings.loc[list(string_balanced_pos)].values,
                                        dtype=torch.float32)
            self.emb_neg = torch.tensor(self.instance_embeddings.loc[list(string_balanced_neg)].values,
                                        dtype=torch.float32)
            # take mean and reshape it into (1,1,embedding_dim) for mini batching.
            self.emb_pos = torch.mean(self.emb_pos, dim=0)
            self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
            self.emb_neg = torch.mean(self.emb_neg, dim=0)
            self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])

            self.train()  # Let model to train to solve given problem

            self.test(string_pos, string_neg)  # Let test model to train to solve given problem

    def train(self):
        """
        Agent trains on quasi ordered concept environment until one of the stopping criterion is fulfilled.
        """
        print('Training starts.')
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root)

        # training loop.
        for th in range(1, self.iter_bound):
            path_of_concepts, rewards = self.sequence_of_actions(root)
            print('{0}.th iteration total reward: {1:.2f}\tEpsilon:{2:.2f}\t Num.Exp:{3}'.format(th, sum(rewards),
                                                                                                 self.epsilon,
                                                                                                 len(self.experiences)))

            self.epsilon -= 0.001
            if self.epsilon < 0:
                break

            self.form_experiences(path_of_concepts, rewards)

            if th % self.learn_per_modulo_iter == 0:
                self.learn_from_replay_memory()

        exit(1)

    def test(self, pos: Set[AnyStr], neg: Set[AnyStr]):
        """ use agent as heuristic function in the concept learning problem. Note that
        this method later on should be executable without pretraining."""
        print('Test start.')

        # string to owlready2 object conversion
        self.pos = {self.kb.str_to_instance_obj[i] for i in pos}
        self.neg = {self.kb.str_to_instance_obj[i] for i in neg}

        self.search_tree.quality_func.set_positive_examples(self.pos)
        self.search_tree.quality_func.set_negative_examples(self.neg)

        self.search_tree.quality_func.applied = 0

        # owlready obj to  integer conversion
        self.idx_pos = torch.LongTensor([self.kb.idx_of_instances[_] for _ in self.pos])
        self.idx_neg = torch.LongTensor([self.kb.idx_of_instances[_] for _ in self.neg])
        self.search_tree.heuristic_func = DrillHeuristic(pos=self.idx_pos, neg=self.idx_neg, model=self.model)
        self.search_tree.reset_tree()
        assert len(self.search_tree) == 0  # search tree is empty.
        # init pretrained agent as heuristic func.
        root = self.rho.getNode(self.start_class, root=True)
        self.assign_embeddings(root)

        self.search_tree.quality_func.apply(root)
        self.search_tree.add_root(root)
        goal_found = False
        for _ in range(10):
            node_to_expand = self.search_tree.get_most_promising()
            for ref in self.apply_rho(node_to_expand):

                self.assign_embeddings(ref)
                goal_found = self.search_tree.add_node(node=ref, refined_node=node_to_expand)
                if goal_found:
                    print(
                        'Goal found after {0} number of concepts tested.'.format(self.search_tree.quality_func.applied))
                    break
            if goal_found:
                break
        print('#### Top predictions after {0} number of concepts tested. ###'.format(
            self.search_tree.quality_func.applied))
        for i in self.search_tree.get_top_n(n=10):
            print(i)
        print('#######')


class PrepareBatchOfPrediction(torch.utils.data.Dataset):

    def __init__(self, current_state: torch.FloatTensor, next_state_batch: torch.Tensor, p: torch.FloatTensor,
                 n: torch.FloatTensor):
        self.S_Prime = next_state_batch
        self.S = current_state.expand(self.S_Prime.shape)
        self.Positives = p.expand(next_state_batch.shape)
        self.Negatives = n.expand(next_state_batch.shape)
        assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
        assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32
        # X.shape()=> batch_size,4,embeddingdim)
        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)
        num_points, depth, dim = self.X.shape
        self.X = self.X.view(num_points, depth, dim)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def get_all(self):
        return self.X


class PrepareBatchOfTraining(torch.utils.data.Dataset):

    def __init__(self, current_state_batch: torch.Tensor, next_state_batch: torch.Tensor, p: torch.Tensor,
                 n: torch.Tensor, q: torch.Tensor):
        self.S = current_state_batch
        self.S_Prime = next_state_batch
        self.y = q.view(len(q), 1)
        assert self.S.shape == self.S_Prime.shape
        assert len(self.y) == len(self.S)

        self.Positives = p.expand(next_state_batch.shape)
        self.Negatives = n.expand(next_state_batch.shape)
        assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
        assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32
        # X.shape()=> batch_size,4,embeddingdim)
        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)
        num_points, depth, dim = self.X.shape
        self.X = self.X.view(num_points, depth, dim)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
