from abc import ABCMeta

from .base_rl_agent import BaseRLTrainer
from .util import get_full_iri
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
from typing import Set, AnyStr


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
            return torch.FloatTensor([.0001])  # How to quantifiy heuristc value of a node ?

        with torch.no_grad():
            return self.model.forward(parent_node.concept.idx_instances, node.concept.idx_instances, self.pos,
                                      self.neg).squeeze()

    def apply(self, node, parent_node=None):
        """ Assign predicted Q-value to node object."""
        predicted_q_val = self.score(node, parent_node)
        node.heuristic = predicted_q_val


class DQLTrainer(BaseRLTrainer):
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
                 train_data):
        super().__init__(kb=knowledge_base,
                         rho=refinement_operator,
                         quality_func=quality_func,
                         search_tree=search_tree,
                         reward_func=reward_func,
                         train_data=train_data)

        self.model = Drill(num_of_indv=len(self.kb.thing.instances))
        self.model.eval()
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(10000)

        self.iter_bound = 10
        self.learn_per_modulo_iter = 2  # 50%10==0 > 5 times.
        self.min_length = 1
        self.size_of_path = 4
        self.epsilon, self.epsilon_decay, self.epsilon_min = 1, 0.999, 0.01
        self.experiences = []

        self.pos, self.neg = None, None
        self.idx_pos, self.idx_neg = None, None

    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)
        self.search_tree.add_root(root)

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)
        return (self.rho.getNode(i, parent_node=node) for i in
                self.rho.refine(node, maxlength=len(node) + self.min_length))

    def next_node_to_expand(self, current_state, next_states) -> Node:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        TODO: vectorized the computation of exploitation.
        """
        if np.random.random() < self.epsilon:  # Exploration vs Exploitation
            next_state = random.choice(next_states)
            self.assign_idx_instances(next_state)
            return next_state
        else:
            self.assign_idx_instances(current_state)
            pred_q_max, next_state = -1, None
            with torch.no_grad():
                self.model.eval()
                # Later, we can remove looping and generate predicted Q values in one step.
                for s_prime in next_states:
                    self.assign_idx_instances(s_prime)
                    pred = \
                        self.model.forward(current_state.concept.idx_instances, s_prime.concept.idx_instances,
                                           self.idx_pos,
                                           self.idx_neg).numpy()[0]
                    if pred > pred_q_max:
                        pred_q_max = pred
                        next_state = s_prime
            return next_state

    def assign_idx_instances(self, node: Node):
        if node.concept.idx_instances is None:
            node.concept.idx_instances = torch.LongTensor([self.kb.idx_of_instances[i] for i in node.concept.instances])

    def sequence_of_actions(self, root):
        current_state = root
        path_of_concepts = []
        rewards = []

        for _ in range(self.size_of_path):
            next_states = [i for i in self.apply_rho(current_state)]

            if len(next_states) == 0:  # DEAD END, what to do ?
                break

            next_state = self.next_node_to_expand(current_state, next_states)
            path_of_concepts.append((current_state, next_state))
            rewards.append(self.reward_func.calculate(current_state, next_state))
            current_state = next_state

        return path_of_concepts, rewards

    def form_experiences(self, state_pairs: List, rewards: List) -> List:
        """
        Form experiences from a sequence of concepts and corresponding rewards.

        state_pairs - a list of tuples containing two consecutive states
        reward      - a list of reward.
        """

        temp_exp = []
        my_temp = []
        for th, experience in enumerate(state_pairs):
            e, e_next = experience
            my_temp.append((e, e_next, max(rewards[th:])))

            assert e.concept.idx_instances is not None
            assert e_next.concept.idx_instances is not None
            assert self.idx_pos is not None
            assert self.idx_neg is not None

            temp_exp.append([e.concept.idx_instances,
                             e_next.concept.idx_instances,
                             self.idx_pos,  # we might not need to store individual. at training time, we can stack it.
                             self.idx_neg,  #
                             max(rewards[th:])])  # given e, e_next, Q val is the max Q value reachable.

        """
        # For debugging purposes.
        for i in my_temp:
            s,s_prime,qmax=i
            print(s,'\t',s_prime,'\t',qmax)
        """

        return temp_exp

    def learn_from_replay_memory(self):
        # to break the correlation.
        random.shuffle(self.experiences)

        self.model.train()
        for m in range(10):
            total_loss = 0
            for exp in self.experiences:
                self.optimizer.zero_grad()
                s, s_prime, pos, neg, q_val = exp
                q_val = torch.Tensor([q_val])

                predicted_q_val = self.model.forward(s, s_prime, pos, neg)
                loss = self.model.loss(predicted_q_val, q_val)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    def start(self):
        """
        Training RL agent on given standard supervised learning problems.
        For each supervised learning problem
        1) Train RL agent.
        2) Use RL agent as Heuristic function.
        """
        for _ in range(10):
            for str_target_concept, examples in self.train_data.items():
                string_pos = set(examples['positive_examples'])
                string_neg = set(examples['negative_examples'])
                print('Target concept: {0}\t|Pos. Ex|:{1}\t|Neg. Ex|:{2}:'.format(str_target_concept, len(string_pos),
                                                                                  len(string_neg)))

                # string to owlready2 object conversion
                self.pos = {self.kb.str_to_instance_obj[i] for i in string_pos}
                self.neg = {self.kb.str_to_instance_obj[i] for i in string_neg}
                self.reward_func.pos = self.pos
                self.reward_func.neg = self.neg

                # owlready obj to  integer conversion
                self.idx_pos = torch.from_numpy(np.array([self.kb.idx_of_instances[_] for _ in self.pos]))
                self.idx_neg = torch.from_numpy(np.array([self.kb.idx_of_instances[_] for _ in self.neg]))

                self.train()  # Let model to train to solve given problem

                self.test(string_pos, string_neg)  # Let test model to train to solve given problem

    def train(self):
        """
        Agent trains on quasi ordered concept environment until one of the stopping criterion is fulfilled.
        """
        print('Training starts.')
        root = self.rho.getNode(self.start_class, root=True)

        self.assign_idx_instances(root)

        # training loop.
        for th in range(1, self.iter_bound):
            path_of_concepts, rewards = self.sequence_of_actions(root)
            self.epsilon -= 0.01
            if self.epsilon < 0:
                break
            # Collect experiences
            self.experiences.extend(self.form_experiences(path_of_concepts, rewards))
            if th % self.learn_per_modulo_iter == 0:
                self.learn_from_replay_memory()

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
        self.assign_idx_instances(root)

        root.heuristic = 0.0  # workaround.
        self.search_tree.quality_func.apply(root)
        self.search_tree.add_root(root)

        for _ in range(10):
            node_to_expand = self.search_tree.get_most_promising()
            for ref in self.apply_rho(node_to_expand):

                self.assign_idx_instances(ref)
                goal_found = self.search_tree.add_node(node=ref, refined_node=node_to_expand)
                if goal_found:
                    print(
                        'Goal found after {0} number of concepts tested.'.format(self.search_tree.quality_func.applied))
                    break

        print('#### Top predictions after {0} number of concepts tested. ###'.format(self.search_tree.quality_func.applied))
        for i in self.search_tree.get_top_n(n=10):
            print(i)

        print('#######')


class Drill(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_of_indv, n_dim=50):
        super(Drill, self).__init__()
        self.num_of_indv = num_of_indv
        self.n_dim = n_dim
        self.loss = torch.nn.MSELoss()

        self.embeddings = torch.nn.Embedding(self.num_of_indv, n_dim, padding_idx=0)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=4)
        self.maxpool2d_1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=4)

        self.bn1 = torch.nn.BatchNorm2d(4)
        self.bn2 = torch.nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(in_features=480, out_features=1)

    def forward(self, s, s_prime, p, n):
        # considering all and masking indv.
        emb_s = torch.zeros(self.num_of_indv, self.n_dim)
        emb_s_prime = torch.zeros(self.num_of_indv, self.n_dim)
        emb_p = torch.zeros(self.num_of_indv, self.n_dim)
        emb_n = torch.zeros(self.num_of_indv, self.n_dim)

        emb_s[s] = self.embeddings(s)
        emb_s_prime[s_prime] = self.embeddings(s_prime)
        emb_p[p] = self.embeddings(p)
        emb_n[n] = self.embeddings(n)

        x = torch.cat([emb_s.view(1, -1, self.num_of_indv, self.n_dim),
                       emb_s_prime.view(1, -1, self.num_of_indv, self.n_dim),
                       emb_p.view(1, -1, self.num_of_indv, self.n_dim),
                       emb_n.view(1, -1, self.num_of_indv, self.n_dim)], 1)

        x = self.bn1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool2d_1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2d_1(x)

        x = x.view(x.shape[0], -1)
        return self.fc1(x)  # F.relu(self.fc1(x))


class ReplayMemory(object):
    Transition = namedtuple('Transition',
                            ('state', 'next_state', 'reward'))

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
