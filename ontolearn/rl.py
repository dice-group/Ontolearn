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
from typing import List
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'reward'))


class DQLTrainer(BaseRLTrainer):
    def __init__(self, *,
                 knowledge_base,
                 refinement_operator,
                 search_tree,
                 reward_func,
                 train_data):
        super().__init__(reward_func=reward_func,
                         kb=knowledge_base,
                         rho=refinement_operator,
                         search_tree=search_tree,
                         train_data=train_data)

        self.model = Drill(num_of_indv=len(self.kb.thing.instances))
        self.model.eval()

        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(10000)

        self.str_to_obj_instance_mapping = dict()
        self.iter_bound = 1000
        self.min_length = 1
        self.size_of_path = 10
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
        """
        if np.random.random() < self.epsilon:  # Exploration vs Exploitation
            return random.choice(next_states)
        else:
            # we need to improve later.
            s = self.get_idx_instances(current_state)
            pred_q_max, next_state = -1, None
            with torch.no_grad():
                self.model.eval()
                # Later, we can remove looping and generate predicted Q values in one step.
                for s_prime in next_states:
                    pred = self.model.forward(s, self.get_idx_instances(s_prime), self.idx_pos, self.idx_neg).numpy()[0]
                    if pred > pred_q_max:
                        pred_q_max = pred
                        next_state = s_prime
            return next_state

    def get_idx_instances(self, node: Node) -> torch.LongTensor:
        """ Form torch Long Tensor from the index of the individuals beloning to a concept stored in
        node object."""
        return torch.LongTensor([self.kb.idx_of_instances[i] for i in node.concept.instances])

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(self.dataset_obj.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

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
        for th, experience in enumerate(state_pairs):
            e, e_next = experience
            temp_exp.append([self.get_idx_instances(e),
                             self.get_idx_instances(e_next),
                             self.idx_pos,
                             self.idx_neg,
                             max(rewards[th:])])  # given e, e_next, Q val is the max Q value reachable.
        return temp_exp

    def learn_from_replay_memory(self):
        # to break the correlation.
        random.shuffle(self.experiences)

        self.model.train()
        for exp in self.experiences:
            s, s_prime, pos, neg, q_val = exp
            q_val = torch.Tensor([q_val])

            predicted_q_val = self.model.forward(s, s_prime, pos, neg).squeeze()
            loss = self.model.loss(predicted_q_val, q_val)
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):

        print(self.kb)
        print(self.rho)
        print(self.search_tree)

        print(root)

        self.search_tree.set_positive_negative_examples(p=self.pos, n=self.neg, all_instances=self.kb.thing.instances)
        # init root
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.quality_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree.heuristic_func.apply(root)  # AccuracyOrTooWeak(n)
        self.search_tree[root] = root

        # 1. Enter search loop
        # 2. Get next most promissing node
        # 3. Refine it
        # 4. Add them into search tree if critertion are fulfilled.
        # 5. Go to 2.

        # training loop.
        for th in range(1, self.iter_bound):
            path_of_concepts, rewards = self.sequence_of_actions(root)
            self.epsilon -= 0.01
            if self.epsilon < 0:
                raise NotImplementedError
            # Collect experiences
            self.experiences.extend(self.form_experiences(path_of_concepts, rewards))
            if th == 10:
                self.learn_from_replay_memory()

        pass

    def train(self):
        """
        Agent trains on quasi ordered concept environment until one of the stopping criterion is fulfilled.
        """
        print('Training starts.')
        root = self.rho.getNode(self.start_class, root=True)

        # training loop.
        for th in range(1, self.iter_bound):
            path_of_concepts, rewards = self.sequence_of_actions(root)
            self.epsilon -= 0.01
            if self.epsilon < 0:
                print('Epsilon', self.epsilon)
                break
            # Collect experiences
            self.experiences.extend(self.form_experiences(path_of_concepts, rewards))
            if th == 10:
                print('Learning from replay memory starts.')
                self.learn_from_replay_memory()
                print('Learning from replay memory ends.')

    def start(self):
        for str_target_concept, examples in self.train_data.items():

            pos = set(examples['positive_examples'])
            neg = set(examples['negative_examples'])
            print('Target concept: {0}\t|Pos. Ex|:{1}\t|Neg. Ex|:{2}:'.format(str_target_concept, len(pos), len(neg)))

            # From string to owlready2 instance type conversion.
            # The following work can be done by https://docs.python.org/3/library/concurrent.futures.html
            for i in self.kb.thing.instances:
                self.str_to_obj_instance_mapping[get_full_iri(i)] = i
            # Write an argument in AbstractSearchtree.
            self.pos = {self.str_to_obj_instance_mapping[i] for i in pos}
            self.neg = {self.str_to_obj_instance_mapping[i] for i in neg}

            self.idx_pos = torch.from_numpy(np.array([self.kb.idx_of_instances[_] for _ in self.pos]))
            self.idx_neg = torch.from_numpy(np.array([self.kb.idx_of_instances[_] for _ in self.neg]))

            self.reward_func.set_positive_examples(self.pos)
            self.reward_func.set_negative_examples(self.neg)

            self.train()  # Let model to train to solve given problem

            self.test()  # Let test model to train to solve given problem


class Drill(nn.Module):
    def __init__(self, num_of_indv, n_dim=50):
        super(Drill, self).__init__()
        self.num_of_indv = num_of_indv
        self.n_dim = n_dim
        self.loss = torch.nn.MSELoss()

        self.embeddings = torch.nn.Embedding(self.num_of_indv, n_dim, padding_idx=0)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=20, kernel_size=2)
        self.maxpool2d_1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=2)

        self.bn1 = torch.nn.BatchNorm2d(4)
        self.bn2 = torch.nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(in_features=33000, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1)

    def forward(self, s, s_prime, p, n):
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

        x = self.fc3(self.fc2(self.fc1(x)))

        return F.relu(x)


class ReplayMemory(object):

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
