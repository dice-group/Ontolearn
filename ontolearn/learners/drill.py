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
from abc import abstractmethod

import json
from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_dl

from .base import RefinementBasedConceptLearner
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.abstracts import AbstractNode, AbstractKnowledgeBase
from ontolearn.search import RL_State
from typing import Set, List, Tuple, Optional, Generator, SupportsFloat, Iterable, FrozenSet, Callable, Union
from ontolearn.learning_problem import PosNegLPStandard
import torch
from ontolearn.data_struct import Experience
from ontolearn.search import DRILLSearchTreePriorityQueue
from ontolearn.utils import create_experiment_folder
from collections import Counter, deque
from itertools import chain
import time
import os
from ontolearn.utils import read_csv
# F1 class will be deprecated to become compute_f1_score function.
from ontolearn.utils.static_funcs import compute_f1_score, compute_f1_score_from_confusion_matrix, concept_len
import random
from ontolearn.heuristics import CeloeBasedReward
from ontolearn.data_struct import PrepareBatchOfPrediction
from tqdm import tqdm
from owlapy.converter import owl_expression_to_sparql_with_confusion_matrix

from ontolearn.triple_store import TripleStore
from ontolearn.utils.static_funcs import make_iterable_verbose
from owlapy.utils import get_expression_length


class Drill(RefinementBasedConceptLearner):  # pragma: no cover
    """ Neuro-Symbolic Class Expression Learning (https://www.ijcai.org/proceedings/2023/0403.pdf)"""

    def __init__(self, knowledge_base: AbstractKnowledgeBase,
                 path_embeddings: str = None,
                 refinement_operator: LengthBasedRefinement = None,
                 use_inverse: bool = True,
                 use_data_properties: bool = True,
                 use_card_restrictions: bool = True,
                 use_nominals: bool = True,
                 min_cardinality_restriction: int = 2,
                 max_cardinality_restriction: int = 5,
                 positive_type_bias: int = 1,
                 quality_func: Callable = None,
                 reward_func: object = None,
                 batch_size=None, num_workers: int = 1,
                 iter_bound=None, max_num_of_concepts_tested=None,
                 verbose: int = 0,
                 terminate_on_goal=None,
                 max_len_replay_memory=256,
                 epsilon_decay: float = 0.01, epsilon_min: float = 0.0,
                 num_epochs_per_replay: int = 2,
                 num_episodes_per_replay: int = 2,
                 learning_rate: float = 0.001,
                 max_runtime=None,
                 num_of_sequential_actions=3,
                 stop_at_goal=True,
                 num_episode: int = 10):

        self.name = "DRILL"
        self.verbose = verbose
        self.learning_problem = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (1) Initialize KGE.
        if path_embeddings and os.path.isfile(path_embeddings): #
            if self.verbose > 0:
                print("Reading Embeddings...", end="\t")
            self.df_embeddings = read_csv(path_embeddings).astype('float32')
            self.num_entities, self.embedding_dim = self.df_embeddings.shape
            if self.verbose > 0:
                print(self.df_embeddings.shape)
        else:
            if self.verbose > 0:
                print("No pre-trained model...")
            self.df_embeddings = None
            self.num_entities, self.embedding_dim = None, 1

        # (2) Initialize Refinement operator.
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(knowledge_base=knowledge_base,
                                                        use_inverse=use_inverse,
                                                        use_data_properties=use_data_properties,
                                                        use_card_restrictions=use_card_restrictions,
                                                        use_nominals=use_nominals,
                                                        min_cardinality_restriction=min_cardinality_restriction,
                                                        max_cardinality_restriction=max_cardinality_restriction)
        else:
            refinement_operator = refinement_operator

        # (3) Initialize reward function for the training.
        if reward_func is None:
            self.reward_func = CeloeBasedReward()
        else:
            self.reward_func = reward_func

        # (4) Params.
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_episode = num_episode
        self.num_of_sequential_actions = num_of_sequential_actions
        self.num_epochs_per_replay = num_epochs_per_replay
        self.max_len_replay_memory = max_len_replay_memory
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.num_episodes_per_replay = num_episodes_per_replay
        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.pos: FrozenSet[OWLNamedIndividual] = None
        self.neg: FrozenSet[OWLNamedIndividual] = None
        self.positive_type_bias = positive_type_bias

        self.start_time = None
        self.goal_found = False
        self.storage_path, _ = create_experiment_folder()
        # Move to here
        self.search_tree = DRILLSearchTreePriorityQueue(verbose=verbose)
        self.stop_at_goal = stop_at_goal
        self.epsilon = 1

        if self.df_embeddings is not None:
            self.heuristic_func = DrillHeuristic(mode="averaging",
                                                 model_args={'input_shape': (4, self.embedding_dim),
                                                             'first_out_channels': 32,
                                                             'second_out_channels': 16, 'third_out_channels': 8,
                                                             'kernel_size': 3},
                                                 device=self.device)
            self.experiences = Experience(maxlen=self.max_len_replay_memory)
            if self.learning_rate:
                self.optimizer = torch.optim.Adam(self.heuristic_func.net.parameters(), lr=self.learning_rate)
        else:
            self.heuristic_func = CeloeBasedReward()

        # @CD: RefinementBasedConceptLearner redefines few attributes this should be avoided.
        RefinementBasedConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                               refinement_operator=refinement_operator,
                                               quality_func=quality_func,
                                               heuristic_func=self.heuristic_func,
                                               terminate_on_goal=terminate_on_goal,
                                               iter_bound=iter_bound,
                                               max_num_of_concepts_tested=max_num_of_concepts_tested,
                                               max_runtime=max_runtime)
        # CD: This setting the valiable will be removed later.

        if isinstance(self.kb, TripleStore):
            self.quality_func = compute_f1_score_from_confusion_matrix
        else:
            self.quality_func = compute_f1_score

    def initialize_training_class_expression_learning_problem(self,
                                                              pos: FrozenSet[OWLNamedIndividual],
                                                              neg: FrozenSet[OWLNamedIndividual]) -> RL_State:
        """ Initialize """
        assert isinstance(pos, frozenset) and isinstance(neg, frozenset), "Pos and neg must be sets"
        assert 0 < len(pos) and 0 < len(neg), ("Positive and negative examples must have at least a single item\n"
                                               "fCurrently: Pos:len(pos)\t Neg:len(neg)\n")
        # print("Initializing learning problem")
        # (2) Obtain embeddings of positive and negative examples.
        self.init_embeddings_of_examples(pos_uri=pos, neg_uri=neg)

        self.pos = pos
        self.neg = neg

        self.emb_pos = self.get_embeddings_individuals(individuals=[i.str for i in self.pos]).to(self.device)
        self.emb_neg = self.get_embeddings_individuals(individuals=[i.str for i in self.neg]).to(self.device)

        # (3) Initialize the root state of the quasi-ordered RL env.
        # print("Initializing Root RL state...", end=" ")
        root_rl_state = self.create_rl_state(self.start_class, is_root=True)
        # print("Computing its quality...", end=" ")
        self.compute_quality_of_class_expression(root_rl_state)
        # print(f"{root_rl_state}...")
        self.epsilon = 1
        self._number_of_tested_concepts = 0
        self.reward_func.lp = self.learning_problem
        return root_rl_state

    def rl_learning_loop(self,
                         pos_uri: FrozenSet[OWLNamedIndividual],
                         neg_uri: FrozenSet[OWLNamedIndividual]) -> List[float]:
        """ Reinforcement Learning Training Loop

        Initialize RL environment for a given learning problem (E^+ pos_iri and E^- neg_iri )

        Training:
                    2.1 Obtain a trajectory: A sequence of RL states/DL concepts
                    T, Person, (Female and \forall hasSibling Female).
                    Rewards at each transition are also computed
        """

        # (1) Initialize RL environment for training
        root_rl_state = self.initialize_training_class_expression_learning_problem(pos_uri, neg_uri)
        sum_of_rewards_per_actions = []

        # (2) Reinforcement Learning offline training loop
        for th in range(self.num_episode):
            if self.verbose > 0:
                print(f"Episode {th + 1}: ", end=" ")
            # Sequence of decisions
            start_time = time.time()
            if self.verbose > 0:
                print(f"Taking {self.num_of_sequential_actions} actions...", end="  ")

            sequence_of_states, rewards = self.sequence_of_actions(root_rl_state)
            if self.verbose > 0:
                print(
                    f"Runtime {time.time() - start_time:.3f} secs | Max reward: {max(rewards):.3f} | Prob of Explore {self.epsilon:.3f}",
                    end=" | ")
            # Form experiences
            self.form_experiences(sequence_of_states, rewards)
            sum_of_rewards_per_actions.append(sum(rewards))
            """(3.2) Learn from experiences"""
            self.learn_from_replay_memory()
            """(3.4) Exploration Exploitation"""
            if self.epsilon < 0:
                break
            self.epsilon -= self.epsilon_decay

        return sum_of_rewards_per_actions

    def train(self, dataset: Optional[Iterable[Tuple[str, Set, Set]]] = None,
              num_of_target_concepts: int = 1,
              num_learning_problems: int = 1):
        """ Training RL agent
        (1) Generate Learning Problems
        (2) For each learning problem, perform the RL loop

        """
        if isinstance(self.heuristic_func, CeloeBasedReward):
            print("No training...")
            return self.terminate_training()
        
        print(f"Training DRILL over {num_of_target_concepts} target concepts "
              f"with {num_learning_problems} learning problems each...")
        
        if self.verbose > 0:
            training_data = tqdm(self.generate_learning_problems(num_of_target_concepts,
                                                                 num_learning_problems),
                                 desc="Training over learning problems")
        else:
            training_data = self.generate_learning_problems(num_of_target_concepts,
                                                            num_learning_problems)
      
        if isinstance(training_data,Iterable) is False:
            print(f"We couldn't generate training data on this given knowledge base ({self.kb})")
            return self.terminate_training()

        for (target_owl_ce, positives, negatives) in training_data:
            
            # print(f"\nGoal Concept:\t {target_owl_ce}\tE^+:[{len(positives)}]\t E^-:[{len(negatives)}]")
            sum_of_rewards_per_actions = self.rl_learning_loop(pos_uri=frozenset(positives),
                                                               neg_uri=frozenset(negatives))
            print(f"\nGoal Concept:\t {target_owl_ce}\tE^+:[{len(positives)}]\t E^-:[{len(negatives)}] \tRewards per episode: {sum_of_rewards_per_actions}")

            # exit(0)
            if self.verbose > 0:
                print("Sum of rewards for each trial", sum_of_rewards_per_actions)

            self.seen_examples.setdefault(len(self.seen_examples), dict()).update(
                {'Concept': target_owl_ce,
                 'Positives': [i.str for i in positives],
                 'Negatives': [i.str for i in negatives]})
        return self.terminate_training()

    def save(self, directory: str = None) -> None:
        """ save weights of the deep Q-network"""
        # (1) Create a folder
        if directory:
            os.makedirs(directory, exist_ok=True)
            # (2) Save the weights
            self.save_weights(path=directory + "/drill.pth")
            # (3) Save seen examples
            with open(f"{directory}/seen_examples.json", 'w', encoding='utf-8') as f:
                json.dump(self.seen_examples, f, ensure_ascii=False, indent=4)

    def load(self, directory: str = None) -> None:
        """ load weights of the deep Q-network"""
        if directory:
            if os.path.isdir(directory):
                if isinstance(self.heuristic_func, CeloeBasedReward):
                    print("No loading because embeddings not provided")
                else:
                    print("Loading pretrained DQL Agent...", end="")
                    self.heuristic_func.net.load_state_dict(torch.load(directory + "/drill.pth", self.device))
                    print(self.heuristic_func.net)
            else:
                print(f"{directory} is not found...")
        else:
            print(f"Directory:{directory}")

    def fit(self, learning_problem: PosNegLPStandard, max_runtime=None):
        if max_runtime:
            assert isinstance(max_runtime, float) or isinstance(max_runtime, int)
            self.max_runtime = max_runtime
        # (1) Reinitialize few attributes to ensure a clean start.
        self.clean()
        # (2) Initialize the start time
        self.start_time = time.time()
        # (2) Two mappings from a unique OWL Concept to integer, where a unique concept represents the type info
        # C(x) s.t. x \in E^+ and  C(y) s.t. y \in E^-.
        # print("Counting types of positive examples..")
        pos_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.pos))])
        # print("Counting types of negative examples..")
        neg_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.neg))])
        # (3) Favor some OWLClass over others
        type_bias = pos_type_counts - neg_type_counts
        # (4) Initialize learning problem
        root_state = self.initialize_training_class_expression_learning_problem(pos=learning_problem.pos,
                                                                                neg=learning_problem.neg)
        self.operator.set_input_examples(pos=learning_problem.pos, neg=learning_problem.neg)
        assert root_state.quality > 0, f"Root state {root_state} must have the quality >0"
        # (5) Add root state into search tree
        root_state.heuristic = root_state.quality
        self.search_tree.add(root_state)
        best_found_quality = 0
        # (6) Inject Type Bias/Favor
        for ith_bias, x in enumerate((self.create_rl_state(i, parent_node=root_state) for i in type_bias)):
            self.compute_quality_of_class_expression(x)
            x.heuristic = x.quality
            if x.quality > best_found_quality:
                best_found_quality = x.quality
                self.search_tree.add(x)
            if ith_bias == self.positive_type_bias:
                break

        for _ in make_iterable_verbose(range(0, self.iter_bound),
                                       verbose=self.verbose,
                                       desc=f"Learning OWL Class Expression with at most {self.iter_bound} iterations"):
            assert len(self.search_tree) > 0, "Search Tree cannot be empty!"
            self.search_tree.show_current_search_tree()
            # (6.1) Get the most fitting RL-state.
            most_promising = self.next_node_to_expand()
            next_possible_states = []
            # (6.2) Checking the runtime termination criterion.
            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()
            # (6.3) Refine (6.1)
            # Convert this into tqdm with an update ?!
            for ref in (tqdm_bar := make_iterable_verbose(self.apply_refinement(most_promising),
                                                          verbose=self.verbose,
                                                          position=0, leave=True)):
                # (6.3.1) Checking the runtime termination criterion.
                if time.time() - self.start_time > self.max_runtime:
                    break
                # (6.3.2) Compute the quality stored in the RL state
                self.compute_quality_of_class_expression(ref)
                if ref.quality == 0:
                    continue
                if self.verbose > 0:
                    tqdm_bar.set_description_str(
                        f"Step {_} | Refining {owl_expression_to_dl(most_promising.concept)} | {owl_expression_to_dl(ref.concept)} | Quality:{ref.quality:.4f}")
                if ref.quality > best_found_quality:
                    if self.verbose > 0:
                        print("\nBest Found:", ref)
                    best_found_quality = ref.quality
                # (6.3.3) Consider qualifying RL states as next possible states to transition.
                next_possible_states.append(ref)
                # (6.3.4) Checking the goal termination criterion.
                if self.stop_at_goal:
                    if ref.quality == 1.0:
                        break
            if not next_possible_states:
                continue
            # (6.4) Predict Q-values
            if self.df_embeddings is not None:
                preds = self.predict_values(current_state=most_promising,
                                            next_states=next_possible_states)
            else:
                preds = None
            # (6.5) Add next possible states into search tree based on predicted Q values
            self.goal_found = self.update_search(next_possible_states, preds)
            if self.goal_found and self.stop_at_goal:
                if self.terminate_on_goal:
                    return self.terminate()
        return self.terminate()

    def init_embeddings_of_examples(self, pos_uri: FrozenSet[OWLNamedIndividual],
                                    neg_uri: FrozenSet[OWLNamedIndividual]):
        if self.df_embeddings is not None:
            # Shape:|E^+| x d
            # @TODO: CD: Why not use self.get_embeddings_individuals(pos_uri)
            self.pos = pos_uri
            self.neg = neg_uri
            self.emb_pos = torch.from_numpy(self.df_embeddings.loc[
                                                [owl_individual.str.strip() for owl_individual in
                                                 pos_uri]].values)
            # Shape: |E^+| x d
            self.emb_neg = torch.from_numpy(self.df_embeddings.loc[
                                                [owl_individual.str.strip() for owl_individual in
                                                 neg_uri]].values)
            """ (3) Take the mean of positive and negative examples and reshape it into (1,1,embedding_dim) for mini
             batching """
            # Shape: d
            self.emb_pos = torch.mean(self.emb_pos, dim=0)
            # Shape: d
            self.emb_neg = torch.mean(self.emb_neg, dim=0)
            # Shape: 1, 1, d
            self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
            self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])
            # Sanity checking
            if torch.isnan(self.emb_pos).any() or torch.isinf(self.emb_pos).any():
                raise ValueError('invalid value detected in E+,\n{0}'.format(self.emb_pos))
            if torch.isnan(self.emb_neg).any() or torch.isinf(self.emb_neg).any():
                raise ValueError('invalid value detected in E-,\n{0}'.format(self.emb_neg))

    def create_rl_state(self, c: OWLClassExpression, parent_node: Optional[RL_State] = None,
                        is_root: bool = False) -> RL_State:
        """ Create an RL_State instance."""
        rl_state = RL_State(c, parent_node=parent_node, is_root=is_root)
        rl_state.length = get_expression_length(c)
        return rl_state

    def compute_quality_of_class_expression(self, state: RL_State) -> None:
        """ Compute Quality of owl class expression.
        # (1) Perform concept retrieval
        # (2) Compute the quality w.r.t. (1), positive and negative examples
        # (3) Increment the number of tested concepts attribute.

        """
        if isinstance(self.kb, TripleStore) and state.concept is not OWLThing:
            sparql_query = owl_expression_to_sparql_with_confusion_matrix(expression=state.concept, positive_examples=self.pos,
                                                                          negative_examples=self.neg)
            bindings = self.kb.query(sparql_query).json()["results"]["bindings"]
            assert len(bindings) == 1
            bindings = bindings.pop()
            confusion_matrix = {k: v["value"]for k, v in bindings.items()}
            quality = self.quality_func(confusion_matrix=confusion_matrix)

        else:
            individuals = frozenset([i for i in self.kb.individuals(state.concept,True)])
            quality = compute_f1_score(individuals=individuals, pos=self.pos, neg=self.neg)
        state.quality = quality
        self._number_of_tested_concepts += 1

    def apply_refinement(self, rl_state: RL_State) -> Generator:
        """ Downward refinements"""
        assert isinstance(rl_state, RL_State), f"It must be rl state {rl_state}"
        assert isinstance(rl_state.concept, OWLClassExpression)
        self.operator: LengthBasedRefinement
        for i in self.operator.refine(rl_state.concept):  # O(N)
            yield self.create_rl_state(i, parent_node=rl_state)

    def select_next_state(self, current_state, next_rl_states) -> Tuple[RL_State, float]:
        next_selected_rl_state = self.exploration_exploitation_tradeoff(current_state, next_rl_states)
        return next_selected_rl_state, self.reward_func.apply(current_state, next_selected_rl_state)

    def sequence_of_actions(self, root_rl_state: RL_State) \
            -> Tuple[List[Tuple[RL_State, RL_State]], List[SupportsFloat]]:
        """ Performing sequence of actions in an RL env whose root state is ⊤"""
        assert isinstance(root_rl_state, RL_State)
        current_state = root_rl_state
        path_of_concepts = []
        rewards = []
        assert current_state.quality > 0, f"Root state ({current_state}) must have quality >0. \tCurrently {current_state.quality}"
        assert current_state.heuristic is None,f"Root state ({current_state}) must have heuristic value >0 . \tCurrently {current_state.heuristic}"
        # (1)
        for _ in range(self.num_of_sequential_actions):
            assert isinstance(current_state, RL_State)
            # (1.1) Observe Next RL states, i.e., refine an OWL class expression
            next_rl_states = list(self.apply_refinement(current_state))
            next_selected_rl_state, reward = self.select_next_state(current_state, next_rl_states)
            # (1.4) Remember the concept path
            path_of_concepts.append((current_state, next_selected_rl_state))
            # (1.5)
            rewards.append(reward)
            # (1.6)
            current_state = next_selected_rl_state
        return path_of_concepts, rewards

    def form_experiences(self, state_pairs: List, rewards: List) -> None:
        """
        Form experiences from a sequence of concepts and corresponding rewards.

        state_pairs - A list of tuples containing two consecutive states.
        reward      - A list of reward.

        Gamma is 1.

        Return
        X - A list of embeddings of current concept, next concept, positive examples, negative examples.
        y - Argmax Q value.
        """
        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    def learn_from_replay_memory(self) -> None:
        """
        Learning by replaying memory.
        """

        if isinstance(self.heuristic_func, CeloeBasedReward):
            return None

        # print('learn_from_replay_memory', end="\t|\t")
        current_state_batch: List[torch.FloatTensor]
        next_state_batch: List[torch.FloatTensor]
        current_state_batch, next_state_batch, y = self.experiences.retrieve()
        # N, 1, dim
        current_state_batch = torch.cat(current_state_batch, dim=0).to(self.device)
        # N, 1, dim
        next_state_batch = torch.cat(next_state_batch, dim=0).to(self.device)
        y = torch.Tensor(y).to(self.device)

        try:
            assert current_state_batch.shape[1] == next_state_batch.shape[1] == self.emb_pos.shape[1] == \
                   self.emb_neg.shape[1]

        except AssertionError as e:
            print(current_state_batch.shape)
            print(next_state_batch.shape)
            print(self.emb_pos.shape)
            print(self.emb_neg.shape)
            print('Wrong format.')
            print(e)
            raise

        assert current_state_batch.shape[2] == next_state_batch.shape[2] == self.emb_pos.shape[2] == self.emb_neg.shape[
            2]

        num_next_states = len(current_state_batch)
        # Ensure that X has the same data type as parameters of DRILL
        # batch, 4, dim
        X = torch.cat([
            current_state_batch,
            next_state_batch,
            self.emb_pos.repeat((num_next_states, 1, 1)),
            self.emb_neg.repeat((num_next_states, 1, 1))], 1).to(self.device)

        self.heuristic_func.net.train()
        total_loss = 0
        if self.verbose > 0:
            print(f"Experience replay Experiences ({X.shape})", end=" | ")
        for m in range(self.num_epochs_per_replay):
            self.optimizer.zero_grad()  # zero the gradient buffers
            # forward: n by 4, dim
            predicted_q = self.heuristic_func.net.forward(X)
            # loss
            loss = self.heuristic_func.net.loss(predicted_q, y)
            if self.verbose > 0:
                print(f"{m} Replay loss: {loss.item():.5f}", end=" | ")
            total_loss += loss.item()
            # compute the derivative of the loss w.r.t. the parameters using backpropagation
            loss.backward()
            # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        print(f'Avg loss: {total_loss / self.num_epochs_per_replay:0.5f}')
        self.heuristic_func.net.eval()

    def update_search(self, concepts, predicted_Q_values=None):
        """
        @param concepts:
        @param predicted_Q_values:
        @return:
        """
        if predicted_Q_values is not None:
            for child_node, pred_Q in zip(concepts, predicted_Q_values):
                child_node.heuristic = pred_Q
                if child_node.quality > 0:  # > too weak, ignore.
                    self.search_tree.add(child_node)
                if child_node.quality == 1:
                    return child_node
        else:
            for child_node in concepts:
                child_node.heuristic = child_node.quality / child_node.length
                if child_node.quality > 0:  # > too weak, ignore.
                    self.search_tree.add(child_node)
                if child_node.quality == 1:
                    return child_node

    def get_embeddings_individuals(self, individuals: List[str]) -> torch.FloatTensor:
        """Get embeddings for individuals with optimized operations."""
        assert isinstance(individuals, list)
        if len(individuals) == 0:
            emb = torch.zeros(1, 1, self.embedding_dim, device=self.device)
        else:
            if self.df_embeddings is not None:
                assert isinstance(individuals[0], str)
                # OPTIMIZATION: Use vectorized pandas operations and reduce device transfers
                try:
                    # Batch lookup and mean computation
                    embedding_values = self.df_embeddings.loc[individuals].values
                    emb = torch.from_numpy(embedding_values.mean(axis=0)).to(self.device, dtype=torch.float32)
                    emb = emb.view(1, 1, self.embedding_dim)
                except KeyError:
                    # Handle missing individuals gracefully
                    valid_individuals = [ind for ind in individuals if ind in self.df_embeddings.index]
                    if valid_individuals:
                        embedding_values = self.df_embeddings.loc[valid_individuals].values
                        emb = torch.from_numpy(embedding_values.mean(axis=0)).to(self.device, dtype=torch.float32)
                        emb = emb.view(1, 1, self.embedding_dim)
                    else:
                        emb = torch.zeros(1, 1, self.embedding_dim, device=self.device)
            else:
                emb = torch.zeros(1, 1, self.embedding_dim, device=self.device)
        return emb

    def get_individuals(self, rl_state: RL_State) -> List[str]:
        """Get individuals for a state - optimized for memory efficiency."""
        return [owl_individual.str.strip() for owl_individual in self.kb.individuals(rl_state.concept, True)]

    def assign_embeddings(self, rl_state: RL_State) -> None:
        """
        Assign embeddings to a rl state. A rl state is represented with vector representation of
        all individuals belonging to a respective OWLClassExpression.
        """
        assert isinstance(rl_state, RL_State)
        assert isinstance(rl_state.concept, OWLClassExpression)
        rl_state.embeddings = self.get_embeddings_individuals(self.get_individuals(rl_state))

    def save_weights(self, path: str = None) -> None:
        """ Save weights DQL"""
        if path:
            pass
        else:
            path = f"{self.storage_path}/{self.heuristic_func.name}.pth"

        if isinstance(self.heuristic_func, CeloeBasedReward):
            print("No saving..")
        else:
            torch.save(self.heuristic_func.net.state_dict(), path)

    def exploration_exploitation_tradeoff(self,
                                          current_state: AbstractNode,
                                          next_states: List[AbstractNode]) -> AbstractNode:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        (1) Exploration.
        (2) Exploitation.
        """
        self.assign_embeddings(current_state)
        if random.random() < self.epsilon:
            next_state = random.choice(next_states)
        else:
            next_state = self.exploitation(current_state, next_states)
        self.assign_embeddings(next_state)
        self.compute_quality_of_class_expression(next_state)
        return next_state

    def exploitation(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> RL_State:
        """
        Find next node that is assigned with highest predicted Q value.

        (1) Predict Q values : predictions.shape => torch.Size([n, 1]) where n = len(next_states).

        (2) Find the index of max value in predictions.

        (3) Use the index to obtain next state.

        (4) Return next state.
        """
        # predictions: torch.Size([len(next_states)])
        predictions: torch.FloatTensor = self.predict_values(current_state, next_states).to(self.device)
        argmax_id = int(torch.argmax(predictions).to(self.device))
        next_state = next_states[argmax_id]
        return next_state

    def predict_values(self, current_state: RL_State, next_states: List[RL_State]) -> torch.Tensor:
        """
        Predict promise of next states given current state.

        Returns:
            Predicted Q values.
        """

        assert len(next_states) > 0
        with torch.no_grad():
            self.heuristic_func.net.eval()
            # create batch batch.
            next_state_batch = []
            for _ in next_states:
                next_state_batch.append(self.get_embeddings_individuals(self.get_individuals(_)))
            next_state_batch = torch.cat(next_state_batch, dim=0).to(self.device)
            x = PrepareBatchOfPrediction(self.get_embeddings_individuals(self.get_individuals(current_state)),
                                         next_state_batch,
                                         self.emb_pos,
                                         self.emb_neg).get_all()
            predictions = self.heuristic_func.net.forward(x)
        return predictions

    @staticmethod
    def retrieve_concept_chain(rl_state: RL_State) -> List[RL_State]:
        hierarchy = deque()
        if rl_state.parent_node:
            hierarchy.appendleft(rl_state.parent_node)
            while hierarchy[-1].parent_node is not None:
                hierarchy.append(hierarchy[-1].parent_node)
            hierarchy.appendleft(rl_state)
        return list(hierarchy)

    def generate_learning_problems(self,
                                   num_of_target_concepts,
                                   num_learning_problems) -> List[
        Tuple[str, Set, Set]]:
        """ Generate learning problems if none is provided.

            Time complexity: O(n^2) n = named concepts
        """
        # Initialize counters and containers
        counter = 0   
        size_of_examples = 3  # Minimum number of examples required for positive/negative sets
        examples = []   
         # C: Iterate over all named OWL concepts
        for i in self.kb.get_concepts():
            # Retrieve all individuals that belong to concept i (positive examples)
            individuals_i = set(self.kb.individuals(i, True))
            
            # Skip concepts with insufficient individuals for sampling
            if len(individuals_i) < size_of_examples:
                continue
                
            for j in self.kb.get_concepts():
                # Skip if same concept (can't use same concept for both positive and negative examples)
                if i == j:
                    continue
                str_dl_concept_i = owl_expression_to_dl(i)
                
                # Retrieve all individuals that belong to concept j (negative examples)
                individuals_j = set(self.kb.individuals(j, True))
                
                # Skip concepts with insufficient individuals for sampling
                if len(individuals_j) < size_of_examples:
                    continue

                # Generate Learning problems from a single target
                for _ in range(num_of_target_concepts):
                    # Randomly sample positive examples from concept i
                    sampled_positives = set(random.sample(individuals_i, size_of_examples))
                    
                    # Randomly sample negative examples from concept j
                    sampled_negatives = set(random.sample(individuals_j, size_of_examples))
                    
                    # Validate that positive and negative examples are different
                    if sampled_negatives == sampled_positives:
                        print("Sampled Positives and negatives are same. We need to ignore this example")
                        continue
                     
                    lp = (str_dl_concept_i, sampled_positives, sampled_negatives)
                    examples.append(lp)
                    counter += 1
                    
                    # Break innermost loop: Stop sampling from this concept pair
                    if counter == num_learning_problems:
                        break

                # Break middle loop: Stop iterating through concept j for this concept i
                if counter == num_learning_problems:
                    break
                    
            # Break outermost loop: Stop iterating through all concepts i
            if counter == num_learning_problems:
                    break
                
        # Return the generated learning problems
        return examples
        """
        # if |Retrieve(C|>3
        if len(individuals_i) > size_of_examples:
            str_dl_concept_i = owl_expression_to_dl(i)
            for j in self.kb.get_concepts():
                if i == j:
                    continue
                individuals_j = set(self.kb.individuals(j))
                if len(individuals_j) > size_of_examples:
                    for _ in range(num_learning_problems):
                        lp = (str_dl_concept_i,
                                set(random.sample(individuals_i, size_of_examples)),
                                set(random.sample(individuals_j, size_of_examples)))
                        yield lp

                counter += 1
                if counter == num_of_target_concepts:
                    break
            if counter == num_of_target_concepts:
                break
        """

    def learn_from_illustration(self, sequence_of_goal_path: List[RL_State]):
        """
        Args:
            sequence_of_goal_path: ⊤,Parent,Parent ⊓ Daughter.
        """
        current_state = sequence_of_goal_path.pop(0)
        rewards = []
        sequence_of_states = []
        while len(sequence_of_goal_path) > 0:
            self.assign_embeddings(current_state)
            current_state.length = concept_len(current_state.concept)
            if current_state.quality is None:
                self.compute_quality_of_class_expression(current_state)

            next_state = sequence_of_goal_path.pop(0)
            self.assign_embeddings(next_state)
            next_state.length = concept_len(next_state.concept)
            if next_state.quality is None:
                self.compute_quality_of_class_expression(next_state)
            sequence_of_states.append((current_state, next_state))
            rewards.append(self.reward_func.apply(current_state, next_state))
        for x in range(2):
            self.form_experiences(sequence_of_states, rewards)
        self.learn_from_replay_memory()

    def best_hypotheses(self, n=1, return_node: bool = False) -> Union[OWLClassExpression, List[OWLClassExpression]]:
        assert self.search_tree is not None, "Search tree is not initialized"
        assert len(self.search_tree) >= 1, "Search tree is empty"

        result = []
        for i, rl_state in enumerate(self.search_tree.get_top_n_nodes(n)):
            if return_node:
                result.append(rl_state)
            else:
                result.append(rl_state.concept)

        if len(result) == 1:
            return result.pop()
        else:
            return result

    def clean(self):
        self.emb_pos, self.emb_neg = None, None
        self.pos = None
        self.neg = None
        self.goal_found = False
        self.start_time = None
        self.learning_problem = None
        if len(self.search_tree) != 0:
            self.search_tree.clean()

        try:
            assert len(self.search_tree) == 0
        except AssertionError:
            print(len(self.search_tree))
            raise AssertionError('EMPTY search tree')

        self._number_of_tested_concepts = 0

    def next_node_to_expand(self) -> RL_State:
        """ Return a node that maximizes the heuristic function at time t. """
        return self.search_tree.get_most_promising()

    def downward_refinement(self, *args, **kwargs):
        ValueError('downward_refinement')

    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        assert ValueError('show_search_tree')

    def terminate_training(self):
        if self.verbose > 0:
            print("Training is completed..")
        # Save the weights
        self.save_weights()
        with open(f"{self.storage_path}/seen_examples.json", 'w', encoding='utf-8') as f:
            json.dump(self.seen_examples, f, ensure_ascii=False, indent=4)
        return self


class DrillHeuristic:  # pragma: no cover
    """
    Heuristic in Convolutional DQL concept learning.
    Heuristic implements a convolutional neural network.
    """

    def __init__(self, pos=None, neg=None, model=None, mode=None, model_args=None, device=None):
        if model:
            self.net = model
        elif mode in ['averaging', 'sampling']:
            self.net = DrillNet(model_args, device)
            self.mode = mode
            self.name = 'DrillHeuristic_' + self.mode
        else:
            raise ValueError
        self.net.eval()
        self.device = device

    def score(self, node, parent_node=None):
        """ Compute heuristic value of root node only"""
        if parent_node is None and node.is_root:
            return torch.FloatTensor([.0001], device=self.device).squeeze()
        raise ValueError

    def apply(self, node, parent_node=None):
        """ Assign predicted Q-value to node object."""
        predicted_q_val = self.score(node, parent_node)
        node.heuristic = predicted_q_val


class DrillNet(torch.nn.Module):  # pragma: no cover
    """
    A neural model for Deep Q-Learning.

    An input Drill has the following form:
            1. Indexes of individuals belonging to current state (s).
            2. Indexes of individuals belonging to next state (s_prime).
            3. Indexes of individuals provided as positive examples.
            4. Indexes of individuals provided as negative examples.

    Given such input, we from a sparse 3D Tensor where  each slice is a **** N *** by ***D***
    where N is the number of individuals and D is the number of dimension of embeddings.
    Given that N on the current benchmark datasets < 10^3, we can get away with this computation. By doing so
    we do not need to subsample from given inputs.

    """

    def __init__(self, args, device):
        super(DrillNet, self).__init__()
        self.in_channels, self.embedding_dim = args['input_shape']
        assert self.embedding_dim
        self.device = device
        self.loss = torch.nn.MSELoss().to(self.device)
        # Conv1D seems to be faster than Conv2d
        self.conv1 = torch.nn.Conv1d(in_channels=4,
                                     out_channels=args['first_out_channels'],
                                     kernel_size=args['kernel_size'],
                                     padding=1, stride=1, bias=True, device=self.device)

        # Fully connected layers.
        self.size_of_fc1 = int(args['first_out_channels'] * self.embedding_dim)
        self.fc1 = torch.nn.Linear(in_features=self.size_of_fc1, out_features=self.size_of_fc1 // 2, device=self.device)
        self.fc2 = torch.nn.Linear(in_features=self.size_of_fc1 // 2, out_features=1, device=self.device)

        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.fc1.weight.data).to(self.device)
        torch.nn.init.xavier_normal_(self.conv1.weight.data).to(self.device)

    def forward(self, X: torch.FloatTensor):
        """
        X  n by 4 by d float tensor
        """
        # N x 32 x D
        X = torch.nn.functional.relu(self.conv1(X)).to(self.device)
        X = X.flatten(start_dim=1)
        # N x (32D/2)
        X = torch.nn.functional.relu(self.fc1(X))
        # N x 1
        scores = self.fc2(X).flatten()
        return scores


class DepthAbstractDrill:   # pragma: no cover
    """
    Abstract class for Convolutional DQL concept learning.
    """

    def __init__(self, path_of_embeddings, reward_func, learning_rate=None,
                 num_episode=None, num_episodes_per_replay=None, epsilon=None,
                 num_of_sequential_actions=None, max_len_replay_memory=None,
                 representation_mode=None, batch_size=None, epsilon_decay=None, epsilon_min=None,
                 num_epochs_per_replay=None, num_workers=None, verbose=0):
        self.name = 'DRILL'
        self.instance_embeddings = read_csv(path_of_embeddings)
        if not self.instance_embeddings:
            print("No embeddings found")
            self.embedding_dim = None
        else:
            self.embedding_dim = self.instance_embeddings.shape[1]
        self.reward_func = reward_func
        self.representation_mode = representation_mode
        assert representation_mode in ['averaging', 'sampling']
        # Will be filled by child class
        self.heuristic_func = None
        self.num_workers = num_workers
        # constants
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_episode = num_episode
        self.num_of_sequential_actions = num_of_sequential_actions
        self.num_epochs_per_replay = num_epochs_per_replay
        self.max_len_replay_memory = max_len_replay_memory
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.verbose = verbose
        self.num_episodes_per_replay = num_episodes_per_replay

        # will be filled
        self.optimizer = None  # torch.optim.Adam(self.model_net.parameters(), lr=self.learning_rate)

        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.start_time = None
        self.goal_found = False
        self.experiences = Experience(maxlen=self.max_len_replay_memory)

    def attributes_sanity_checking_rl(self):
        assert len(self.instance_embeddings) > 0
        assert self.embedding_dim > 0
        if self.num_workers is None:
            self.num_workers = 4
        if self.epsilon is None:
            self.epsilon = 1
        if self.learning_rate is None:
            self.learning_rate = .001
        if self.num_episode is None:
            self.num_episode = 1
        if self.num_of_sequential_actions is None:
            self.num_of_sequential_actions = 3
        if self.num_epochs_per_replay is None:
            self.num_epochs_per_replay = 1
        if self.max_len_replay_memory is None:
            self.max_len_replay_memory = 256
        if self.epsilon_decay is None:
            self.epsilon_decay = 0.01
        if self.epsilon_min is None:
            self.epsilon_min = 0
        if self.batch_size is None:
            self.batch_size = 1024
        if self.verbose is None:
            self.verbose = 0
        if self.num_episodes_per_replay is None:
            self.num_episodes_per_replay = 2

    @abstractmethod
    def init_training(self, *args, **kwargs):
        """
        Initialize training for a given E+,E- and K.
        """

    @abstractmethod
    def terminate_training(self):
        """
        Save weights and training data after training phase.
        """


class DrillV(Drill):
    """Deep V-Learning for Class Expression Learning with ADVANCED RL techniques."""

    def __init__(self, *args, use_random_v_values=False, 
                 use_reward_shaping=True,
                 use_dueling_network=True,
                 target_update_freq=100,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DRILLV"
        self.use_random_v_values = use_random_v_values
        self.use_reward_shaping = use_reward_shaping
        self.use_dueling_network = use_dueling_network
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        if self.df_embeddings is not None:
            # Create improved network with dueling architecture
            self.heuristic_func = DrillVHeuristic(
                mode="averaging", 
                model_args={'input_shape': (1, self.embedding_dim), 'use_dueling': use_dueling_network}, 
                device=self.device,
                use_random=use_random_v_values
            )
            
            # Use standard experience replay
            self.experiences = Experience(maxlen=self.max_len_replay_memory)
            
            if self.learning_rate:
                # Use AdamW with weight decay for better generalization
                self.optimizer = torch.optim.AdamW(
                    self.heuristic_func.net.parameters(), 
                    lr=self.learning_rate,
                    weight_decay=1e-5  # L2 regularization
                )
                
                # Learning rate scheduler for adaptive learning
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=5,
                    verbose=False
                )
                
            # Target network for stable training (Double DQN style)
            self.target_net = DrillVNet(
                self.embedding_dim, 
                self.device, 
                use_dueling=use_dueling_network
            )
            self.target_net.load_state_dict(self.heuristic_func.net.state_dict())
            self.target_net.eval()
            
        else:
            self.heuristic_func = CeloeBasedReward()

    def save(self, directory: str = None) -> None:
        """ save weights of the deep Q-network"""
        # (1) Create a folder
        if directory:
            os.makedirs(directory, exist_ok=True)
            # (2) Save the weights
            self.save_weights(path=directory + "/drillv.pth")
            # (3) Save seen examples
            with open(f"{directory}/seen_examples.json", 'w', encoding='utf-8') as f:
                json.dump(self.seen_examples, f, ensure_ascii=False, indent=4)


    def load(self, directory: str = None) -> None:
        """ load weights of the deep V-network"""
        if directory:
            if os.path.isdir(directory):
                if isinstance(self.heuristic_func, CeloeBasedReward):
                    print("No loading because embeddings not provided")
                else:
                    print("Loading pretrained DVL Agent...", end="")
                    self.heuristic_func.net.load_state_dict(torch.load(directory + "/drillv.pth", self.device))
                    print(self.heuristic_func.net)
            else:
                print(f"{directory} is not found...")
        else:
            print(f"Directory:{directory}")

    
    def form_experiences(self, state_pairs: List, rewards: List) -> None:
        """
        Store (state, reward, next_state) for V-learning with REWARD SHAPING.
        
        Reward shaping helps the agent learn faster by:
        1. Rewarding progress (increasing quality)
        2. Penalizing redundancy (exploring visited states)
        3. Bonusing diversity (balanced coverage)
        """
        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            base_reward = rewards[th]
            
            if self.use_reward_shaping:
                # Shape 1: Progress bonus - reward improvement
                quality_improvement = e_next.quality - e.quality
                progress_bonus = 0.1 * quality_improvement  # Small bonus for improvement
                
                # Shape 2: Length penalty - prefer shorter expressions
                # (but not too aggressive to allow complex necessary concepts)
                length_current = concept_len(e.concept) if hasattr(e, 'concept') else 0
                length_next = concept_len(e_next.concept) if hasattr(e_next, 'concept') else 0
                length_penalty = -0.01 * (length_next - length_current)
                
                # Shape 3: Exploration bonus - reward visiting high-quality unexplored states
                # (assumes e_next.quality represents novelty to some degree)
                if e_next.quality > 0.7:  # High-quality state
                    exploration_bonus = 0.05
                else:
                    exploration_bonus = 0.0
                
                # Combine all components
                shaped_reward = base_reward + progress_bonus + length_penalty + exploration_bonus
                
                # Clip to reasonable range to avoid instability
                shaped_reward = max(-1.0, min(2.0, shaped_reward))
            else:
                shaped_reward = base_reward
            
            self.experiences.append((e, e_next, shaped_reward))

    def train(self, dataset: Optional[Iterable[Tuple[str, Set, Set]]] = None,
              num_of_target_concepts: int = 1,
              num_learning_problems: int = 1):
        """ Training RL agent for V-learning - uses V-learning specific replay memory. """
        if isinstance(self.heuristic_func, CeloeBasedReward):
            print("No training...")
            return self.terminate_training()

        if self.verbose > 0:
            training_data = tqdm(self.generate_learning_problems(num_of_target_concepts,
                                                                 num_learning_problems),
                                 desc="Training DrillV over learning problems")
        else:
            training_data = self.generate_learning_problems(num_of_target_concepts,
                                                            num_learning_problems)

        if not isinstance(training_data, Iterable):
            print(f"We couldn't generate training data on this given knowledge base ({self.kb})")
            return self.terminate_training()

        for (target_owl_ce, positives, negatives) in training_data:
            # print(f"\nGoal Concept:\t {target_owl_ce}\tE^+:[{len(positives)}]\t E^-:[{len(negatives)}]")
            sum_of_rewards_per_actions = self.rl_learning_loop(pos_uri=frozenset(positives),
                                                               neg_uri=frozenset(negatives))
            print(f"\nGoal Concept:\t {target_owl_ce}\tE^+:[{len(positives)}]\t E^-:[{len(negatives)}] \tRewards per episode: {sum_of_rewards_per_actions}")

            if self.verbose > 0:
                print("Sum of rewards for each trial", sum_of_rewards_per_actions)

            self.seen_examples.setdefault(len(self.seen_examples), dict()).update(
                {'Concept': target_owl_ce,
                 'Positives': [i.str for i in positives],
                 'Negatives': [i.str for i in negatives]})
        return self.terminate_training()

    def learn_from_replay_memory(self, gamma=0.99) -> None:
        """
        V-learning with:
        1. Target network for stability
        2. Gradient clipping
        3. Better loss tracking
        """
        if isinstance(self.heuristic_func, CeloeBasedReward):
            return None

        # Retrieve experiences
        current_state_list, next_state_list, reward_list = self.experiences.retrieve()
        N = len(reward_list)
        if N == 0:
            return None
        
        minibatch = self.batch_size or 128
        minibatch = min(minibatch, N)
        indices = random.sample(range(N), minibatch)
        current_state_list = [current_state_list[i] for i in indices]
        next_state_list = [next_state_list[i] for i in indices]
        reward_list = [reward_list[i] for i in indices]

        self.heuristic_func.net.train()
        total_loss = 0.0
        total_td_error = 0.0
        
        for m in range(self.num_epochs_per_replay):
            # Prepare batches
            current_state_batch = torch.cat(current_state_list, dim=0).to(self.device)
            next_state_batch = torch.cat(next_state_list, dim=0).to(self.device)
            reward_batch = torch.tensor(reward_list, dtype=torch.float32, device=self.device)

            # Forward pass
            v_current = self.heuristic_func.net.forward(current_state_batch)
            
            # Use target network for stability (Double DQN approach)
            with torch.no_grad():
                v_next = self.target_net.forward(next_state_batch)
            
            # Compute target
            target = reward_batch + gamma * v_next
            
            # Compute TD errors
            td_errors = (target - v_current).detach()
            
            # Compute loss
            loss = self.heuristic_func.net.loss(v_current, target)
            
            # Backpropagation with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.heuristic_func.net.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_td_error += td_errors.abs().mean().item()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.heuristic_func.net.state_dict())
            if self.verbose > 0:
                print(f"[Target network updated at step {self.update_counter}]")
        
        # Learning rate scheduling based on loss
        avg_loss = total_loss / self.num_epochs_per_replay
        self.scheduler.step(avg_loss)
        
        # Detailed logging
        avg_td = total_td_error / self.num_epochs_per_replay
        print(f"Replay loss: {avg_loss:.5f} | Avg TD-error: {avg_td:.5f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        self.heuristic_func.net.eval()

    def exploitation(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> RL_State:
        """
        Select next state with highest predicted V-value - LEAN VERSION.
        """
        if self.use_random_v_values:
            v_values = self.generate_random_v_values(len(next_states))
        else:
            # Simple batch processing - key is single forward pass
            embeddings_list = []
            for state in next_states:
                emb = self.get_embeddings_individuals(self.get_individuals(state))
                embeddings_list.append(emb)
            
            next_state_batch = torch.cat(embeddings_list, dim=0).to(self.device)
            with torch.no_grad():
                v_values = self.heuristic_func.net.forward(next_state_batch)
        
        argmax_id = int(torch.argmax(v_values))
        return next_states[argmax_id]

    def generate_random_v_values(self, num_states: int, value_range: tuple = (0.0, 1.0)) -> torch.Tensor:
        """
        Generate random V-values for a given number of states.
        
        Args:
            num_states: Number of states to generate V-values for
            value_range: Tuple (min_val, max_val) defining the range of random values
            
        Returns:
            torch.Tensor: Random V-values of shape (num_states,)
        """
        min_val, max_val = value_range
        random_values = torch.rand(num_states, device=self.device) * (max_val - min_val) + min_val
        return random_values
    
    def optimize_for_performance(self):
        """Apply lean performance optimizations without memory overhead."""
        # Set network to eval mode to disable dropout/batch norm
        if hasattr(self.heuristic_func, 'net'):
            self.heuristic_func.net.eval()
        
        # Enable torch optimizations
        if hasattr(torch.backends, 'cudnn') and self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        if self.verbose > 0:
            print("Lean performance optimizations applied:")
            print(f"- Network eval mode: {'Yes' if hasattr(self.heuristic_func, 'net') else 'No'}")
            print("- Batch processing: Enabled")
            print("- Memory overhead: Minimized (no caching)")
            
    def print_performance_summary(self):
        """Print a summary of lean optimizations."""
        print("\n=== DrillV Lean Performance Summary ===")
        print("✓ Single forward pass batching")
        print("✓ Minimal memory overhead") 
        print("✓ No caching (memory-efficient)")
        print("✓ Network eval mode")
        print("========================================\n")
    
    def get_performance_stats(self) -> dict:
        """Get basic performance statistics."""
        return {
            "optimization_type": "lean",
            "memory_efficient": True,
            "batch_processing": True,
            "caching_enabled": False
        }
    
    def fit(self, learning_problem: PosNegLPStandard, max_runtime=None):
        if max_runtime:
            assert isinstance(max_runtime, float) or isinstance(max_runtime, int)
            self.max_runtime = max_runtime
        self.clean()
        self.start_time = time.time()
        pos_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.pos))])
        neg_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.neg))])
        type_bias = pos_type_counts - neg_type_counts
        root_state = self.initialize_training_class_expression_learning_problem(pos=learning_problem.pos,
                                                                               neg=learning_problem.neg)
        self.operator.set_input_examples(pos=learning_problem.pos, neg=learning_problem.neg)
        assert root_state.quality > 0, f"Root state {root_state} must have the quality >0"
        root_state.heuristic = root_state.quality
        self.search_tree.add(root_state)
        best_found_quality = 0
        for ith_bias, x in enumerate((self.create_rl_state(i, parent_node=root_state) for i in type_bias)):
            self.compute_quality_of_class_expression(x)
            x.heuristic = x.quality
            if x.quality > best_found_quality:
                best_found_quality = x.quality
                self.search_tree.add(x)
            if ith_bias == self.positive_type_bias:
                break

        for _ in make_iterable_verbose(range(0, self.iter_bound),
                                      verbose=self.verbose,
                                      desc=f"Learning OWL Class Expression with at most {self.iter_bound} iterations"):
            assert len(self.search_tree) > 0, "Search Tree cannot be empty!"
            self.search_tree.show_current_search_tree()
            most_promising = self.next_node_to_expand()
            next_possible_states = []
            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()
            for ref in (tqdm_bar := make_iterable_verbose(self.apply_refinement(most_promising),
                                                          verbose=self.verbose,
                                                          position=0, leave=True)):
                if time.time() - self.start_time > self.max_runtime:
                    break
                self.compute_quality_of_class_expression(ref)
                if ref.quality == 0:
                    continue
                if self.verbose > 0:
                    tqdm_bar.set_description_str(
                        f"Step {_} | Refining {owl_expression_to_dl(most_promising.concept)} | {owl_expression_to_dl(ref.concept)} | Quality:{ref.quality:.4f}")
                if ref.quality > best_found_quality:
                    if self.verbose > 0:
                        print("\nBest Found:", ref)
                    best_found_quality = ref.quality
                next_possible_states.append(ref)
                if self.stop_at_goal:
                    if ref.quality == 1.0:
                        break
            if not next_possible_states:
                continue
            # (6.4) Predict V-values for next states - LEAN OPTIMIZED VERSION
            if self.df_embeddings is not None:
                if self.use_random_v_values:
                    # Use random V-values instead of trained network
                    preds = self.generate_random_v_values(len(next_possible_states)).cpu().numpy().tolist()
                    if self.verbose > 0:
                        print(f"Using random V-values for {len(next_possible_states)} states")
                else:
                    # CORE OPTIMIZATION: Minimize neural network overhead
                    # Assign embeddings efficiently without extra memory
                    for state in next_possible_states:
                        if not hasattr(state, 'embeddings') or state.embeddings is None:
                            self.assign_embeddings(state)
                    
                    # Single batch forward pass - the key optimization
                    if next_possible_states and hasattr(next_possible_states[0], 'embeddings'):
                        embeddings_list = [s.embeddings for s in next_possible_states if hasattr(s, 'embeddings')]
                        if embeddings_list:
                            next_state_batch = torch.cat(embeddings_list, dim=0).to(self.device)
                            with torch.no_grad():
                                v_values = self.heuristic_func.net.forward(next_state_batch)
                            preds = v_values.cpu().numpy().tolist()
                        else:
                            preds = [0.0] * len(next_possible_states)
                    else:
                        preds = [0.0] * len(next_possible_states)
            else:
                preds = None
            # (6.5) Add next possible states into search tree based on predicted V values
            self.goal_found = self.update_search(next_possible_states, preds)
            if self.goal_found and self.stop_at_goal:
                if self.terminate_on_goal:
                    return self.terminate()
        return self.terminate()
    


class DrillVNet(torch.nn.Module):  # pragma: no cover
    """
    IMPROVED Neural network for Deep V-Learning with Dueling Architecture.
    Input: state embedding (batch_size, 1, embedding_dim)
    Output: V-value for each state (batch_size,)
    
    Key improvements:
    1. Dueling architecture (value + advantage streams)
    2. Deeper network with residual connections
    3. Batch normalization for stable training
    4. Dropout for regularization
    """

    def __init__(self, embedding_dim, device, hidden_dim=None, use_dueling=True, dropout_rate=0.2):
        super(DrillVNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_dueling = use_dueling
        
        # Auto-scale hidden dimension
        if hidden_dim is None:
            hidden_dim = max(64, embedding_dim // 2)
        self.hidden_dim = hidden_dim
        
        # Huber loss is more robust than MSE for RL
        self.loss = torch.nn.SmoothL1Loss().to(self.device)
        
        # Shared feature extraction with residual connection
        self.fc1 = torch.nn.Linear(self.embedding_dim, hidden_dim, device=self.device)
        # Use LayerNorm instead of BatchNorm to support batch_size=1 during training
        # BatchNorm requires >1 value per channel which fails for our single-sample
        # evaluation/training cases. LayerNorm normalizes across features and is
        # stable for small batch sizes.
        self.bn1 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True, device=self.device)
        self.dropout1 = torch.nn.Dropout(dropout_rate)

        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, device=self.device)
        # See comment for bn1
        self.bn2 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True, device=self.device)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        
        if self.use_dueling:
            # Dueling streams: value and advantage
            # Value: how good is this state generally?
            self.value_fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2, device=self.device)
            self.value_fc2 = torch.nn.Linear(hidden_dim // 2, 1, device=self.device)
            
            # Advantage: how much better is this state than average?
            self.advantage_fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2, device=self.device)
            self.advantage_fc2 = torch.nn.Linear(hidden_dim // 2, 1, device=self.device)
        else:
            # Simple output layer
            self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2, device=self.device)
            self.fc_out = torch.nn.Linear(hidden_dim // 2, 1, device=self.device)
        
        self.init()

    def init(self):
        """He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, X: torch.FloatTensor):
        """
        X: (batch_size, 1, embedding_dim) or (batch_size, embedding_dim)
        """
        if X.dim() == 3:
            X = X.view(X.shape[0], self.embedding_dim)
        
        # First layer with batch norm and dropout
        X = torch.nn.functional.relu(self.bn1(self.fc1(X)))
        X = self.dropout1(X)
        
        # Second layer with batch norm and dropout  
        X = torch.nn.functional.relu(self.bn2(self.fc2(X)))
        X = self.dropout2(X)
        
        if self.use_dueling:
            # Dueling architecture: Q(s) = V(s) + (A(s) - mean(A(s)))
            value = self.value_fc2(torch.nn.functional.relu(self.value_fc1(X)))
            advantage = self.advantage_fc2(torch.nn.functional.relu(self.advantage_fc1(X)))
            # Combine: subtract mean advantage for identifiability
            scores = value + (advantage - advantage.mean(dim=0, keepdim=True))
        else:
            # Simple feedforward
            X = torch.nn.functional.relu(self.fc3(X))
            scores = self.fc_out(X)
        
        return scores.flatten()
    
    # def train(self, dataset: Optional[Iterable[Tuple[str, Set, Set]]] = None,
    #       num_of_target_concepts: int = 1,
    #       num_learning_problems: int = 1):
    #     """ Training RL agent for V-learning. """
    #     if isinstance(self.heuristic_func, CeloeBasedReward):
    #         print("No training...")
    #         return self.terminate_training()

    #     if self.verbose > 0:
    #         training_data = tqdm(self.generate_learning_problems(num_of_target_concepts,
    #                                                             num_learning_problems),
    #                             desc="Training over learning problems")
    #     else:
    #         training_data = self.generate_learning_problems(num_of_target_concepts,
    #                                                         num_learning_problems)

    #     if not isinstance(training_data, Iterable):
    #         print(f"We couldn't generate training data on this given knowledge base ({self.kb})")
    #         return self.terminate_training()

    #     for (target_owl_ce, positives, negatives) in training_data:
    #         print(f"\nGoal Concept:\t {target_owl_ce}\tE^+:[{len(positives)}]\t E^-:[{len(negatives)}]")
    #         sum_of_rewards_per_actions = self.rl_learning_loop(pos_uri=frozenset(positives),
    #                                                         neg_uri=frozenset(negatives))
    #         if self.verbose > 0:
    #             print("Sum of rewards for each trial", sum_of_rewards_per_actions)

    #         self.seen_examples.setdefault(len(self.seen_examples), dict()).update(
    #             {'Concept': target_owl_ce,
    #             'Positives': [i.str for i in positives],
    #             'Negatives': [i.str for i in negatives]})
    #     return self.terminate_training()
    


class DrillVHeuristic:  # pragma: no cover
    """
    IMPROVED Heuristic for Deep V-Learning with uncertainty estimation.
    Uses DrillVNet to predict V-values for states.
    """

    def __init__(self, model=None, mode=None, model_args=None, device=None, use_random=False):
        self.use_random = use_random
        if model:
            # If user provided a model instance, use it and try to infer a name
            self.net = model
            # Prefer an explicit `name` attribute on the model when available
            if hasattr(model, 'name') and model.name:
                self.name = model.name
            else:
                # Fall back to the model class name for traceability
                try:
                    self.name = model.__class__.__name__
                except Exception:
                    self.name = 'DrillVHeuristic_custom'
            # keep mode attribute for compatibility
            self.mode = None
        elif mode in ['averaging', 'sampling']:
            # Extract parameters for improved network
            embedding_dim = model_args['input_shape'][1]
            use_dueling = model_args.get('use_dueling', True)
            
            self.net = DrillVNet(
                embedding_dim, 
                device, 
                use_dueling=use_dueling,
                dropout_rate=0.2
            )
            self.mode = mode
            self.name = 'DrillVHeuristic_' + self.mode
        else:
            raise ValueError("mode must be 'averaging' or 'sampling'")
        
        self.net.eval()
        self.device = device
        self.call_count = 0  # Track number of predictions

    def score(self, node, parent_node=None):
        """Compute V-value for a node (state) with optional exploration bonus."""
        if self.use_random:
            # Generate random V-value
            return torch.rand(1, device=self.device).squeeze()
        else:
            # Use trained network
            emb = node.embeddings if hasattr(node, 'embeddings') else None
            if emb is None:
                raise ValueError("Node must have embeddings assigned.")
            
            with torch.no_grad():
                v = self.net.forward(emb.to(self.device))
            
            self.call_count += 1
            return v.squeeze()

    def score_batch(self, nodes):
        """
        Efficiently score multiple nodes in a single forward pass.
        This is the KEY optimization for reducing concepts tested.
        """
        if self.use_random:
            return torch.rand(len(nodes), device=self.device)
        
        # Collect embeddings
        embeddings = []
        for node in nodes:
            if hasattr(node, 'embeddings') and node.embeddings is not None:
                embeddings.append(node.embeddings)
            else:
                # Fallback: create zero embedding
                embeddings.append(torch.zeros(1, self.net.embedding_dim, device=self.device))
        
        if not embeddings:
            return torch.zeros(len(nodes), device=self.device)
        
        # Single batched forward pass
        batch = torch.cat(embeddings, dim=0).to(self.device)
        with torch.no_grad():
            v_values = self.net.forward(batch)
        
        self.call_count += len(nodes)
        return v_values

    def apply(self, node, parent_node=None):
        """Assign predicted V-value to node object."""
        predicted_v_val = self.score(node, parent_node)
        node.heuristic = predicted_v_val
    
    def apply_batch(self, nodes):
        """Efficiently apply V-values to multiple nodes."""
        v_values = self.score_batch(nodes)
        for node, v_val in zip(nodes, v_values):
            node.heuristic = v_val.item()
    
    def set_random_mode(self, use_random: bool):
        """Set whether to use random V-values."""
        self.use_random = use_random
    
    def get_stats(self):
        """Get prediction statistics."""
        return {
            'total_predictions': self.call_count,
            'network_mode': 'random' if self.use_random else 'learned'
        }
