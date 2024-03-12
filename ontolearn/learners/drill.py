from ontolearn.base_concept_learner import RefinementBasedConceptLearner
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.abstracts import AbstractScorer, AbstractNode
from ontolearn.search import RL_State
from typing import Set, List, Tuple, Optional, Generator, SupportsFloat, Iterable, FrozenSet
from owlapy.model import OWLNamedIndividual, OWLClassExpression
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
import torch
from ontolearn.data_struct import Experience
from ontolearn.search import DRILLSearchTreePriorityQueue
from ontolearn.utils import create_experiment_folder
from collections import Counter, deque
from itertools import chain
import time
import dicee
import os
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.metrics import F1
import random
from ontolearn.heuristics import CeloeBasedReward
import torch
from ontolearn.data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction

class Drill(RefinementBasedConceptLearner):
    """ Neuro-Symbolic Class Expression Learning (https://www.ijcai.org/proceedings/2023/0403.pdf)"""

    def __init__(self, knowledge_base,
                 path_pretrained_kge: str = None,
                 path_pretrained_drill: str = None,
                 refinement_operator: LengthBasedRefinement = None,
                 use_inverse=True,
                 use_data_properties=True,
                 use_card_restrictions=True,
                 card_limit=10,
                 nominals=True,
                 quality_func: AbstractScorer = None,
                 reward_func: object = None,
                 batch_size=None, num_workers: int = 1, pretrained_model_name=None,
                 iter_bound=None, max_num_of_concepts_tested=None, verbose: int = 0, terminate_on_goal=None,
                 max_len_replay_memory=256,
                 epsilon_decay: float = 0.01, epsilon_min: float = 0.0,
                 num_epochs_per_replay: int = 100,
                 num_episodes_per_replay: int = 2, learning_rate: float = 0.001,
                 max_runtime=None,
                 num_of_sequential_actions=3,
                 stop_at_goal=True,
                 num_episode=10):

        self.name = "DRILL"
        self.learning_problem = None
        # (1) Initialize KGE.
        assert path_pretrained_drill is None, "Not implemented the integration of using pre-trained model"
        if path_pretrained_kge is not None and os.path.isdir(path_pretrained_kge):
            self.pre_trained_kge = dicee.KGE(path=path_pretrained_kge)
            self.embedding_dim = self.pre_trained_kge.configs["embedding_dim"]
        else:
            print("No pre-trained model...")
            self.pre_trained_kge = None
            self.embedding_dim = None

        # (2) Initialize Refinement operator.
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(knowledge_base=knowledge_base,
                                                        use_data_properties=use_data_properties,
                                                        use_card_restrictions=use_card_restrictions,
                                                        card_limit=card_limit,
                                                        use_inverse=use_inverse,
                                                        nominals=nominals)
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
        self.verbose = verbose
        self.num_episodes_per_replay = num_episodes_per_replay
        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.start_time = None
        self.goal_found = False
        self.storage_path, _ = create_experiment_folder()
        self.search_tree = DRILLSearchTreePriorityQueue()
        self.renderer = DLSyntaxObjectRenderer()
        self.stop_at_goal=stop_at_goal

        if self.pre_trained_kge:
            self.representation_mode = "averaging"
            self.sample_size = 1
            self.heuristic_func = DrillHeuristic(mode=self.representation_mode,
                                                 model_args={'input_shape': (4 * self.sample_size, self.embedding_dim),
                                                             'first_out_channels': 32,
                                                             'second_out_channels': 16, 'third_out_channels': 8,
                                                             'kernel_size': 3})
            self.experiences = Experience(maxlen=self.max_len_replay_memory)
            self.epsilon = 1

            if self.learning_rate:
                self.optimizer = torch.optim.Adam(self.heuristic_func.net.parameters(), lr=self.learning_rate)

            if pretrained_model_name:
                self.pre_trained_model_loaded = True
                self.heuristic_func.net.load_state_dict(torch.load(pretrained_model_name, torch.device('cpu')))
            else:
                self.pre_trained_model_loaded = False
        else:
            self.heuristic_func = CeloeBasedReward()
            self.representation_mode = None

        RefinementBasedConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                               refinement_operator=refinement_operator,
                                               quality_func=quality_func,
                                               heuristic_func=self.heuristic_func,
                                               terminate_on_goal=terminate_on_goal,
                                               iter_bound=iter_bound,
                                               max_num_of_concepts_tested=max_num_of_concepts_tested,
                                               max_runtime=max_runtime)

    def initialize_class_expression_learning_problem(self, pos: Set[OWLNamedIndividual], neg: Set[OWLNamedIndividual]):
        """
            Determine the learning problem and initialize the search.
            1) Convert the string representation of an individuals into the owlready2 representation.
            2) Sample negative examples if necessary.
            3) Initialize the root and search tree.
            """
        self.clean()
        assert 0 < len(pos) and 0 < len(neg)

        # 1.
        # Generate a Learning Problem
        self.learning_problem = PosNegLPStandard(pos=set(pos), neg=set(neg)).encode_kb(self.kb)
        # 2. Obtain embeddings of positive and negative examples.
        if self.pre_trained_kge is None:
            self.emb_pos = None
            self.emb_neg = None
        else:
            self.emb_pos = self.pre_trained_kge.get_entity_embeddings([owl_indv.get_iri().as_str() for owl_indv in pos])
            self.emb_neg = self.pre_trained_kge.get_entity_embeddings([owl_indv.get_iri().as_str() for owl_indv in neg])

            # (3) Take the mean of positive and negative examples and reshape it into (1,1,embedding_dim) for mini batching.
            self.emb_pos = torch.mean(self.emb_pos, dim=0)
            self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
            self.emb_neg = torch.mean(self.emb_neg, dim=0)
            self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])
            # Sanity checking
            if torch.isnan(self.emb_pos).any() or torch.isinf(self.emb_pos).any():
                raise ValueError('invalid value detected in E+,\n{0}'.format(self.emb_pos))
            if torch.isnan(self.emb_neg).any() or torch.isinf(self.emb_neg).any():
                raise ValueError('invalid value detected in E-,\n{0}'.format(self.emb_neg))

        # Initialize ROOT STATE
        root_rl_state = self.create_rl_state(self.start_class, is_root=True)
        self.compute_quality_of_class_expression(root_rl_state)
        return root_rl_state

    def fit(self, learning_problem: PosNegLPStandard, max_runtime=None):
        if max_runtime:
            assert isinstance(max_runtime, float)
            self.max_runtime = max_runtime

        pos_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.pos))])
        neg_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in learning_problem.neg))])
        type_bias = pos_type_counts - neg_type_counts
        # (1) Initialize learning problem
        root_state = self.initialize_class_expression_learning_problem(pos=learning_problem.pos, neg=learning_problem.neg)
        # (2) Add root state into search tree
        root_state.heuristic = root_state.quality
        self.search_tree.add(root_state)

        self.start_time = time.time()
        # (3) Inject Type Bias
        for x in (self.create_rl_state(i, parent_node=root_state) for i in type_bias):
            self.compute_quality_of_class_expression(x)
            x.heuristic = x.quality
            self.search_tree.add(x)

        # (3) Search
        for i in range(1, self.iter_bound):
            # (1) Get the most fitting RL-state
            most_promising = self.next_node_to_expand()
            next_possible_states = []
            # (2) Refine (1)
            for ref in self.apply_refinement(most_promising):
                if time.time() - self.start_time > self.max_runtime:
                    return self.terminate()
                # (2.1) If the next possible RL-state is not a dead end
                # (2.1.) If the refinement of (1) is not equivalent to \bottom

                if len(ref.instances):
                    # Compute quality
                    self.compute_quality_of_class_expression(ref)
                    if ref.quality == 0:
                        continue
                    next_possible_states.append(ref)

                    if self.stop_at_goal:
                        if ref.quality == 1.0:
                            break
            try:
                assert len(next_possible_states) > 0
            except AssertionError:
                print(f'DEAD END at {most_promising}')
                continue
            if len(next_possible_states) == 0:
                # We do not need to compute Q value based on embeddings of "zeros".
                continue

            if self.pre_trained_kge:
                preds = self.predict_values(current_state=most_promising, next_states=next_possible_states)
            else:
                preds = None
            self.goal_found = self.update_search(next_possible_states, preds)
            if self.goal_found:
                if self.terminate_on_goal:
                    return self.terminate()
            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()

    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        assert ValueError('show_search_tree')

    def terminate_training(self):
        return self

    def fit_from_iterable(self,
                          dataset: List[Tuple[object, Set[OWLNamedIndividual], Set[OWLNamedIndividual]]],
                          max_runtime: int = None) -> List:
        """
        Dataset is a list of tuples where the first item is either str or OWL class expression indicating target
        concept.
        """
        if max_runtime:
            self.max_runtime = max_runtime
        renderer = DLSyntaxObjectRenderer()

        results = []
        for (target_ce, p, n) in dataset:
            print(f'TARGET OWL CLASS EXPRESSION:\n{target_ce}')
            print(f'|Sampled Positive|:{len(p)}\t|Sampled Negative|:{len(n)}')
            start_time = time.time()
            self.fit(pos=p, neg=n, max_runtime=max_runtime)
            rn = time.time() - start_time
            h: RL_State = next(iter(self.best_hypotheses()))
            # TODO:CD: We need to remove this first returned boolean for the sake of readability.
            _, f_measure = F1().score_elp(instances=h.instances_bitset, learning_problem=self._learning_problem)
            _, accuracy = Accuracy().score_elp(instances=h.instances_bitset, learning_problem=self._learning_problem)

            report = {'Target': str(target_ce),
                      'Prediction': renderer.render(h.concept),
                      'F-measure': f_measure,
                      'Accuracy': accuracy,
                      'NumClassTested': self._number_of_tested_concepts,
                      'Runtime': rn}
            results.append(report)

        return results

    def init_training(self, pos_uri: Set[OWLNamedIndividual], neg_uri: Set[OWLNamedIndividual]) -> None:
        """
        Initialize training.
        """
        """ (1) Generate a Learning Problem """
        self._learning_problem = PosNegLPStandard(pos=pos_uri, neg=neg_uri).encode_kb(self.kb)
        """ (2) Update REWARD FUNC FOR each learning problem """
        self.reward_func.lp = self._learning_problem
        """ (3) Obtain embeddings of positive and negative examples """
        if self.pre_trained_kge is not None:
            self.emb_pos = self.pre_trained_kge.get_entity_embeddings(
                [owl_individual.get_iri().as_str() for owl_individual in pos_uri])
            self.emb_neg = self.pre_trained_kge.get_entity_embeddings(
                [owl_individual.get_iri().as_str() for owl_individual in neg_uri])
            """ (3) Take the mean of positive and negative examples and reshape it into (1,1,embedding_dim) for mini
             batching """
            self.emb_pos = torch.mean(self.emb_pos, dim=0)
            self.emb_pos = self.emb_pos.view(1, 1, self.emb_pos.shape[0])
            self.emb_neg = torch.mean(self.emb_neg, dim=0)
            self.emb_neg = self.emb_neg.view(1, 1, self.emb_neg.shape[0])
            # Sanity checking
            if torch.isnan(self.emb_pos).any() or torch.isinf(self.emb_pos).any():
                raise ValueError('invalid value detected in E+,\n{0}'.format(self.emb_pos))
            if torch.isnan(self.emb_neg).any() or torch.isinf(self.emb_neg).any():
                raise ValueError('invalid value detected in E-,\n{0}'.format(self.emb_neg))
        else:
            self.emb_pos = None
            self.emb_neg = None

        # Default exploration exploitation tradeoff.
        """ (3) Default  exploration exploitation tradeoff and number of expression tested """
        self.epsilon = 1
        self._number_of_tested_concepts = 0

    def create_rl_state(self, c: OWLClassExpression, parent_node: Optional[RL_State] = None,
                        is_root: bool = False) -> RL_State:
        """ Create an RL_State instance."""
        instances: Generator
        instances = set(self.kb.individuals(c))
        instances_bitset: FrozenSet[OWLNamedIndividual]
        instances_bitset = self.kb.individuals_set(c)

        if self.pre_trained_kge is not None:
            raise NotImplementedError("No pre-trained knowledge")

        rl_state = RL_State(c, parent_node=parent_node,
                            is_root=is_root,
                            instances=instances,
                            instances_bitset=instances_bitset, embeddings=None)
        rl_state.length = self.kb.concept_len(c)
        return rl_state

    def compute_quality_of_class_expression(self, state: RL_State) -> None:
        """ Compute Quality of owl class expression."""
        self.quality_func.apply(state, state.instances_bitset, self.learning_problem)
        self._number_of_tested_concepts += 1

    def apply_refinement(self, rl_state: RL_State) -> Generator:
        """
        Refine an OWL Class expression \\|= Observing next possible states.

        1. Generate concepts by refining a node.
        1.1. Compute allowed length of refinements.
        1.2. Convert concepts if concepts do not belong to  self.concepts_to_ignore.
             Note that          i.str not in self.concepts_to_ignore => O(1) if a set is being used.
        3. Return Generator.
        """
        assert isinstance(rl_state, RL_State)
        self.operator: LengthBasedRefinement
        # 1.
        for i in self.operator.refine(rl_state.concept):  # O(N)
            yield self.create_rl_state(i, parent_node=rl_state)

    def rl_learning_loop(self, num_episode: int,
                         pos_uri: Set[OWLNamedIndividual],
                         neg_uri: Set[OWLNamedIndividual],
                         goal_path: List[RL_State] = None) -> List[float]:
        """ Reinforcement Learning Training Loop

        Initialize RL environment for a given learning problem (E^+ pos_iri and E^- neg_iri )

        Training:
                    2.1 Obtain a trajectory: A sequence of RL states/DL concepts
                    T, Person, (Female and \forall hasSibling Female).
                    Rewards at each transition are also computed
        """

        # (1) Initialize RL environment for training
        print("Reinforcement Learning loop started...")
        assert isinstance(pos_uri, Set) and isinstance(neg_uri, Set)
        self.init_training(pos_uri=pos_uri, neg_uri=neg_uri)
        root_rl_state = self.create_rl_state(self.start_class, is_root=True)
        self.compute_quality_of_class_expression(root_rl_state)
        sum_of_rewards_per_actions = []

        # () Reinforcement Learning offline training loop
        for th in range(num_episode):
            print(f"Episode {th + 1}: ", end=" ")
            # Sequence of decisions
            start_time = time.time()
            sequence_of_states, rewards = self.sequence_of_actions(root_rl_state)
            print(f"Runtime {time.time() - start_time:.3f} secs", end=" | ")
            print(f"Max reward: {max(rewards)}", end=" | ")
            print(f"Epsilon : {self.epsilon}")
            """
            print('#' * 10, end='')
            print(f'\t{th}.th Sequence of Actions\t', end='')
            print('#' * 10)
            for step, (current_state, next_state) in enumerate(sequence_of_states):
                print(f'{step}. Transition \n{current_state}\n----->\n{next_state}')
                print(f'Reward:{rewards[step]}')

            print('{0}.th iter. SumOfRewards: {1:.2f}\t'
                  'Epsilon:{2:.2f}\t'
                  '|ReplayMem.|:{3}'.format(th, sum(rewards),
                                            self.epsilon,
                                            len(self.experiences)))
            """
            # Form experiences
            self.form_experiences(sequence_of_states, rewards)
            sum_of_rewards_per_actions.append(sum(rewards))
            """(3.2) Learn from experiences"""
            # if th % self.num_episodes_per_replay == 0:
            self.learn_from_replay_memory()
            """(3.4) Exploration Exploitation"""
            if self.epsilon < 0:
                break
            self.epsilon -= self.epsilon_decay

        return sum_of_rewards_per_actions

    def select_next_state(self, current_state, next_rl_states) -> Tuple[RL_State, float]:
        if True:
            next_selected_rl_state = self.exploration_exploitation_tradeoff(current_state, next_rl_states)
            return next_selected_rl_state, self.reward_func.apply(current_state, next_selected_rl_state)
        else:
            for i in next_rl_states:
                print(i)
            exit(1)

    def sequence_of_actions(self, root_rl_state: RL_State) -> Tuple[List[Tuple[AbstractNode, AbstractNode]],
    List[SupportsFloat]]:
        assert isinstance(root_rl_state, RL_State)

        current_state = root_rl_state
        path_of_concepts = []
        rewards = []

        assert len(current_state.embeddings) > 0  # Embeddings are initialized
        assert current_state.quality > 0
        assert current_state.heuristic is None

        # (1)
        for _ in range(self.num_of_sequential_actions):
            assert isinstance(current_state, RL_State)
            # (1.1) Observe Next RL states, i.e., refine an OWL class expression
            next_rl_states = list(self.apply_refinement(current_state))
            # (1.2)
            if len(next_rl_states) == 0:  # DEAD END
                # assert (current_state.length + 3) <= self.max_child_length
                print('No next state')
                break
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
        # print('learn_from_replay_memory', end="\t|\t")
        current_state_batch: List[torch.FloatTensor]
        next_state_batch: List[torch.FloatTensor]
        current_state_batch, next_state_batch, y = self.experiences.retrieve()
        # N, 1, dim
        current_state_batch = torch.cat(current_state_batch, dim=0)
        # N, 1, dim
        next_state_batch = torch.cat(next_state_batch, dim=0)
        y = torch.Tensor(y)

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

        # batch, 4, dim
        X = torch.cat([current_state_batch, next_state_batch, self.emb_pos.repeat((num_next_states, 1, 1)),
                       self.emb_neg.repeat((num_next_states, 1, 1))], 1)
        """
        # We can skip this part perhaps
        dataset = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                         next_state_batch=next_state_batch,
                                         p=self.emb_pos, n=self.emb_neg, q=q_values)
        num_experience = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)
        """
        # print(f'Experiences:{X.shape}', end="\t|\t")
        self.heuristic_func.net.train()
        total_loss = 0
        for m in range(self.num_epochs_per_replay):
            self.optimizer.zero_grad()  # zero the gradient buffers
            # forward: n by 4, dim
            predicted_q = self.heuristic_func.net.forward(X)
            # loss
            loss = self.heuristic_func.net.loss(predicted_q, y)
            total_loss += loss.item()
            # compute the derivative of the loss w.r.t. the parameters using backpropagation
            loss.backward()
            # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        # print(f'Average loss during training: {total_loss / self.num_epochs_per_replay:0.5f}')
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
                child_node.heuristic = child_node.quality
                if child_node.quality > 0:  # > too weak, ignore.
                    self.search_tree.add(child_node)
                if child_node.quality == 1:
                    return child_node

    def assign_embeddings(self, rl_state: RL_State) -> None:
        """
        Assign embeddings to a rl state. A rl state is represented with vector representation of
        all individuals belonging to a respective OWLClassExpression.
        """
        assert isinstance(rl_state, RL_State)
        # (1) Detect mode of representing OWLClassExpression
        if self.representation_mode == 'averaging':
            # (2) if input node has not seen before, assign embeddings.
            if rl_state.embeddings is None:
                assert isinstance(rl_state.concept, OWLClassExpression)
                # (3) Retrieval instances via our retrieval function (R(C)). Be aware Open World and Closed World

                rl_state.instances = set(self.kb.individuals(rl_state.concept))
                # (4) Retrieval instances in terms of bitset.
                rl_state.instances_bitset = self.kb.individuals_set(rl_state.concept)
                # (5) |R(C)|=\emptyset ?
                if len(rl_state.instances) == 0:
                    # If|R(C)|=\emptyset, then represent C with zeros
                    if self.pre_trained_kge is not None:
                        emb = torch.zeros(1, self.sample_size, self.embedding_dim)
                    else:
                        emb = torch.rand(size=(1, self.sample_size, self.embedding_dim))
                else:
                    # If|R(C)| \not= \emptyset, then take the mean of individuals.
                    str_individuals = [i.get_iri().as_str() for i in rl_state.instances]
                    assert len(str_individuals) > 0
                    if self.pre_trained_kge is not None:
                        emb = self.pre_trained_kge.get_entity_embeddings(str_individuals)
                        emb = torch.mean(emb, dim=0)
                        emb = emb.view(1, self.sample_size, self.embedding_dim)
                    else:
                        emb = torch.rand(size=(1, self.sample_size, self.embedding_dim))
                # (6) Assign embeddings
                rl_state.embeddings = emb
            else:
                """ Embeddings already assigned."""
                try:
                    assert rl_state.embeddings.shape == (1, self.sample_size, self.embedding_dim)
                except AssertionError as e:
                    print(e)
                    print(rl_state)
                    print(rl_state.embeddings.shape)
                    print((1, self.sample_size, self.instance_embeddings.shape[1]))
                    raise
        else:
            """ No embeddings available assigned."""""
            assert self.representation_mode is None

    def get_embeddings(self, instances) -> None:
        if self.representation_mode == 'averaging':
            # (2) if input node has not seen before, assign embeddings.
            if rl_state.embeddings is None:
                assert isinstance(rl_state.concept, OWLClassExpression)
                # (3) Retrieval instances via our retrieval function (R(C)). Be aware Open World and Closed World

                rl_state.instances = set(self.kb.individuals(rl_state.concept))
                # (4) Retrieval instances in terms of bitset.
                rl_state.instances_bitset = self.kb.individuals_set(rl_state.concept)
                # (5) |R(C)|=\emptyset ?
                if len(rl_state.instances) == 0:
                    # If|R(C)|=\emptyset, then represent C with zeros
                    if self.pre_trained_kge is not None:
                        emb = torch.zeros(1, self.sample_size, self.embedding_dim)
                    else:
                        emb = torch.rand(size=(1, self.sample_size, self.embedding_dim))
                else:
                    # If|R(C)| \not= \emptyset, then take the mean of individuals.
                    str_individuals = [i.get_iri().as_str() for i in rl_state.instances]
                    assert len(str_individuals) > 0
                    if self.pre_trained_kge is not None:
                        emb = self.pre_trained_kge.get_entity_embeddings(str_individuals)
                        emb = torch.mean(emb, dim=0)
                        emb = emb.view(1, self.sample_size, self.embedding_dim)
                    else:
                        emb = torch.rand(size=(1, self.sample_size, self.embedding_dim))
                # (6) Assign embeddings
                rl_state.embeddings = emb
            else:
                """ Embeddings already assigned."""
                try:
                    assert rl_state.embeddings.shape == (1, self.sample_size, self.embedding_dim)
                except AssertionError as e:
                    print(e)
                    print(rl_state)
                    print(rl_state.embeddings.shape)
                    print((1, self.sample_size, self.instance_embeddings.shape[1]))
                    raise
        else:
            """ No embeddings available assigned."""""
            assert self.representation_mode is None

    def save_weights(self):
        """
        Save pytorch weights.
        """
        # Save model.
        torch.save(self.heuristic_func.net.state_dict(),
                   self.storage_path + '/{0}.pth'.format(self.heuristic_func.name))

    def exploration_exploitation_tradeoff(self, current_state: AbstractNode,
                                          next_states: List[AbstractNode]) -> AbstractNode:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        (1) Exploration.
        (2) Exploitation.
        """
        if random.random() < self.epsilon:
            next_state = random.choice(next_states)
            self.assign_embeddings(next_state)
        else:
            next_state = self.exploitation(current_state, next_states)
        self.compute_quality_of_class_expression(next_state)
        return next_state

    def exploitation(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> AbstractNode:
        """
        Find next node that is assigned with highest predicted Q value.

        (1) Predict Q values : predictions.shape => torch.Size([n, 1]) where n = len(next_states).

        (2) Find the index of max value in predictions.

        (3) Use the index to obtain next state.

        (4) Return next state.
        """
        predictions: torch.Tensor = self.predict_values(current_state, next_states)
        argmax_id = int(torch.argmax(predictions))
        next_state = next_states[argmax_id]
        """
        # Sanity checking
        print('#'*10)
        for s, q in zip(next_states, predictions):
            print(s, q)
        print('#'*10)
        print(next_state,f'\t {torch.max(predictions)}')
        """
        return next_state

    def predict_values(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> torch.Tensor:
        """
        Predict promise of next states given current state.

        Returns:
            Predicted Q values.
        """
        # Instead it should be get embeddings ?
        self.assign_embeddings(current_state)
        assert len(next_states) > 0
        with torch.no_grad():
            self.heuristic_func.net.eval()
            # create batch batch.
            next_state_batch = []
            for _ in next_states:
                self.assign_embeddings(_)
                next_state_batch.append(_.embeddings)
            next_state_batch = torch.cat(next_state_batch, dim=0)
            x = PrepareBatchOfPrediction(current_state.embeddings,
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

    def generate_learning_problems(self, dataset: Optional[Iterable[Tuple[str, Set, Set]]] = None,
                                   num_of_target_concepts: int = 3,
                                   num_learning_problems: int = 5) -> Iterable[
        Tuple[str, Set, Set]]:
        """ Generate learning problems if none is provided.

            Time complexity: O(n^2) n = named concepts
        """

        if dataset is None:
            counter = 0
            size_of_examples = 3
            print("Generating learning problems on the fly...")
            for i in self.kb.get_concepts():
                individuals_i = set(self.kb.individuals(i))

                if len(individuals_i) > size_of_examples:
                    str_dl_concept_i = self.renderer.render(i)
                    for j in self.kb.get_concepts():
                        if i == j:
                            continue
                        individuals_j = set(self.kb.individuals(j))
                        if len(individuals_j) < size_of_examples:
                            continue
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
                else:
                    """Empy concept"""
        else:
            return dataset

    def train(self, dataset: Optional[Iterable[Tuple[str, Set, Set]]] = None, num_of_target_concepts: int = 3,
              num_episode: int = 3, num_learning_problems: int = 3):
        """ Train an RL agent on description logic concept learning problems """

        if self.pre_trained_kge is None:
            return self.terminate_training()

        counter = 1
        for (target_owl_ce, positives, negatives) in self.generate_learning_problems(dataset,
                                                                                     num_of_target_concepts,
                                                                                     num_learning_problems):
            print(f"Goal Concept:\t {target_owl_ce}\tE^+:[{len(positives)}]\t E^-:[{len(negatives)}]")
            sum_of_rewards_per_actions = self.rl_learning_loop(num_episode=num_episode, pos_uri=positives,
                                                               neg_uri=negatives)
            # print(f'Sum of Rewards in last 3 trajectories:{sum_of_rewards_per_actions[:3]}')

            self.seen_examples.setdefault(counter, dict()).update(
                {'Concept': target_owl_ce,
                 'Positives': [i.get_iri().as_str() for i in positives],
                 'Negatives': [i.get_iri().as_str() for i in negatives]})
            counter += 1
            if counter % 100 == 0:
                self.save_weights()
        return self.terminate_training()

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
            current_state.length = self.kb.concept_len(current_state.concept)
            if current_state.quality is None:
                self.compute_quality_of_class_expression(current_state)

            next_state = sequence_of_goal_path.pop(0)
            self.assign_embeddings(next_state)
            next_state.length = self.kb.concept_len(next_state.concept)
            if next_state.quality is None:
                self.compute_quality_of_class_expression(next_state)
            sequence_of_states.append((current_state, next_state))
            rewards.append(self.reward_func.apply(current_state, next_state))
        for x in range(2):
            self.form_experiences(sequence_of_states, rewards)
        self.learn_from_replay_memory()

    def best_hypotheses(self, n=1):
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        if n == 1:
            return [i for i in self.search_tree.get_top_n_nodes(n)][0]
        else:
            return [i for i in self.search_tree.get_top_n_nodes(n)]

    def clean(self):
        self.emb_pos, self.emb_neg = None, None
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

    def downward_refinement(self, *args, **kwargs):
        ValueError('downward_refinement')

    def next_node_to_expand(self) -> RL_State:
        """ Return a node that maximizes the heuristic function at time t. """
        return self.search_tree.get_most_promising()


class DrillHeuristic:
    """
    Heuristic in Convolutional DQL concept learning.
    Heuristic implements a convolutional neural network.
    """

    def __init__(self, pos=None, neg=None, model=None, mode=None, model_args=None):
        if model:
            self.net = model
        elif mode in ['averaging', 'sampling']:
            self.net = DrillNet(model_args)
            self.mode = mode
            self.name = 'DrillHeuristic_' + self.mode
        else:
            raise ValueError
        self.net.eval()

    def score(self, node, parent_node=None):
        """ Compute heuristic value of root node only"""
        if parent_node is None and node.is_root:
            return torch.FloatTensor([.0001]).squeeze()
        raise ValueError

    def apply(self, node, parent_node=None):
        """ Assign predicted Q-value to node object."""
        predicted_q_val = self.score(node, parent_node)
        node.heuristic = predicted_q_val


class DrillNet(torch.nn.Module):
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

    def __init__(self, args):
        super(DrillNet, self).__init__()
        self.in_channels, self.embedding_dim = args['input_shape']
        assert self.embedding_dim

        self.loss = torch.nn.MSELoss()
        # Conv1D seems to be faster than Conv2d
        self.conv1 = torch.nn.Conv1d(in_channels=4,
                                     out_channels=args['first_out_channels'],
                                     kernel_size=args['kernel_size'],
                                     padding=1, stride=1, bias=True)

        # Fully connected layers.
        self.size_of_fc1 = int(args['first_out_channels'] * self.embedding_dim)
        self.fc1 = torch.nn.Linear(in_features=self.size_of_fc1, out_features=self.size_of_fc1 // 2)
        self.fc2 = torch.nn.Linear(in_features=self.size_of_fc1 // 2, out_features=1)

        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.fc1.weight.data)
        torch.nn.init.xavier_normal_(self.conv1.weight.data)

    def forward(self, X: torch.FloatTensor):
        """
        X  n by 4 by d float tensor
        """
        # N x 32 x D
        X = torch.nn.functional.relu(self.conv1(X))
        X = X.flatten(start_dim=1)
        # N x (32D/2)
        X = torch.nn.functional.relu(self.fc1(X))
        # N x 1
        scores = self.fc2(X).flatten()
        return scores
