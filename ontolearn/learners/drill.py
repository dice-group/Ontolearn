from ontolearn.base_concept_learner import RefinementBasedConceptLearner
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.abstracts import AbstractScorer, AbstractNode
from ontolearn.search import RL_State
from typing import Set, List, Tuple, Optional, Generator, SupportsFloat, Iterable
from owlapy.model import OWLNamedIndividual, OWLClassExpression
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
import torch
from ontolearn.data_struct import Experience
from ontolearn.search import DRILLSearchTreePriorityQueue
from ontolearn.utils import create_experiment_folder
from collections import Counter
from itertools import chain
import time

class Drill(RefinementBasedConceptLearner):
    """Deep Reinforcement Learning for Refinement Operators in ALC."""

    def __init__(self, knowledge_base,
                 path_of_embeddings: str = None, refinement_operator: LengthBasedRefinement = None,
                 use_inverse=True,
                 use_data_properties=True,
                 use_card_restrictions=True,
                 card_limit=10,
                 quality_func: AbstractScorer = None,
                 reward_func=None,
                 batch_size=None, num_workers=None, pretrained_model_name=None,
                 iter_bound=None, max_num_of_concepts_tested=None, verbose=None, terminate_on_goal=None,
                 max_len_replay_memory=None, epsilon_decay=None, epsilon_min=None, num_epochs_per_replay=None,
                 num_episodes_per_replay=None, learning_rate=None, max_runtime=None, num_of_sequential_actions=None,
                 num_episode=None):

        print("***DRILL has not yet been fully integrated***")
        self.name = "DRILL"
        # TODO: Clear difference between training and testing should be defined at init
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(knowledge_base=knowledge_base,
                                                        use_data_properties=use_data_properties,
                                                        use_card_restrictions=use_card_restrictions,
                                                        card_limit=card_limit,
                                                        use_inverse=use_inverse)
        self.reward_func = reward_func
        self.representation_mode = "averaging"
        self.heuristic_func = None
        self.num_workers = num_workers
        self.epsilon = 1
        self.learning_rate = .001
        self.num_episode = 1
        self.num_of_sequential_actions = 3
        self.num_epochs_per_replay = 1
        self.max_len_replay_memory = 256
        self.epsilon_decay = 0.01
        self.epsilon_min = 0
        self.batch_size = 1024
        self.verbose = 0
        self.num_episodes_per_replay = 2
        self.seen_examples = dict()
        self.emb_pos, self.emb_neg = None, None
        self.start_time = None
        self.goal_found = False
        self.experiences = Experience(maxlen=self.max_len_replay_memory)

        if path_of_embeddings is not None and os.path.exists(path_of_embeddings):
            self.instance_embeddings = pd.read_csv(path_of_embeddings)
            self.embedding_dim = self.instance_embeddings.shape[1]
        else:
            self.instance_embeddings = None
            self.embedding_dim = 12

        self.sample_size = 1
        arg_net = {'input_shape': (4 * self.sample_size, self.embedding_dim),
                   'first_out_channels': 32, 'second_out_channels': 16, 'third_out_channels': 8,
                   'kernel_size': 3}
        self.heuristic_func = DrillHeuristic(mode='averaging', model_args=arg_net)
        if self.learning_rate:
            self.optimizer = torch.optim.Adam(self.heuristic_func.net.parameters(), lr=self.learning_rate)

        if pretrained_model_name:
            self.pre_trained_model_loaded = True
            self.heuristic_func.net.load_state_dict(torch.load(pretrained_model_name, torch.device('cpu')))
        else:
            self.pre_trained_model_loaded = False

        RefinementBasedConceptLearner.__init__(self, knowledge_base=knowledge_base,
                                               refinement_operator=refinement_operator,
                                               quality_func=quality_func,
                                               heuristic_func=self.heuristic_func,
                                               terminate_on_goal=terminate_on_goal,
                                               iter_bound=iter_bound,
                                               max_num_of_concepts_tested=max_num_of_concepts_tested,
                                               max_runtime=max_runtime)
        print('Number of parameters: ', sum([p.numel() for p in self.heuristic_func.net.parameters()]))

        self.search_tree = DRILLSearchTreePriorityQueue()
        self._learning_problem = None
        self.storage_path, _ = create_experiment_folder()

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
        self._learning_problem = PosNegLPStandard(pos=set(pos), neg=set(neg)).encode_kb(self.kb)
        # 2. Obtain embeddings of positive and negative examples.
        if self.instance_embeddings is None:
            self.emb_pos = None
            self.emb_neg = None
        else:
            self.emb_pos = torch.tensor(
                self.instance_embeddings.loc[[owl_indv.get_iri().as_str() for owl_indv in pos]].values,
                dtype=torch.float32)
            self.emb_neg = torch.tensor(
                self.instance_embeddings.loc[[owl_indv.get_iri().as_str() for owl_indv in neg]].values,
                dtype=torch.float32)

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

    def fit(self, lp: PosNegLPStandard, max_runtime=None):
        if max_runtime:
            assert isinstance(max_runtime, int)
            self.max_runtime = max_runtime
        # @TODO: Type injection should be possible for all
        pos_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in lp.pos))])
        neg_type_counts = Counter(
            [i for i in chain.from_iterable((self.kb.get_types(ind, direct=True) for ind in lp.neg))])
        type_bias = pos_type_counts - neg_type_counts

        # (1) Initialize learning problem
        root_state = self.initialize_class_expression_learning_problem(pos=lp.pos, neg=lp.neg)
        # (2) Add root state into search tree
        root_state.heuristic = root_state.quality
        self.search_tree.add(root_state)
        # (3) Inject Type Bias
        for x in (self.create_rl_state(i, parent_node=root_state) for i in type_bias):
            self.compute_quality_of_class_expression(x)
            x.heuristic = x.quality
            self.search_tree.add(x)
        self.start_time = time.time()
        # (3) Search
        for i in range(1, self.iter_bound):
            # (1) Get the most fitting RL-state
            most_promising = self.next_node_to_expand()
            next_possible_states = []
            # (2) Refine (1)
            for ref in self.apply_refinement(most_promising):
                # (2.1) If the next possible RL-state is not a dead end
                # (2.1.) If the refinement of (1) is not equivalent to \bottom
                if len(ref.instances):
                    # Compute quality
                    self.compute_quality_of_class_expression(ref)
                    if ref.quality == 0:
                        continue
                    next_possible_states.append(ref)
                    if ref.quality == 1.0:
                        break
            try:
                assert len(next_possible_states) > 0
            except AssertionError:
                if self.verbose > 1:
                    logger.info(f'DEAD END at {most_promising}')
                continue
            if len(next_possible_states) == 0:
                # We do not need to compute Q value based on embeddings of "zeros".
                continue

            if self.pre_trained_model_loaded is True:
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
        ValueError('show_search_tree')

    def terminate_training(self):
        ValueError('terminate_training')

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
            if self.verbose > 0:
                logger.info(f'TARGET OWL CLASS EXPRESSION:\n{target_ce}')
                logger.info(f'|Sampled Positive|:{len(p)}\t|Sampled Negative|:{len(n)}')
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
        self.emb_pos = torch.tensor(
            self.instance_embeddings.loc[[owl_indv.get_iri().as_str() for owl_indv in pos_uri]].values,
            dtype=torch.float32)
        self.emb_neg = torch.tensor(
            self.instance_embeddings.loc[[owl_indv.get_iri().as_str() for owl_indv in neg_uri]].values,
            dtype=torch.float32)
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

        # Default exploration exploitation tradeoff.
        """ (3) Default  exploration exploitation tradeoff and number of expression tested """
        self.epsilon = 1
        self._number_of_tested_concepts = 0

    def create_rl_state(self, c: OWLClassExpression, parent_node: Optional[RL_State] = None,
                        is_root: bool = False) -> RL_State:
        """ Create an RL_State instance."""
        # Create State
        rl_state = RL_State(c, parent_node=parent_node, is_root=is_root)
        # Assign Embeddings to it. Later, assign_embeddings can be also done in RL_STATE
        self.assign_embeddings(rl_state)
        rl_state.length = self.kb.concept_len(c)
        return rl_state

    def compute_quality_of_class_expression(self, state: RL_State) -> None:
        """ Compute Quality of owl class expression."""
        self.quality_func.apply(state, state.instances_bitset, self._learning_problem)
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
        # 1.
        for i in self.operator.refine(rl_state.concept):  # O(N)
            yield self.create_rl_state(i, parent_node=rl_state)

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

    def rl_learning_loop(self, pos_uri: Set[OWLNamedIndividual], neg_uri: Set[OWLNamedIndividual],
                         goal_path: List[RL_State] = None) -> List[float]:
        """
        Standard RL training loop.

        1. Initialize RL environment for training.

        2. Learn from an illustration if possible.
        2. Training Loop.
        """
        """ (1) Initialize RL environment for training """
        self.init_training(pos_uri=pos_uri, neg_uri=neg_uri)
        root_rl_state = self.create_rl_state(self.start_class, is_root=True)
        self.compute_quality_of_class_expression(root_rl_state)
        sum_of_rewards_per_actions = []
        log_every_n_episodes = int(self.num_episode * .1) + 1
        """ (2) Learn from an illustration if possible """
        if goal_path:
            self.learn_from_illustration(goal_path)

        """ (3) Reinforcement Learning offline training loop  """
        for th in range(self.num_episode):
            """ (3.1) Sequence of decisions """
            sequence_of_states, rewards = self.sequence_of_actions(root_rl_state)

            if self.verbose >= 10:
                logger.info('#' * 10, end='')
                logger.info(f'{th}\t.th Sequence of Actions', end='')
                logger.info('#' * 10)
                for step, (current_state, next_state) in enumerate(sequence_of_states):
                    logger.info(f'{step}. Transition \n{current_state}\n----->\n{next_state}')
                    logger.info(f'Reward:{rewards[step]}')

            if th % log_every_n_episodes == 0:
                if self.verbose >= 1:
                    logger.info('{0}.th iter. SumOfRewards: {1:.2f}\t'
                                'Epsilon:{2:.2f}\t'
                                '|ReplayMem.|:{3}'.format(th, sum(rewards),
                                                          self.epsilon,
                                                          len(self.experiences)))
            """(3.2) Form experiences"""
            self.form_experiences(sequence_of_states, rewards)
            sum_of_rewards_per_actions.append(sum(rewards))
            """(3.2) Learn from experiences"""
            if th % self.num_episodes_per_replay == 0:
                self.learn_from_replay_memory()
            """(3.4) Exploration Exploitation"""
            if self.epsilon < 0:
                break
            self.epsilon -= self.epsilon_decay

        return sum_of_rewards_per_actions

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
            # (1.3)
            next_selected_rl_state = self.exploration_exploitation_tradeoff(current_state, next_rl_states)
            # (1.4) Remember the concept path
            path_of_concepts.append((current_state, next_selected_rl_state))
            # (1.5)
            rewards.append(self.reward_func.apply(current_state, next_selected_rl_state))
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

        if self.verbose > 1:
            print('Form Experiences for the training')

        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    def learn_from_replay_memory(self) -> None:
        """
        Learning by replaying memory.
        """
        if self.verbose > 1:
            print('Learn from Experience')

        current_state_batch, next_state_batch, q_values = self.experiences.retrieve()
        current_state_batch = torch.cat(current_state_batch, dim=0)
        next_state_batch = torch.cat(next_state_batch, dim=0)
        q_values = torch.Tensor(q_values)

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
        dataset = PrepareBatchOfTraining(current_state_batch=current_state_batch,
                                         next_state_batch=next_state_batch,
                                         p=self.emb_pos, n=self.emb_neg, q=q_values)
        num_experience = len(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)
        if self.verbose > 1:
            print(f'Number of experiences:{num_experience}')
            print('DQL agent is learning via experience replay')
        self.heuristic_func.net.train()
        for m in range(self.num_epochs_per_replay):
            total_loss = 0
            for X, y in data_loader:
                self.optimizer.zero_grad()  # zero the gradient buffers
                # forward
                predicted_q = self.heuristic_func.net.forward(X)
                # loss
                loss = self.heuristic_func.net.loss(predicted_q, y)
                total_loss += loss.item()
                # compute the derivative of the loss w.r.t. the parameters using backpropagation
                loss.backward()
                # clip gradients if gradients are killed. =>torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
            if self.verbose > 1:
                print(f'{m}.th Epoch average loss during training:{total_loss / num_experience}')

        self.heuristic_func.net.train().eval()

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
                # Assumption
                rl_state.instances = set(self.kb.individuals(rl_state.concept))
                # (4) Retrieval instances in terms of bitset.
                rl_state.instances_bitset = self.kb.individuals_set(rl_state.concept)
                # (5) |R(C)|=\emptyset ?
                if len(rl_state.instances) == 0:
                    # If|R(C)|=\emptyset, then represent C with zeros
                    if self.instance_embeddings is not None:
                        emb = torch.zeros(1, self.sample_size, self.instance_embeddings.shape[1])
                    else:
                        emb = torch.rand(size=(1, self.sample_size, self.embedding_dim))
                else:
                    # If|R(C)| \not= \emptyset, then take the mean of individuals.
                    str_idx = [i.get_iri().as_str() for i in rl_state.instances]
                    assert len(str_idx) > 0
                    if self.instance_embeddings is not None:
                        emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
                        emb = torch.mean(emb, dim=0)
                        emb = emb.view(1, self.sample_size, self.instance_embeddings.shape[1])
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
        elif self.representation_mode == 'sampling':
            raise NotImplementedError('Sampling technique for state representation is not implemented.')
            """
                        if node.embeddings is None:
                str_idx = [get_full_iri(i).replace('\n', '') for i in node.concept.instances]
                if len(str_idx) >= self.sample_size:
                    sampled_str_idx = random.sample(str_idx, self.sample_size)
                    emb = torch.tensor(self.instance_embeddings.loc[sampled_str_idx].values, dtype=torch.float32)
                else:
                    num_rows_to_fill = self.sample_size - len(str_idx)
                    emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
                    emb = torch.cat((torch.zeros(num_rows_to_fill, self.instance_embeddings.shape[1]), emb))
                emb = emb.view(1, self.sample_size, self.instance_embeddings.shape[1])
                node.embeddings = emb
            else:
                try:
                    assert node.embeddings.shape == (1, self.sample_size, self.instance_embeddings.shape[1])
                except AssertionError:
                    print(node)
                    print(self.sample_size)
                    print(node.embeddings.shape)
                    print((1, self.sample_size, self.instance_embeddings.shape[1]))
                    raise ValueError
            """
        else:
            raise ValueError

        # @todo remove this testing in experiments.
        if torch.isnan(rl_state.embeddings).any() or torch.isinf(rl_state.embeddings).any():
            # No individual contained in the input concept.
            # Sanity checking.
            raise ValueError

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
        if np.random.random() < self.epsilon:
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

    def train(self, dataset: Iterable[Tuple[str, Set, Set]], relearn_ratio: int = 2):
        """
        Train RL agent on learning problems with relearn_ratio.

        Args:
            dataset: An iterable containing training data. Each item corresponds to a tuple of string representation
                of target concept, a set of positive examples in the form of URIs amd a set of negative examples in
                the form of URIs, respectively.
            relearn_ratio: An integer indicating the number of times dataset is iterated.

        Computation:
        1. Dataset and relearn_ratio loops: Learn each problem relearn_ratio times.

        2. Learning loop.

        3. Take post process action that implemented by subclass.

        Returns:
            self.
        """
        if self.verbose > 0:
            logger.info(f'Training starts.\nNumber of learning problem:{len(dataset)},\t Relearn ratio:{relearn_ratio}')
        counter = 1
        renderer = DLSyntaxObjectRenderer()

        # 1.
        for _ in range(relearn_ratio):
            for (target_owl_ce, positives, negatives) in dataset:

                if self.verbose > 0:
                    logger.info(
                        'Goal Concept:{0}\tE^+:[{1}] \t E^-:[{2}]'.format(target_owl_ce,
                                                                          len(positives), len(negatives)))
                    logger.info(f'RL training on {counter}.th learning problem starts')

                goal_path = list(reversed(self.retrieve_concept_chain(target_owl_ce)))
                # goal_path: [⊤, Daughter, Daughter ⊓ Mother]
                sum_of_rewards_per_actions = self.rl_learning_loop(pos_uri=positives, neg_uri=negatives,
                                                                   goal_path=goal_path)

                if self.verbose > 2:
                    logger.info(f'Sum of Rewards in first 3 trajectory:{sum_of_rewards_per_actions[:3]}')
                    logger.info(f'Sum of Rewards in last 3 trajectory:{sum_of_rewards_per_actions[:3]}')

                self.seen_examples.setdefault(counter, dict()).update(
                    {'Concept': renderer.render(target_owl_ce.concept),
                     'Positives': [i.get_iri().as_str() for i in positives],
                     'Negatives': [i.get_iri().as_str() for i in negatives]})

                counter += 1
                if counter % 100 == 0:
                    self.save_weights()
                # 3.
        return self.terminate_training()


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
        X = F.relu(self.conv1(X))
        X = X.flatten(start_dim=1)
        # N x (32D/2)
        X = F.relu(self.fc1(X))
        # N x 1
        scores = self.fc2(X).flatten()
        return scores
