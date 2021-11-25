import logging
import random
import time
from collections import deque
from contextlib import contextmanager
from itertools import islice
from typing import Dict, Set, List, Tuple, Iterable, Optional, Generator, SupportsFloat

import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.nn.init import xavier_normal_

from ontolearn import KnowledgeBase
from ontolearn.abstracts import AbstractDrill, AbstractScorer, AbstractNode, BaseRefinement, AbstractHeuristic
from ontolearn.base_concept_learner import BaseConceptLearner
from ontolearn.core.owl.utils import EvaluatedDescriptionSet, ConceptOperandSorter, OperandSetTransform
from ontolearn.data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction
from ontolearn.heuristics import OCELHeuristic
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.metrics import Accuracy, F1
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.search import DRILLSearchTreePriorityQueue, HeuristicOrderedNode, OENode, TreeNode, LengthOrderedNode, \
    QualityOrderedNode, RL_State
from ontolearn.utils import oplogging, create_experiment_folder
from owlapy.model import OWLClassExpression, OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.util import OrderedOWLObject
from sortedcontainers import SortedSet

# pd.set_option('display.max_columns', 100)

logger = logging.getLogger(__name__)

_concept_operand_sorter = ConceptOperandSorter()


class CELOE(BaseConceptLearner[OENode]):
    __slots__ = 'best_descriptions', 'max_he', 'min_he', 'best_only', 'calculate_min_max', 'heuristic_queue', \
                'search_tree', '_learning_problem', '_max_runtime', '_seen_norm_concepts'

    name = 'celoe_python'

    kb: KnowledgeBase

    max_he: int
    min_he: int
    best_only: bool
    calculate_min_max: bool

    search_tree: Dict[OWLClassExpression, TreeNode[OENode]]
    seen_norm_concepts: Set[OWLClassExpression]
    heuristic_queue: 'SortedSet[OENode]'
    best_descriptions: EvaluatedDescriptionSet[OENode, QualityOrderedNode]
    _learning_problem: Optional[EncodedPosNegLPStandard]

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 refinement_operator: Optional[BaseRefinement[OENode]] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 terminate_on_goal: Optional[bool] = None,
                 iter_bound: Optional[int] = None,
                 max_num_of_concepts_tested: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 max_results: int = 10,
                 best_only: bool = False,
                 calculate_min_max: bool = True):
        super().__init__(knowledge_base=knowledge_base,
                         refinement_operator=refinement_operator,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound,
                         max_num_of_concepts_tested=max_num_of_concepts_tested,
                         max_runtime=max_runtime)

        self.search_tree = dict()
        self.heuristic_queue = SortedSet(key=HeuristicOrderedNode)
        self._seen_norm_concepts = set()
        self.best_descriptions = EvaluatedDescriptionSet(max_size=max_results, ordering=QualityOrderedNode)

        self.best_only = best_only
        self.calculate_min_max = calculate_min_max

        self.max_he = 0
        self.min_he = 1
        # TODO: CD: This could be defined in BaseConceptLearner as it is used in all classes that inherits from
        # TODO: CD: BaseConceptLearner
        self._learning_problem = None
        self._max_runtime = None

    def next_node_to_expand(self, step: int) -> OENode:
        if not self.best_only:
            for node in reversed(self.heuristic_queue):
                if node.quality < 1.0:
                    return node
            else:
                raise ValueError("No Node with lesser accuracy found")
        else:
            # from reimplementation, pick without quality criterion
            return self.heuristic_queue[-1]

        # Original reimplementation of CELOE: Sort search tree at each step. Quite inefficient.
        # self.search_tree.sort_search_tree_by_decreasing_order(key='heuristic')
        # if self.verbose > 1:
        #     self.search_tree.show_search_tree(step)
        # for n in self.search_tree:
        #     return n
        # raise ValueError('Search Tree can not be empty.')

    def best_hypotheses(self, n=10) -> Iterable[OENode]:
        yield from islice(self.best_descriptions, n)

    def make_node(self, c: OWLClassExpression, parent_node: Optional[OENode] = None, is_root: bool = False) -> OENode:
        r = OENode(c, self.kb.concept_len(c), parent_node=parent_node, is_root=is_root)
        return r

    @contextmanager
    def updating_node(self, node: OENode):
        self.heuristic_queue.discard(node)
        yield node
        self.heuristic_queue.add(node)

    def downward_refinement(self, node: OENode) -> Iterable[OENode]:
        assert isinstance(node, OENode)

        with self.updating_node(node):
            # TODO: NNF
            refinements = SortedSet(
                map(_concept_operand_sorter.sort,
                    self.operator.refine(
                        node.concept,
                        max_length=node.h_exp,
                        current_domain=self.start_class)
                    )  # noqa: E203
                ,
                key=OrderedOWLObject)

            node.increment_h_exp()
            node.refinement_count = len(refinements)
            self.heuristic_func.apply(node, None, self._learning_problem)

        def make_node_with_parent(c: OWLClassExpression):
            return self.make_node(c, parent_node=node)

        return map(make_node_with_parent, refinements)

    def fit(self, *args, **kwargs):
        """
        Find hypotheses that explain pos and neg.
        """
        self.clean()
        max_runtime = kwargs.pop("max_runtime", None)
        learning_problem = self.construct_learning_problem(PosNegLPStandard, args, kwargs)

        assert not self.search_tree
        self._learning_problem = learning_problem.encode_kb(self.kb)

        if max_runtime is not None:
            self._max_runtime = max_runtime
        else:
            self._max_runtime = self.max_runtime

        root = self.make_node(_concept_operand_sorter.sort(self.start_class), is_root=True)
        self._add_node(root, None)
        assert len(self.heuristic_queue) == 1
        # TODO:CD:suggest to add another assert,e.g. assert #. of instance in root > 1

        self.start_time = time.time()
        for j in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(j)
            tree_parent = self.tree_node(most_promising)
            minimum_length = most_promising.h_exp
            if logger.isEnabledFor(oplogging.TRACE):
                logger.debug("now refining %s", most_promising)
            for ref in self.downward_refinement(most_promising):
                # we ignore all refinements with lower length
                # (this also avoids duplicate node children)
                # TODO: ignore too high depth
                if ref.len < minimum_length:
                    # ignoring refinement, it does not satisfy minimum_length condition
                    continue

                # note: tree_parent has to be equal to node_tree_parent(ref.parent_node)!
                added = self._add_node(ref, tree_parent)

                goal_found = added and ref.quality == 1.0

                if goal_found and self.terminate_on_goal:
                    return self.terminate()

            if self.calculate_min_max:
                # This is purely a statistical function, it does not influence CELOE
                self.update_min_max_horiz_exp(most_promising)

            if time.time() - self.start_time > self._max_runtime:
                return self.terminate()

            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()

            if logger.isEnabledFor(oplogging.TRACE) and j % 100 == 0:
                self._log_current_best(j)

        return self.terminate()

    def tree_node(self, node: OENode) -> TreeNode[OENode]:
        tree_parent = self.search_tree[node.concept]
        return tree_parent

    def _add_node(self, ref: OENode, tree_parent: Optional[TreeNode[OENode]]):
        # TODO:CD: Why have this constraint ?
        #  We should not ignore a concept due to this constraint.
        #  It might be the case that new path to ref.concept is a better path. Hence, we should update its parent
        #  depending on the new heuristic value.
        #  Solution: If concept exists we should compare its first heuristic value  with the new one
        if ref.concept in self.search_tree:
            # ignoring refinement, it has been refined from another parent
            return False

        norm_concept = OperandSetTransform().simplify(ref.concept)
        if norm_concept in self._seen_norm_concepts:
            norm_seen = True
        else:
            norm_seen = False
            self._seen_norm_concepts.add(norm_concept)

        self.search_tree[ref.concept] = TreeNode(ref, tree_parent, is_root=ref.is_root)
        ref_individuals = self.kb.individuals_set(ref.concept)
        ref.individuals_count = len(ref_individuals)
        self.quality_func.apply(ref, ref_individuals, self._learning_problem)  # AccuracyOrTooWeak(n)
        self._number_of_tested_concepts += 1
        if ref.quality == 0:  # > too weak
            return False
        assert 0 <= ref.quality <= 1.0
        # TODO: expression rewriting
        self.heuristic_func.apply(ref, ref_individuals, self._learning_problem)
        if not norm_seen and self.best_descriptions.maybe_add(ref):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Better description found: %s", ref)
        self.heuristic_queue.add(ref)
        # TODO: implement noise
        return True

    def _log_current_best(self, heading_step, top_n: int = 10) -> None:
        logger.debug('######## %s step Best Hypotheses ###########', heading_step)

        predictions = list(self.best_hypotheses(top_n))
        for ith, node in enumerate(predictions):
            logger.debug('{0}-\t{1}\t{2}:{3}\tHeuristic:{4}:'.format(
                ith + 1, DLSyntaxObjectRenderer().render(node.concept),
                type(self.quality_func).name, node.quality,
                node.heuristic))

    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        """
        Show search tree.
        """
        rdr = DLSyntaxObjectRenderer()

        print('######## ', heading_step, 'step Search Tree ###########')

        def tree_node_as_length_ordered_concept(tn: TreeNode[OENode]):
            return LengthOrderedNode(tn.node, tn.node.len)

        def print_partial_tree_recursive(tn: TreeNode[OENode], depth: int = 0):
            if tn.node.heuristic is not None:
                heur_idx = len(self.heuristic_queue) - self.heuristic_queue.index(tn.node)
            else:
                heur_idx = None

            if tn.node in self.best_descriptions:
                best_idx = len(self.best_descriptions.items) - self.best_descriptions.items.index(tn.node)
            else:
                best_idx = None

            render_str = rdr.render(tn.node.concept)

            depths = "`" * depth

            if best_idx is not None or heur_idx is not None:
                if best_idx is None:
                    best_idx = ""
                if heur_idx is None:
                    heur_idx = ""

                print("[%3s] [%4s] %s %s \t HE:%s Q:%f Heur:%s |RC|:%s" % (best_idx, heur_idx, depths, render_str,
                                                                           tn.node.h_exp, tn.node.quality,
                                                                           tn.node.heuristic, tn.node.refinement_count))

            for c in sorted(tn.children, key=tree_node_as_length_ordered_concept):
                print_partial_tree_recursive(c, depth + 1)

        print_partial_tree_recursive(self.search_tree[self.start_class])

        print('######## ', heading_step, 'step Best Hypotheses ###########')

        predictions = list(self.best_hypotheses(top_n))
        for ith, node in enumerate(predictions):
            print('{0}-\t{1}\t{2}:{3}\tHeuristic:{4}:'.format(ith + 1, rdr.render(node.concept),
                                                              type(self.quality_func).name, node.quality,
                                                              node.heuristic))
        print('######## Search Tree ###########\n')

    def update_min_max_horiz_exp(self, node: OENode):
        he = node.h_exp
        # update maximum value
        self.max_he = max(self.max_he, he)

        if self.min_he == he - 1:
            threshold_score = node.heuristic + 1 - node.quality

            for n in reversed(self.heuristic_queue):
                if n == node:
                    continue
                if n.h_exp == self.min_he:
                    """ we can stop instantly when another node with min. """
                    return
                if n.heuristic < threshold_score:
                    """ we can stop traversing nodes when their score is too low. """
                    break
            # inc. minimum since we found no other node which also has min. horiz. exp.
            self.min_he += 1

            if logger.isEnabledFor(oplogging.TRACE):
                logger.info("minimum horizontal expansion is now %d", self.min_he)

    def clean(self):
        self.heuristic_queue.clear()
        self.best_descriptions.clean()
        self.search_tree.clear()
        self._seen_norm_concepts.clear()
        self.max_he = 0
        self.min_he = 1
        self._learning_problem = None
        self._max_runtime = None
        super().clean()


class OCEL(CELOE):
    __slots__ = ()

    name = 'ocel_python'

    def __init__(self, knowledge_base, quality_func=None, iter_bound=None, max_num_of_concepts_tested=None,
                 terminate_on_goal=None):
        super().__init__(knowledge_base=knowledge_base,
                         quality_func=quality_func,
                         heuristic_func=OCELHeuristic(),
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested)


class Drill(AbstractDrill, BaseConceptLearner):

    def __init__(self, knowledge_base,
                 path_of_embeddings: str, refinement_operator: LengthBasedRefinement, quality_func: AbstractScorer,
                 reward_func=None, batch_size=None, num_workers=None, pretrained_model_path=None,
                 iter_bound=None, max_num_of_concepts_tested=None, verbose=None, terminate_on_goal=None,
                 max_len_replay_memory=None, epsilon_decay=None, epsilon_min=None, num_epochs_per_replay=None,
                 num_episodes_per_replay=None, learning_rate=None, max_runtime=None, num_of_sequential_actions=None,
                 num_episode=None):
        AbstractDrill.__init__(self,
                               path_of_embeddings=path_of_embeddings,
                               reward_func=reward_func,
                               max_len_replay_memory=max_len_replay_memory,
                               num_episodes_per_replay=num_episodes_per_replay,
                               batch_size=batch_size, epsilon_min=epsilon_min,
                               num_epochs_per_replay=num_epochs_per_replay,
                               representation_mode='averaging',
                               epsilon_decay=epsilon_decay,
                               num_of_sequential_actions=num_of_sequential_actions, num_episode=num_episode,
                               learning_rate=learning_rate,
                               num_workers=num_workers, verbose=verbose
                               )

        self.sample_size = 1
        arg_net = {'input_shape': (4 * self.sample_size, self.embedding_dim),
                   'first_out_channels': 32, 'second_out_channels': 16, 'third_out_channels': 8,
                   'kernel_size': 3}
        self.heuristic_func = DrillHeuristic(mode='averaging', model_args=arg_net)
        if self.learning_rate:
            self.optimizer = torch.optim.Adam(self.heuristic_func.net.parameters(), lr=self.learning_rate)

        if pretrained_model_path:
            m = torch.load(pretrained_model_path, torch.device('cpu'))
            self.heuristic_func.net.load_state_dict(m)

        BaseConceptLearner.__init__(self, knowledge_base=knowledge_base,
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

        self.attributes_sanity_checking_rl()

        self.storage_path, _ = create_experiment_folder()

    def best_hypotheses(self, n=1) -> Iterable:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
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

    def next_node_to_expand(self, t: int = None) -> RL_State:
        """
        Return a node that maximizes the heuristic function at time t
        @param t:
        @return:
        """
        if self.verbose > 5:
            self.search_tree.show_search_tree(t)
        return self.search_tree.get_most_promising()

    def initialize_class_expression_learning_problem(self, pos: Set[OWLNamedIndividual], neg: Set[OWLNamedIndividual]):
        """
            Determine the learning problem and initialize the search.
            1) Convert the string representation of an individuals into the owlready2 representation.
            2) Sample negative examples if necessary.
            3) Initialize the root and search tree.
            """
        self.clean()

        assert isinstance(pos, set) and isinstance(neg, set)
        assert 0 < len(pos) and 0 < len(neg)

        # 1.
        # Generate a Learning Problem
        self._learning_problem = PosNegLPStandard(pos=pos, neg=neg).encode_kb(self.kb)
        # 2. Obtain embeddings of positive and negative examples.
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

    def fit(self, pos: Set[OWLNamedIndividual], neg: Set[OWLNamedIndividual], max_runtime=None):
        """
        Find an OWL Class Expression h s.t.
        \\forall e in E^+ K \\model h(e)
        \\forall e in E^- K \\not\\model h(e)
        """
        assert isinstance(pos, set) and isinstance(neg, set)
        assert sum([type(_) == OWLNamedIndividual for _ in pos]) == len(pos)
        assert sum([type(_) == OWLNamedIndividual for _ in pos]) == len(neg)

        if max_runtime:
            assert isinstance(max_runtime, int)
            self.max_runtime = max_runtime
        # 2. Initialize learning problem
        root_state = self.initialize_class_expression_learning_problem(pos=pos, neg=neg)
        root_state.heuristic = 0
        self.search_tree.add(root_state)
        # (3) Add root state into search tree
        self.start_time = time.time()
        # 5. Iterate until the second criterion is satisfied.
        for i in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(i)
            next_possible_states = []
            for ref in self.apply_refinement(most_promising):
                if len(ref.instances):
                    # Compute quality
                    self.compute_quality_of_class_expression(ref)
                    if ref.quality == 0:
                        continue
                    next_possible_states.append(ref)
                    if ref.quality == 1:
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
            predicted_Q_values = self.predict_Q(current_state=most_promising, next_states=next_possible_states)
            self.goal_found = self.update_search(next_possible_states, predicted_Q_values)
            if self.goal_found:
                if self.terminate_on_goal:
                    return self.terminate()
            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()

    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        ValueError('show_search_tree')

    def terminate_training(self):
        ValueError('terminate_training')

    def fit_from_iterable(self,
                          dataset: List[Tuple[object, Set[OWLNamedIndividual], Set[OWLNamedIndividual]]],
                          max_runtime: int = None) -> List:
        """
        dataset is a list of tuples where the first item is either str or OWL class expression indicating target concept
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
            h: RL_State = self.best_hypotheses()[0]
            # TODO:CD: We need to remove this first returned boolean for the sake of readability.
            _, f_measure = F1().score(instances=h.instances_bitset, learning_problem=self._learning_problem)
            _, accuracy = Accuracy().score(instances=h.instances_bitset, learning_problem=self._learning_problem)

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


        @return:
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
        """ Create an RL_State instance """
        # Create State
        rl_state = RL_State(c, parent_node=parent_node, is_root=is_root)
        # Assign Embeddings to it. Later, assign_embeddings can be also done in RL_STATE
        self.assign_embeddings(rl_state)
        rl_state.length = self.kb.concept_len(c)
        return rl_state

    def compute_quality_of_class_expression(self, state: RL_State) -> None:
        """ Compute Quality of owl class expression of"""
        self.quality_func.apply(state, state.instances_bitset, self._learning_problem)
        self._number_of_tested_concepts += 1

    def apply_refinement(self, rl_state: RL_State) -> Generator:
        """
        Refine an OWL Class expression \\|= Observing next possible states

        1. Generate concepts by refining a node
        1.1. Compute allowed length of refinements
        1.2. Convert concepts if concepts do not belong to  self.concepts_to_ignore
             Note that          i.str not in self.concepts_to_ignore => O(1) if a set is being used.
        3. Return Generator
        """
        assert isinstance(rl_state, RL_State)
        # 1.
        for i in self.operator.refine(rl_state.concept):  # O(N)
            # TODO: CURRENTLY IGNORED the checking not wanted concetpts if i.str not in self.concepts_to_ignore:  # O(1)
            yield self.create_rl_state(i, parent_node=rl_state)

    def learn_from_illustration(self, sequence_of_goal_path: List[RL_State]):
        """
        sequence_of_goal_path: ⊤,Parent,Parent ⊓ Daughter
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
        Standard RL training loop

        1. Initialize RL environment for training

        2. Learn from an illustration if possible
        2. Training Loop
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

        state_pairs - a list of tuples containing two consecutive states
        reward      - a list of reward.

        Gamma is 1.

        Return
        X - a list of embeddings of current concept, next concept, positive examples, negative examples
        y - argmax Q value.
        """

        if self.verbose > 1:
            print('Form Experiences for the training')

        for th, consecutive_states in enumerate(state_pairs):
            e, e_next = consecutive_states
            self.experiences.append(
                (e, e_next, max(rewards[th:])))  # given e, e_next, Q val is the max Q value reachable.

    def learn_from_replay_memory(self) -> None:
        """
        Learning by replaying memory
        @return:
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

    def update_search(self, concepts, predicted_Q_values):
        """
        @param concepts:
        @param predicted_Q_values:
        @return:
        """
        for child_node, pred_Q in zip(concepts, predicted_Q_values):
            child_node.heuristic = pred_Q
            if child_node.quality > 0:  # > too weak, ignore.
                self.search_tree.add(child_node)
            if child_node.quality == 1:
                return child_node

    def assign_embeddings(self, rl_state: RL_State) -> None:
        """
        Assign embeddings to an rl state. An rl state is represented with vector representation of
        all individuals belonging to a respective OWLClassExpression
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
                    emb = torch.zeros(1, self.sample_size, self.instance_embeddings.shape[1])
                else:
                    # If|R(C)| \not= \emptyset, then take the mean of individuals.
                    str_idx = [i.get_iri().as_str() for i in rl_state.instances]
                    assert len(str_idx) > 0
                    emb = torch.tensor(self.instance_embeddings.loc[str_idx].values, dtype=torch.float32)
                    emb = torch.mean(emb, dim=0)
                    emb = emb.view(1, self.sample_size, self.instance_embeddings.shape[1])
                # (6) Assign embeddings
                rl_state.embeddings = emb
            else:
                """ Embeddings already assigned."""
                try:
                    assert rl_state.embeddings.shape == (1, self.sample_size, self.instance_embeddings.shape[1])
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
        @return:
        """
        # Save model.
        torch.save(self.heuristic_func.net.state_dict(),
                   self.storage_path + '/{0}.pth'.format(self.heuristic_func.name))

    def exploration_exploitation_tradeoff(self, current_state: AbstractNode,
                                          next_states: List[AbstractNode]) -> AbstractNode:
        """
        Exploration vs Exploitation tradeoff at finding next state.
        (1) Exploration
        (2) Exploitation
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

        (1) Predict Q values : predictions.shape => torch.Size([n, 1]) where n = len(next_states)

        (2) Find the index of max value in predictions

        (3) Use the index to obtain next state.

        (4) Return next state.
        """
        predictions: torch.Tensor = self.predict_Q(current_state, next_states)
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

    def predict_Q(self, current_state: AbstractNode, next_states: List[AbstractNode]) -> torch.Tensor:
        """
        Predict promise of next states given current state.
        @param current_state:
        @param next_states:
        @return: predicted Q values.
        """
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
            ds = PrepareBatchOfPrediction(current_state.embeddings,
                                          next_state_batch,
                                          self.emb_pos,
                                          self.emb_neg)
            predictions = self.heuristic_func.net.forward(ds.get_all())
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
        @param dataset: An iterable containing training data. Each item corresponds to a tuple of string representation
        of target concept, a set of positive examples in the form of URIs amd a set of negative examples in the form of
        URIs, respectively.
        @param relearn_ratio: An integer indicating the number of times dataset is iterated.

        Computation
        1. Dataset and relearn_ratio loops: Learn each problem relearn_ratio times,

        2. Learning loop

        3. Take post process action that implemented by subclass.

        @return: self
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


class DrillNet(nn.Module):
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

    def __init__(self, args):
        super(DrillNet, self).__init__()
        self.in_channels, self.embedding_dim = args['input_shape']
        self.loss = nn.MSELoss()

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=args['first_out_channels'],
                               kernel_size=args['kernel_size'],
                               padding=1, stride=1, bias=True)

        # Fully connected layers.
        self.size_of_fc1 = int(args['first_out_channels'] * self.embedding_dim)
        self.fc1 = nn.Linear(in_features=self.size_of_fc1, out_features=self.size_of_fc1 // 2)
        self.fc2 = nn.Linear(in_features=self.size_of_fc1 // 2, out_features=1)

        self.init()
        assert self.__sanity_checking(torch.rand(32, 4, 1, self.embedding_dim)).shape == (32, 1)

    def init(self):
        xavier_normal_(self.fc1.weight.data)
        xavier_normal_(self.conv1.weight.data)

    def __sanity_checking(self, X):
        return self.forward(X)

    def forward(self, X: torch.FloatTensor):
        # X denotes a batch of tensors where each tensor has the shape of (4, 1, embedding_dim)
        # 4 => S, S', E^+, E^- \in R^embedding_dim
        X = F.relu(self.conv1(X))
        X = X.view(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        X = F.relu(self.fc1(X))
        return self.fc2(X)


class CustomConceptLearner(CELOE):
    def __init__(self, knowledge_base, quality_func=None, iter_bound=None, max_num_of_concepts_tested=None,
                 heuristic_func=None,
                 ignored_concepts=None, verbose=None, terminate_on_goal=None):
        super().__init__(knowledge_base=knowledge_base,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound, max_num_of_concepts_tested=max_num_of_concepts_tested, verbose=verbose)
        self.name = heuristic_func.name
