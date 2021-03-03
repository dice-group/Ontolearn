import logging
import time
from contextlib import contextmanager
from itertools import islice
from typing import Iterable, Optional, TypeVar, Dict

import pandas as pd
from sortedcontainers import SortedSet

from ontolearn.search import HeuristicOrderedNode, OENode, TreeNode, LengthOrderedNode, LBLNode, LBLSearchTree, \
    QualityOrderedNode
from . import KnowledgeBase
from .abstracts import AbstractScorer, BaseRefinement, AbstractHeuristic
from .base_concept_learner import BaseConceptLearner
from .core.owl.utils import EvaluatedDescriptionSet, OrderedOWLObject, ConceptOperandSorter
from .heuristics import CELOEHeuristic, OCELHeuristic
from .learning_problem import PosNegLPStandard
from .metrics import F1
from owlapy.model import OWLClassExpression
from owlapy.render import DLSyntaxRenderer
from .refinement_operators import LengthBasedRefinement
from .search import SearchTreePriorityQueue
from .utils import oplogging

_N = TypeVar('_N')
_O = TypeVar('_O')

pd.set_option('display.max_columns', 100)

logger = logging.getLogger(__name__)

_concept_operand_sorter = ConceptOperandSorter()


class CELOE(BaseConceptLearner[OENode]):
    __slots__ = 'best_descriptions', 'max_he', 'min_he', 'best_only', 'calculate_min_max', 'heuristic_queue', \
                'search_tree'

    name = 'celoe_python'

    kb: KnowledgeBase

    max_he: int
    min_he: int
    best_only: bool
    calculate_min_max: bool

    search_tree: Dict[OWLClassExpression, TreeNode[OENode]]
    heuristic_queue: 'SortedSet[OENode]'
    best_descriptions: EvaluatedDescriptionSet[OENode, QualityOrderedNode]

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 learning_problem: PosNegLPStandard,
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
                         learning_problem=learning_problem,
                         refinement_operator=refinement_operator,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound,
                         max_num_of_concepts_tested=max_num_of_concepts_tested,
                         max_runtime=max_runtime)

        self.search_tree = dict()
        self.heuristic_queue = SortedSet(key=HeuristicOrderedNode)
        self.best_descriptions = EvaluatedDescriptionSet(max_size=max_results, ordering=QualityOrderedNode)

        self.best_only = best_only
        self.calculate_min_max = calculate_min_max

        self.max_he = 0
        self.min_he = 1

    def next_node_to_expand(self, step: int) -> OENode:
        """
        Return most promising node/concept based
        """
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
        r = OENode(c, self.kb.cl(c), parent_node=parent_node, is_root=is_root)
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
                    )
                ,
                key=OrderedOWLObject)

            node.increment_h_exp()
            node.refinement_count = len(refinements)
            self.heuristic_func.apply(node)

        def make_node_with_parent(c: OWLClassExpression):
            return self.make_node(c, parent_node=node)

        return map(make_node_with_parent, refinements)

    def fit(self,
            max_runtime: Optional[int] = None):
        """
        Find hypotheses that explain pos and neg.
        """
        assert not self.search_tree

        if max_runtime is not None:
            self.max_runtime = max_runtime

        root = self.make_node(_concept_operand_sorter.sort(self.start_class), is_root=True)
        self._add_node(root, None)
        assert len(self.heuristic_queue) == 1

        self.start_time = time.time()
        for j in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(j)
            tree_parent = self.node_tree_parent(most_promising)
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

            if time.time() - self.start_time > self.max_runtime:
                return self.terminate()

            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()

        return self.terminate()

    def node_tree_parent(self, node: OENode) -> TreeNode[OENode]:
        tree_parent = self.search_tree[node.concept]
        return tree_parent

    def _add_node(self, ref: OENode, tree_parent: Optional[TreeNode[OENode]]):
        if ref.concept in self.search_tree:
            # ignoring refinement, it has been refined from another parent
            return False

        self.search_tree[ref.concept] = TreeNode(ref, tree_parent, is_root=ref.is_root)
        ref_individuals = self.kb.individuals_set(ref.concept)
        ref.individuals_count = len(ref_individuals)
        self.quality_func.apply(ref, ref_individuals)  # AccuracyOrTooWeak(n)
        if ref.quality == 0:  # > too weak
            return False
        assert 0 <= ref.quality <= 1.0
        # TODO: expression rewriting
        self.heuristic_func.apply(ref)
        if self.best_descriptions.maybe_add(ref):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Better description found: %s", ref)
        self.heuristic_queue.add(ref)
        # TODO: implement noise
        return True

    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        """
        Show search tree.
        """
        rdr = DLSyntaxRenderer()

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
        self.max_he = 0
        self.min_he = 1
        self.heuristic_queue.clear()
        self.best_descriptions.items.clear()
        self.search_tree.clear()
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


class LengthBaseLearner(BaseConceptLearner):
    """
    CD: An idea for next possible work.
    Propose a Heuristic func based on embeddings
    Use LengthBasedRef.
    """
    __slots__ = 'search_tree', 'concepts_to_ignore', 'min_length'

    name = 'LengthBaseLearner'

    kb: KnowledgeBase
    search_tree: LBLSearchTree
    min_length: int

    def __init__(self, *,
                 knowledge_base: KnowledgeBase,
                 learning_problem: PosNegLPStandard,
                 refinement_operator: Optional[BaseRefinement] = None,
                 search_tree: Optional[LBLSearchTree] = None,
                 quality_func: Optional[AbstractScorer] = None,
                 heuristic_func: Optional[AbstractHeuristic] = None,
                 iter_bound: int = 10_000,
                 terminate_on_goal: bool = False,
                 max_num_of_concepts_tested: int = 10_000,
                 min_length: int = 1,
                 ignored_concepts = None):

        if ignored_concepts is None:
            ignored_concepts = {}
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(knowledge_base=knowledge_base)
        if quality_func is None:
            quality_func = F1(learning_problem=learning_problem)
        if heuristic_func is None:
            heuristic_func = CELOEHeuristic()
        if search_tree is None:
            search_tree = SearchTreePriorityQueue(quality_func=quality_func, heuristic_func=heuristic_func)

        super().__init__(knowledge_base=knowledge_base,
                         learning_problem=learning_problem,
                         refinement_operator=refinement_operator,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound,
                         max_num_of_concepts_tested=max_num_of_concepts_tested
                         )
        self.search_tree = search_tree
        self.concepts_to_ignore = ignored_concepts
        self.min_length = min_length

    def get_node(self, c: OWLClassExpression, **kwargs):
        return LBLNode(c, self.kb.cl(c), self.kb.individuals_set(c), **kwargs)

    def next_node_to_expand(self, step) -> LBLNode:
        return self.search_tree.get_most_promising()

    def downward_refinement(self, node: LBLNode):
        assert isinstance(node, LBLNode)
        refinements = (self.get_node(i, parent_node=node) for i in
                       self.operator.refine(node.concept, max_length=node.len + 1 + self.min_length)
                       if i not in self.concepts_to_ignore)
        return refinements

    def fit(self):
        """
        Find hypotheses that explain pos and neg.
        """
        self.start_time = time.time()
        root = self.get_node(self.start_class, is_root=True)
        self.search_tree.add_root(node=root)
        for j in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(j)
            for ref in self.downward_refinement(most_promising):
                goal_found = self.search_tree.add_node(node=ref, parent_node=most_promising)
                if goal_found:
                    if self.terminate_on_goal:
                        return self.terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self.terminate()
        return self.terminate()

    def clean(self):
        self.quality_func.clean()
        self.heuristic_func.clean()
        self.search_tree.clean()
        self.concepts_to_ignore.clear()

    def best_hypotheses(self, n=10) -> Iterable[LBLNode]:
        yield from self.search_tree.get_top_n(n)

    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        rdr = DLSyntaxRenderer()

        self.search_tree.show_search_tree(root_concept=self.start_class, heading_step=heading_step)

        print('######## ', heading_step, 'step Best Hypotheses ###########')

        predictions = list(self.best_hypotheses(top_n))
        for ith, node in enumerate(predictions):
            print('{0}-\t{1}\t{2}:{3}\tHeuristic:{4}:'.format(ith + 1, rdr.render(node.concept),
                                                              type(self.quality_func).name, node.quality,
                                                              node.heuristic))
        print('######## Search Tree ###########\n')


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
