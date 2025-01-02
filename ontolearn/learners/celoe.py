from ..base_concept_learner import RefinementBasedConceptLearner
from ..knowledge_base import KnowledgeBase

from ..abstracts import AbstractScorer, BaseRefinement, AbstractHeuristic, EncodedPosNegLPStandardKind
from ..learning_problem import PosNegLPStandard
from ..search import OENode, TreeNode, EvaluatedConcept, HeuristicOrderedNode, QualityOrderedNode, LengthOrderedNode

from typing import Optional, Union, Iterable, Dict
import owlapy

from owlapy.class_expression import OWLClassExpression
from contextlib import contextmanager
from sortedcontainers import SortedSet
from owlapy.utils import OrderedOWLObject
from owlapy.utils import EvaluatedDescriptionSet, ConceptOperandSorter, OperandSetTransform
import time
from itertools import islice
from owlapy.render import DLSyntaxObjectRenderer

from ..utils.static_funcs import evaluate_concept

_concept_operand_sorter = ConceptOperandSorter()

class CELOE(RefinementBasedConceptLearner):
    """Class Expression Learning for Ontology Engineering.
    Attributes:
        best_descriptions (EvaluatedDescriptionSet[OENode, QualityOrderedNode]): Best hypotheses ordered.
        best_only (bool): If False pick only nodes with quality < 1.0, else pick without quality restrictions.
        calculate_min_max (bool): Calculate minimum and maximum horizontal expansion? Statistical purpose only.
        heuristic_func (AbstractHeuristic): Function to guide the search heuristic.
        heuristic_queue (SortedSet[OENode]): A sorted set that compares the nodes based on Heuristic.
        iter_bound (int): Limit to stop the algorithm after n refinement steps are done.
        kb (KnowledgeBase): The knowledge base that the concept learner is using.
        max_child_length (int): Limit the length of concepts generated by the refinement operator.
        max_he (int): Maximal value of horizontal expansion.
        max_num_of_concepts_tested (int) Limit to stop the algorithm after n concepts tested.
        max_runtime (int): Limit to stop the algorithm after n seconds.
        min_he (int): Minimal value of horizontal expansion.
        name (str): Name of the model = 'celoe_python'.
        _number_of_tested_concepts (int): Yes, you got it. This stores the number of tested concepts.
        operator (BaseRefinement): Operator used to generate refinements.
        quality_func (AbstractScorer) The quality function to be used.
        reasoner (AbstractOWLReasoner): The reasoner that this model is using.
        search_tree (Dict[OWLClassExpression, TreeNode[OENode]]): Dict to store the TreeNode for a class expression.
        start_class (OWLClassExpression): The starting class expression for the refinement operation.
        start_time (float): The time when :meth:`fit` starts the execution. Used to calculate the total time :meth:`fit`
                            takes to execute.
        terminate_on_goal (bool): Whether to stop the algorithm if a perfect solution is found.

    """
    __slots__ = 'best_descriptions', 'max_he', 'min_he', 'best_only', 'calculate_min_max', 'heuristic_queue', \
        'search_tree', '_learning_problem', '_max_runtime', '_seen_norm_concepts'

    name = 'celoe_python'

    def __init__(self,
                 knowledge_base: KnowledgeBase=None,
                 reasoner: Optional[owlapy.abstracts.AbstractOWLReasoner] = None,
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
        """ Create a new instance of CELOE.

        Args:
            best_only (bool): If False pick only nodes with quality < 1.0, else pick without quality restrictions.
                                Defaults to False.
            calculate_min_max (bool): Calculate minimum and maximum horizontal expansion? Statistical purpose only.
                                        Defaults to True.
            refinement_operator (BaseRefinement[OENode]): Operator used to generate refinements.
                                                        Defaults to `ModifiedCELOERefinement`.
            heuristic_func (AbstractHeuristic): Function to guide the search heuristic. Defaults to `CELOEHeuristic`.
            iter_bound (int): Limit to stop the algorithm after n refinement steps are done. Defaults to 10'000.
            knowledge_base (KnowledgeBase): The knowledge base that the concept learner is using.
            max_num_of_concepts_tested (int) Limit to stop the algorithm after n concepts tested. Defaults to 10'000.
            max_runtime (int): Limit to stop the algorithm after n seconds. Defaults to 5.
            max_results (int): Maximum hypothesis to store. Defaults to 10.
            quality_func (AbstractScorer) The quality function to be used. Defaults to `F1`.
            reasoner (AbstractOWLReasoner): Optionally use a different reasoner. If reasoner=None, the reasoner of
                                    the :attr:`knowledge_base` is used.
            terminate_on_goal (bool): Whether to stop the algorithm if a perfect solution is found. Defaults to True.


        """
        super().__init__(knowledge_base=knowledge_base,
                         reasoner=reasoner,
                         refinement_operator=refinement_operator,
                         quality_func=quality_func,
                         heuristic_func=heuristic_func,
                         terminate_on_goal=terminate_on_goal,
                         iter_bound=iter_bound,
                         max_num_of_concepts_tested=max_num_of_concepts_tested,
                         max_runtime=max_runtime)
        self.search_tree: Dict[OWLClassExpression, TreeNode[OENode]] = dict()
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

    def next_node_to_expand(self, step: int) -> OENode:  # pragma: no cover
        if not self.best_only:
            for node in reversed(self.heuristic_queue):
                if node.quality < 1.0:
                    return node
            else:
                raise ValueError("No Node with lesser accuracy found")
        else:
            # from reimplementation, pick without quality criterion
            return self.heuristic_queue[-1]


    def best_hypotheses(self, n: int = 1, return_node: bool = False) -> Union[
        OWLClassExpression | Iterable[OWLClassExpression],
        OENode | Iterable[OENode]]:
        x = islice(self.best_descriptions, n)
        if n == 1:
            if return_node:
                return next(x)
            else:
                return next(x).concept
        else:
            if return_node:
                return [i for i in x]
            else:
                return [i.concept for i in x]

    def make_node(self, c: OWLClassExpression, parent_node: Optional[OENode] = None, is_root: bool = False) -> OENode:
        return OENode(c, self.kb.concept_len(c), parent_node=parent_node, is_root=is_root)
    # TODO:CD: Why do we need this ?
    @contextmanager
    def updating_node(self, node: OENode):
        """
        Removes the node from the heuristic sorted set and inserts it again.

        Args:
            Node to update.

        Yields:
            The node itself.
        """
        try:
            self.heuristic_queue.discard(node)
        except ValueError:
            # TODO:CD: We need to understand this
            pass
        yield node
        self.heuristic_queue.add(node)

    def downward_refinement(self, node: OENode) -> Iterable[OENode]:
        with self.updating_node(node):
            downward_refinements = self.operator.refine(node.concept, max_length=node.h_exp,
                                                        current_domain=self.start_class)
            sorted_downward_refinements = SortedSet((_concept_operand_sorter.sort(i) for i in downward_refinements),
                                                    key=OrderedOWLObject)
            node.increment_h_exp()
            node.refinement_count = len(sorted_downward_refinements)
            self.heuristic_func.apply(node, None, self._learning_problem)
        return [ self.make_node(i, parent_node=node) for i in sorted_downward_refinements]

    def fit(self, *args, **kwargs):
        """
        Find hypotheses that explain pos and neg.
        """
        self.clean()
        max_runtime = kwargs.pop("max_runtime", None)
        learning_problem = self.construct_learning_problem(PosNegLPStandard, args, kwargs)
        assert not self.search_tree, "search_tree cannot be None"
        self._learning_problem = learning_problem.encode_kb(self.kb)
        self._max_runtime = max_runtime if max_runtime is not None else self.max_runtime
        root = self.make_node(_concept_operand_sorter.sort(self.start_class), is_root=True)
        self._add_node(root, None)
        assert len(self.heuristic_queue) == 1, "The length of heuristic_queue must be equal to 1 after root init."
        self.start_time = time.time()
        for j in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(j)
            tree_parent = self.tree_node(most_promising)
            minimum_length = most_promising.h_exp
            # print("now refining %s", most_promising)
            for ref in self.downward_refinement(most_promising):
                # we ignore all refinements with lower length
                # (this also avoids duplicate node children)
                if ref.len < minimum_length:
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

        return self.terminate()

    def encoded_learning_problem(self) -> Optional[EncodedPosNegLPStandardKind]:
        """Fetch the most recently used learning problem from the fit method."""
        return self._learning_problem

    def tree_node(self, node: OENode) -> TreeNode[OENode]:
        """
        Get the TreeNode of the given node.

        Args:
            node: The node.

        Returns:
            TreeNode of the given node.
        """
        return self.search_tree[node.concept]

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
        e = evaluate_concept(self.kb, ref.concept, self.quality_func, self._learning_problem)

        ref.quality = e.q
        self._number_of_tested_concepts += 1
        if ref.quality == 0:  # > too weak
            return False
        assert 0 <= ref.quality <= 1.0
        # TODO: expression rewriting
        self.heuristic_func.apply(ref, e.inds, self._learning_problem)
        if not norm_seen and self.best_descriptions.maybe_add(ref):
            #if logger.isEnabledFor(logging.DEBUG):
            # print("Better description found: %s", ref)
            pass
        self.heuristic_queue.add(ref)
        # TODO: implement noise
        return True

    def _add_node_evald(self, ref: OENode, eval_: EvaluatedConcept, tree_parent: Optional[TreeNode[OENode]]):  # pragma: no cover
        norm_concept = OperandSetTransform().simplify(ref.concept)
        if norm_concept in self._seen_norm_concepts:
            norm_seen = True
        else:
            norm_seen = False
            self._seen_norm_concepts.add(norm_concept)

        self.search_tree[ref.concept] = TreeNode(ref, tree_parent, is_root=ref.is_root)

        ref.quality = eval_.q
        self._number_of_tested_concepts += 1
        if ref.quality == 0:  # > too weak
            return False
        assert 0 <= ref.quality <= 1.0
        # TODO: expression rewriting
        self.heuristic_func.apply(ref, eval_.inds, self._learning_problem)
        if not norm_seen and self.best_descriptions.maybe_add(ref):
            print("Better description found: %s", ref)
        self.heuristic_queue.add(ref)
        # TODO: implement noise
        return True

    def _log_current_best(self, heading_step:int, top_n: int = 10) -> None:
        print(f'######## {heading_step} step Best Hypotheses ###########')

        predictions = list(self.best_hypotheses(top_n, return_node=True))
        for ith, node in enumerate(predictions):
            print('{0}-\t{1}\t{2}:{3}\tHeuristic:{4}:'.format(
                ith + 1, DLSyntaxObjectRenderer().render(node.concept),
                type(self.quality_func).name, node.quality,
                node.heuristic))

    def show_search_tree(self, heading_step: str, top_n: int = 10) -> None:
        """
        Show search tree.
        """
        rdr = DLSyntaxObjectRenderer()

        print(f'######## {heading_step} step Search Tree ###########')

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

        # print_partial_tree_recursive(self.search_tree[self.start_class])

        print('######## ', heading_step, 'step Best Hypotheses ###########')

        predictions = list(self.best_hypotheses(top_n, return_node=True))
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

            # print("minimum horizontal expansion is now %d", self.min_he)

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
