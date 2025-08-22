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
from fontTools.unicodedata import block

from ..base_concept_learner import RefinementBasedConceptLearner

from ..abstracts import AbstractScorer, BaseRefinement, AbstractHeuristic, EncodedPosNegLPStandardKind, \
    AbstractKnowledgeBase
from ..learning_problem import PosNegLPStandard
from ..quality_funcs import evaluate_concept
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
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
import requests
from functools import lru_cache
import matplotlib.pyplot as plt

from ..utils.static_funcs import concept_len
from ontolearn.learners import CELOE, OCEL
_concept_operand_sorter = ConceptOperandSorter()
import random


class CELOE_PREF(CELOE):
    def __init__(self, knowledge_base, url, *args, **kwargs):
        super().__init__(knowledge_base, *args, **kwargs)
        self.pareto_front = []
        self.final_pareto_front = []
        self.url = url
        # self.max_num_of_concepts_tested = 500


    def next_node_to_expand(self, step: int) -> OENode:
        # Return node with highest (quality, preference) lexicographically
        candidates = [node for node in self.heuristic_queue if node.quality == 1.]
        if not candidates:
            # print("Not perfect candidate found")
            # Fallback: just return the least recently added node
            return self.heuristic_queue[-1]

        return max(candidates, key=lambda n: (n.quality, n.preference or 0.0))


    def best_hypotheses(self, n: int = 1, return_node: bool = False):
        pareto_sorted = sorted(self.pareto_front, key=lambda node: (-(node.quality or 0), -(node.preference or 0)))

        x = islice(pareto_sorted, n)
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
        return OENode(c, concept_len(c), parent_node=parent_node, is_root=is_root)

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


    def preference_score_utility_based(self, concept: OWLClassExpression) -> float:
        """
        Compute preference score as the average imdb:hasRatingValue of individuals
        in the extension of the OWL class expression `concept`.
        """
        subquery = owl_expression_to_sparql(concept)
        query = f"""
            PREFIX imdb: <http://example.org/imdb/>

            SELECT ?x ?rating
            WHERE {{
                {{
                    {subquery}
                }}
                ?x imdb:hasRatingValue ?rating .
            }} 
            """

        try:
            response = requests.Session().post(self.url, data={"query": query})
            response.raise_for_status()
            bindings = response.json()["results"]["bindings"]
        except Exception as e:
            print(f"[ERROR] SPARQL query failed: {e}")
            print("Query was:\n", query)
            exit(0)
            return 0.0

        # Step 3: Extract ratings
        ratings = []
        for row in bindings:
            try:
                rating_str = row["rating"]["value"]
                rating = float(rating_str)
                ratings.append(rating)
            except (KeyError, ValueError):
                continue

        # Step 4: Aggregate
        if ratings:
            return sum(ratings) / len(ratings)
        else:
            return 0.0

    def is_dominated(self, node, others):
        for other in others:
            if (other.quality >= node.quality and
                    other.preference >= node.preference and
                    (other.quality > node.quality or other.preference > node.preference)):
                return True
        return False



    def plot_pareto_front(self, title="Pareto Front after CELOE-PREF", top_k_labels=25):
        """
        Plot all concepts evaluated during search (quality vs preference),
        and highlight Pareto front in red. Annotate top-k concepts with shortened names.
        Avoid overlapping annotations for identical coordinates.
        """
        import matplotlib.pyplot as plt
        import time

        if not hasattr(self, "search_tree") or not self.search_tree:
            print("[WARNING] No search tree available for plotting.")
            return
        #
        # all_nodes = [tn.node for tn in self.search_tree.values()]
        # all_qualities = []
        # all_preferences = []
        # labels = []
        #
        # for node in all_nodes:
        #     if node.quality is not None and hasattr(node, "preference") and node.preference is not None and node.preference != 0.:
        #         all_qualities.append(node.quality)
        #         all_preferences.append(node.preference)
        #         labels.append(node)
        #
        # if not all_qualities:
        #     print("[WARNING] No concepts with both quality and preference for plotting.")
        #     return
        #
        # pareto_nodes = getattr(self, "final_pareto_front", [])
        # pareto_qualities = [n.quality for n in pareto_nodes]
        # pareto_preferences = [n.preference for n in pareto_nodes]
        #
        # # Use a larger figure and font size for paper quality
        # plt.figure(figsize=(7, 9))
        # plt.rcParams.update({'font.size': 14})  # Global font size
        #
        # plt.scatter(
        #     all_qualities, all_preferences,
        #     c="gray", alpha=0.6,
        #     label="Concepts", s=60, edgecolors='k'
        # )
        # if pareto_qualities and pareto_preferences:
        #     plt.scatter(
        #         pareto_qualities, pareto_preferences,
        #         c="red", label="Pareto front", s=120,
        #         edgecolors='black', linewidths=1.2
        #     )
        #
        # rdr = DLSyntaxObjectRenderer()
        #
        # # Annotate top-K most informative points (by combined score)
        # top_nodes = sorted(labels, key=lambda n: (n.quality or 0) + (n.preference or 0), reverse=True)[:top_k_labels]
        # seen_coords = set()
        # for node in top_nodes:
        #     coord = (round(node.quality, 4), round(node.preference, 4))
        #     if coord in seen_coords:
        #         continue
        #     seen_coords.add(coord)
        #     label = rdr.render(node.concept)
        #     short_label = (
        #         label.replace("http://example.org/imdb/", "")
        #         .replace("xsd:double", "")
        #         .replace("hasRatingValue", "Rating")
        #         .replace("∃ ", "")[:30]
        #     )
        #     plt.annotate(short_label, (node.quality, node.preference),
        #                  fontsize=10, alpha=0.85, weight='bold')
        #
        # plt.xlabel("F1-score (Quality)", fontsize=16)
        # plt.ylabel("Preference Score", fontsize=16)
        # plt.title(title, fontsize=18, weight='bold')
        # plt.legend(fontsize=14)
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.tight_layout()
        #
        # # Show non-blocking for script continuation
        # plt.show(block=True)
        # time.sleep(1)

    def _finalize_and_terminate(self):
        self.final_pareto_front = self.pareto_front
        self.plot_pareto_front(title="Pareto Front after CELOE-PREF")
        return self.terminate()

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
        self.pareto_front = [self.search_tree[root.concept].node]
        for j in range(1, self.iter_bound):
            most_promising = self.next_node_to_expand(j)
            tree_parent = self.tree_node(most_promising)
            minimum_length = most_promising.h_exp
            # print("now refining %s", most_promising)
            refinements = list(self.downward_refinement(most_promising))
            # most_promising.refinement_count = len(refinements)
            for ref in self.downward_refinement(most_promising):
                # we ignore all refinements with lower length
                # (this also avoids duplicate node children)
                if ref.len < minimum_length:# and ref.preference <= most_promising.preference:
                    continue
                # note: tree_parent has to be equal to node_tree_parent(ref.parent_node)!
                added = self._add_node(ref, tree_parent)
                if added:
                    if not self.is_dominated(ref, self.pareto_front):
                        # Keep only non-dominated concepts
                        self.pareto_front = [n for n in self.pareto_front if not self.is_dominated(n, [ref])]
                        self.pareto_front.append(ref)
                    #     # print(f"length of the pareto front is now {len(self.pareto_front)}")
                    goal_found = ref.quality == 1.0
                    if goal_found and self.terminate_on_goal:
                        print(f"[INFO] Found good concept {owl_expression_to_dl(ref.concept)} but continuing due to PREF.")
                        continue
            if self.calculate_min_max:
                # This is purely a statistical function, it does not influence CELOE
                self.update_min_max_horiz_exp(most_promising)
            if time.time() - self.start_time > self._max_runtime:
                return self._finalize_and_terminate()
            if self.number_of_tested_concepts >= self.max_num_of_concepts_tested:
                return self._finalize_and_terminate()
        return self._finalize_and_terminate()

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
        if ref.concept in self.search_tree:
            return False

        norm_concept = OperandSetTransform().simplify(ref.concept)
        if norm_concept in self._seen_norm_concepts:
            norm_seen = True
        else:
            norm_seen = False
            self._seen_norm_concepts.add(norm_concept)

        self.search_tree[ref.concept] = TreeNode(ref, tree_parent, is_root=ref.is_root)
        e = evaluate_concept(self.kb, ref.concept, self.quality_func, self._learning_problem)

        #compute quality
        ref.quality =  e.q

        self._number_of_tested_concepts += 1
        if self._number_of_tested_concepts % 10 == 0.0:
            print(f"number of tested concepts {self._number_of_tested_concepts}")
        #
        # if self._number_of_tested_concepts >= 500:
        #     return self._finalize_and_terminate()

        if ref.quality == 0:
            return False
        assert 0 <= ref.quality <= 1.0



        # Heuristic scoring
        self.heuristic_func.apply(ref, e.inds, self._learning_problem)

        if not norm_seen and self.best_descriptions.maybe_add(ref):
            pass  # could log here

        self.heuristic_queue.add(ref)
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
        Show the search tree with preference information.
        """
        rdr = DLSyntaxObjectRenderer()

        print(f'######## {heading_step} step Search Tree ###########')

        def tree_node_as_length_ordered_concept(tn: TreeNode[OENode]):
            return LengthOrderedNode(tn.node, tn.node.len)

        def print_partial_tree_recursive(tn: TreeNode[OENode], depth: int = 0):
            node = tn.node
            if node.heuristic is not None:
                heur_idx = len(self.heuristic_queue) - self.heuristic_queue.index(node)
            else:
                heur_idx = None

            if node in self.best_descriptions:
                best_idx = len(self.best_descriptions.items) - self.best_descriptions.items.index(node)
            else:
                best_idx = None

            is_pareto = (
                    hasattr(self, "final_pareto_front")
                    and any(node.concept == p.concept for p in self.final_pareto_front)
            )

            render_str = rdr.render(node.concept)
            depths = "`" * depth

            if best_idx is not None or heur_idx is not None:
                if best_idx is None:
                    best_idx = ""
                if heur_idx is None:
                    heur_idx = ""

                print(
                    "[%3s] [%4s] %s %s \t HE:%s Q:%f Heur:%s |RC|:%s" % (
                        best_idx,
                        heur_idx,
                        depths,
                        render_str,
                        node.h_exp,
                        round(node.quality, 4) if node.quality is not None else "None",
                        # round(node.preference, 4) if hasattr(node,
                        #                                      "preference") and node.preference is not None else "None",
                        # "(PARETO)" if is_pareto else "",
                        node.heuristic,
                        node.refinement_count
                    )
                )

            for c in sorted(tn.children, key=tree_node_as_length_ordered_concept):
                print_partial_tree_recursive(c, depth + 1)

        # Start printing from the root
        # root_node = self.search_tree[self.start_class]
        # print_partial_tree_recursive(root_node)

        print('######## ', heading_step, 'step Best Hypotheses ###########')

        predictions = list(self.best_hypotheses(top_n, return_node=True))
        for ith, node in enumerate(predictions):
            print('{0}-\t{1}\t{2}:{3}\tPref.:{4}:\tHeuristic:{5}:'.format(
                ith + 1, DLSyntaxObjectRenderer().render(node.concept),
                type(self.quality_func).name, node.quality, node.preference,
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
