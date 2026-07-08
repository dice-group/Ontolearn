"""
Covering-based Concept Learning Algorithm with Fragment Collection

The refinement operator automatically collects high-precision, low-recall fragments
during refinement. This script just runs beam search and retrieves the collected
fragments at the end.
"""
import heapq
import json
import time
from typing import Set, Optional, List

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import PruneCELBasedRefinement
from owlapy.class_expression import OWLClassExpression, OWLThing, OWLClass
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.utils import simplify_class_expression, get_expression_length

from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectIntersectionOf, \
    OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, OWLObjectUnionOf, OWLClass, \
    OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLDataSomeValuesFrom, \
    OWLDatatypeRestriction, OWLDataHasValue, OWLObjectExactCardinality, OWLObjectOneOf


class BeamSearchLearner:
    """
    Simple beam search learner that relies on the refinement operator
    to collect high-precision fragments automatically.
    """
    
    def __init__(self, kb: KnowledgeBase, operator: PruneCELBasedRefinement,
                 beam_width: int = 10,
                 max_depth: int = 5,
                 max_concepts: int = 500,
                 time_limit: float = 300.0,
                 verbose: bool = False):
        """
        Initialize the learner.
        
        Args:
            kb: Knowledge base
            operator: Refinement operator (collects fragments automatically)
            beam_width: Number of top concepts to refine each iteration (default: 10)
            max_depth: Maximum refinement depth (default: 5)
            max_concepts: Budget for best_first_search — max heap pops (default: 500)
            time_limit: Time limit in seconds (default: 300)
            verbose: Print per-pop progress (default: False)
        """
        self.kb = kb
        self.operator = operator
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_concepts = max_concepts
        self.time_limit = time_limit
        self.verbose = verbose
        self.renderer = DLSyntaxObjectRenderer()
        
        # Statistics
        self.total_refinements_explored = 0
    
    def get_instances(self, concept: OWLClassExpression) -> Set[OWLNamedIndividual]:
        """Get all instances of a concept."""
        return self.kb.individuals_set(concept)
    
    def evaluate(self, concept: OWLClassExpression, pos: Set, neg: Set):
        """
        Evaluate a concept on given examples.
        
        Returns:
            (f1, precision, recall, covered_positives)
        """
        instances = self.get_instances(concept)
        
        covered_pos = instances & pos
        covered_neg = instances & neg
        
        tp = len(covered_pos)
        fp = len(covered_neg)
        fn = len(pos - covered_pos)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall, covered_pos
    
    def best_first_search(self, pos: Set, neg: Set, start, max_concepts: int = 500,
                          _subproblem_keys=None, _allow_recursion: bool = True) -> Optional[OWLClassExpression]:
        """
        K-best search with PruneCEL-R inline recursion.

        Priority = F1 - length * length_penalty  (PruneCEL's LengthBasedRefinementScorer)

        Skip rule: enqueue child only if child_f1 > local_best OR child adds a new role.
          local_best ratchets up within each K-best batch, blocking same-F1 duplicates
          (e.g. Female sets local_best=0.804, so ¬Male, ¬Father, … are all dropped).

          when _allow_recursion=True, matches SimpleRecursivePruneCEL.java):
          During each batch expansion, any child with precision >= precision_threshold
          that covers E+' ⊂ E+ (|E+'| >= min_recursion_pos, |E+'| < |E+|-1) is
          collected as a recursion candidate.  After the batch, each candidate triggers
          a limited sub-search on (E+ \ E+', E-).  The best sub-result is combined as
          D ⊔ sub_result and pushed back into the main heap, competing normally with
          all other concepts.  This short-circuits long search paths by decomposing
          the problem when a clean partial solution is found.
        """
        if _subproblem_keys is None:
            _subproblem_keys = set()

        min_recursion_pos = max(2, int(len(pos) * 0.1 + 0.999))
        # Match Java's MAX_ITERATIONS_FOR_RECURSION = 10
        _RECURSIVE_BUDGET = max(10, self.beam_width * 2)

        # Seed the heap with ⊤
        top_f1, _, _, top_covered = self.evaluate(OWLThing, pos, neg)
        top_tp = len(top_covered)
        top_score = self.operator.refinement_score(OWLThing, top_f1, top_tp)

        _seq = 0
        heap = [(-top_score, -top_f1, _seq, OWLThing, top_f1, top_tp)]
        heapq.heapify(heap)

        seen = {str(OWLThing)}
        best_concept = OWLThing
        best_f1 = top_f1
        pops = 0

        label = "" if _allow_recursion else " [sub-problem]"
        print(f"\nK-best search{label} (budget={max_concepts}, beam={self.beam_width}, |pos|={len(pos)}, penalty={self.operator.length_penalty})")
        print("=" * 80)

        while heap and pops < max_concepts:
            if (time.time() - start) >= self.time_limit:
                break

            # Pop beam_width concepts per round (K-best)
            batch = []
            while heap and len(batch) < self.beam_width:
                batch.append(heapq.heappop(heap))

            batch_fragments = []   
            done = False

            for neg_score, _, _, parent, parent_f1, parent_tp in batch:
                pops += 1
                if pops > max_concepts or (time.time() - start) >= self.time_limit:
                    done = True
                    break

                parent_n_q = self.operator.count_quantifiers(parent)

                if self.verbose:
                    concept_str = self.renderer.render(parent)
                    print(f"  [{pops:4d}] score={-neg_score:.4f}  F1={parent_f1:.3f}  {concept_str[:70]}")

                # local_best ratchets up within each expansion batch.
                # First child to hit F1=0.804 sets the bar; all same-F1 siblings are dropped.
                local_best = best_f1

                try:
                    for child, child_f1, _, precision, recall, tp in self.operator.refine(parent):
                        child_str = str(child)
                        if child_str in seen:
                            continue
                        seen.add(child_str)
                        self.total_refinements_explored += 1

                    
                        if (_allow_recursion
                                and precision >= self.operator.precision_threshold
                                and tp >= min_recursion_pos
                                and tp < len(pos) - 1):
                            inst_cache = getattr(self.operator, '_inst_cache', {})
                            batch_fragments.append((child, child_f1, tp, inst_cache.get(child_str)))
                        # ──────────────────────────────────────────────────────────────

                        # ── Skip rule ─────────────────────────────────────────────────
                        child_n_q = self.operator.count_quantifiers(child)
                        added_role = child_n_q > parent_n_q
                        if child_f1 <= local_best and not added_role:
                            continue
                        local_best = max(local_best, child_f1)
                        # ──────────────────────────────────────────────────────────────

                        child_score = self.operator.refinement_score(child, child_f1, tp)
                        _seq += 1
                        heapq.heappush(heap, (-child_score, -child_f1, _seq, child, child_f1, tp))

                        if child_f1 == 1.0:
                            print(f"\n✓ PERFECT F1 at pop {pops}!")
                            print(f"  {self.renderer.render(child)}")
                            return child

                        if child_f1 > best_f1:
                            best_concept = child
                            best_f1 = child_f1

                except Exception as e:
                    print(f"  Exception refining {self.renderer.render(parent)}: {e}")

            if done:
                break
            # mostPreciseExpressions, run limited sub-search, combine into main heap.
            if _allow_recursion and batch_fragments and best_f1 < 1.0:
                for fragment, frag_f1, frag_tp, frag_instances in batch_fragments:
                    if (time.time() - start) >= self.time_limit:
                        break

                    # Retrieve instances if not already cached
                    if frag_instances is None:
                        frag_instances = self.get_instances(fragment)

                    # Remaining positives after removing those covered by fragment
                    remaining_pos = pos - frag_instances
                    if len(remaining_pos) < 2:
                        continue

                    # Deduplicate sub-problems by the remaining positive set
                    sub_key = frozenset(i.str for i in remaining_pos)
                    if sub_key in _subproblem_keys:
                        continue
                    _subproblem_keys.add(sub_key)

                    frag_label = self.renderer.render(fragment)[:50]
                    print(f"\n  ↳ [PruneCEL-R] '{frag_label}' covers {frag_tp}/{len(pos)} pos"
                          f" → sub-problem: {len(remaining_pos)} remaining")

                    # Save main problem cache, run sub-search, restore
                    saved_cache = dict(getattr(self.operator, '_inst_cache', {}))
                    self.operator.set_input_examples(frozenset(remaining_pos), frozenset(neg))

                    sub_result = self.best_first_search(
                        remaining_pos, neg, start,
                        max_concepts=_RECURSIVE_BUDGET,
                        _subproblem_keys=_subproblem_keys,
                        _allow_recursion=False,   # no further recursion inside sub-search
                    )

                    # Restore main problem: re-set examples and merge both caches
                    sub_cache = dict(getattr(self.operator, '_inst_cache', {}))
                    self.operator.set_input_examples(frozenset(pos), frozenset(neg))
                    if hasattr(self.operator, '_inst_cache'):
                        self.operator._inst_cache.update(saved_cache)
                        self.operator._inst_cache.update(sub_cache)  # instance sets are problem-independent

                    if sub_result is None:
                        continue

                    # Combine: D ⊔ sub_result, score against full pos/neg
                    combined = OWLObjectUnionOf([fragment, sub_result])
                    combined_str = str(combined)
                    if combined_str in seen:
                        continue
                    seen.add(combined_str)

                    c_f1, _, _, c_covered = self.evaluate(combined, pos, neg)
                    c_tp = len(c_covered)
                    c_score = self.operator.refinement_score(combined, c_f1, c_tp)
                    _seq += 1
                    heapq.heappush(heap, (-c_score, -c_f1, _seq, combined, c_f1, c_tp))
                    print(f"    ↳ Combined: F1={c_f1:.3f}  {self.renderer.render(combined)[:60]}")

                    if c_f1 == 1.0:
                        print(f"\n✓ PERFECT F1 via recursion!")
                        return combined

                    if c_f1 > best_f1:
                        best_concept = combined
                        best_f1 = c_f1
            # ─────────────────────────────────────────────────────────────────────

        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s  pops={pops}  explored={self.total_refinements_explored}  best_f1={best_f1:.3f}")
        return best_concept


    def learn_recursive(self, pos, neg):
        """
        Single-pass learning.
        Union construction is handled internally by best_first_search via
        inline recursive sub-problems — no outer covering loop needed.
        """
        start = time.time()
        self.operator.set_input_examples(frozenset(pos), frozenset(neg))
        result = self.best_first_search(set(pos), set(neg), start, max_concepts=self.max_concepts)

        f1 = 0.0
        if result is not None:
            f1, _, _, _ = self.evaluate(result, set(pos), set(neg))

        print(f"\n{'=' * 80}")
        print(f"FINAL RESULT")
        print(f"{'=' * 80}")
        print(f"{self.renderer.render(result) if result is not None else 'None'}")
        print(f"\nF1: {f1:.3f}")
        print(f"{'=' * 80}")
        return result



def main():
    """Main test function."""
    # Load KB
    print("Loading knowledge base...")
    kb = KnowledgeBase(path='KGs/Family/family.owl')
    
    # Load learning problems
    print("Loading learning problems...")
    with open('LPs/Family/lps.json', 'r') as f:
        lps = json.load(f)
    
    # Test on Grandgranddaughter problem
    problem_name = 'Uncle'  # 'Aunt' or 'Uncle'
    problem = lps['problems'][problem_name]
    pos = frozenset({OWLNamedIndividual(IRI.create(iri)) for iri in problem['positive_examples']})
    neg = frozenset({OWLNamedIndividual(IRI.create(iri)) for iri in problem['negative_examples']})

    
    print(f"\nLearning problem: {problem_name}")
    print(f"Positives: {len(pos)}")
    print(f"Negatives: {len(neg)}")
    
    # Create refinement operator
    print("\nInitializing refinement operator...")
    operator = PruneCELBasedRefinement(
        knowledge_base=kb,
        sparql_endpoint='http://localhost:3030/family/sparql'
        # max_concepts=100
    )

    # operator = LengthBasedRefinement(
    #     knowledge_base=kb
    # )

    operator.set_input_examples(pos, neg)

    # PruneCEL-R requires precision=1.0: fragment must cover some positives with ZERO negatives.
    if isinstance(operator, PruneCELBasedRefinement):
        operator.precision_threshold = 1.0

    # Create learner
    learner = BeamSearchLearner(
        kb=kb,
        operator=operator,
        beam_width=5,
        max_concepts=500,
        time_limit=60.0,
        verbose=True
    )
    
    # Run learning
    start_time = time.time()
    best_concept = learner.learn_recursive(pos=set(pos), neg=set(neg))
    elapsed_time = time.time() - start_time
    print(f"\n✓ Learning complete!")
    print(f"Elapsed time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()