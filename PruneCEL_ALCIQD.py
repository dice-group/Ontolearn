"""
Covering-based Concept Learning Algorithm with Fragment Collection

The refinement operator automatically collects high-precision, low-recall fragments
during refinement. This script just runs beam search and retrieves the collected
fragments at the end.
"""
import cProfile
import pstats
import json
import time
from bisect import insort
from typing import Set, Optional, List
from ontolearn.triple_store import TripleStore

from ontolearn.knowledge_base import KnowledgeBase
#from ontolearn.refinement_operators_23042026 import PruneCELBasedRefinement
from ontolearn.refinement_operators_10_06_2026 import PruneCELBasedRefinement
from ontolearn.utils.static_funcs import compute_f1_score_from_confusion_matrix
from owlapy import owl_expression_to_sparql_with_confusion_matrix
from owlapy.class_expression import OWLClassExpression, OWLThing, OWLClass
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.utils import simplify_class_expression, get_expression_length

from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectIntersectionOf, \
    OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, OWLObjectUnionOf, OWLClass, \
    OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLDataSomeValuesFrom, \
    OWLDatatypeRestriction, OWLDataHasValue, OWLObjectExactCardinality, OWLObjectOneOf
from sympy.physics.continuum_mechanics import truss


class BeamSearchLearner:
    """
    Simple beam search learner that relies on the refinement operator
    to collect high-precision fragments automatically.
    """

    def __init__(self, kb: KnowledgeBase, operator: PruneCELBasedRefinement,
                 time_limit: float = 120.0,
                 max_beam_size: int = 1000,
                 length_penalty: float = 0.1):
        """
        Initialize the learner.

        Args:
            kb: Knowledge base
            operator: Refinement operator (collects fragments automatically)
            time_limit: Time limit in seconds (default: 300)
        """
        self.kb = kb
        self.operator = operator
        self.time_limit = time_limit
        self.renderer = DLSyntaxObjectRenderer()
        self.length_penalty = length_penalty
        self.max_beam_size = max_beam_size

        # Statistics
        self.total_refinements_explored = 0

    def refinement_score(self, concept: OWLClassExpression, f1: float, tp: int) -> float:
        """
        Calculate refinement score = F1 - length * penalty.
        Filters out concepts with tp <= 1.
        """
        if tp <= 1:
            return 0.0
        length = self.concept_length(concept)
        return f1 - length * self.length_penalty

    def concept_length(self, concept: OWLClassExpression) -> int:
        """Calculate concept length (number of symbols)."""
        if isinstance(concept, OWLClass):
            return 1
        if isinstance(concept, OWLObjectComplementOf):
            return 1 + self.concept_length(concept.get_operand())
        if isinstance(concept, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
            return 1 + sum(self.concept_length(op) for op in concept.operands())
        if isinstance(concept, (OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom)):
            return 2 + self.concept_length(concept.get_filler())

        # Q
        if isinstance(concept, (OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality)):
            return 3 + self.concept_length(concept.get_filler())
        # D
        if isinstance(concept, OWLDataSomeValuesFrom):
            return 2 + self.concept_length(concept.get_filler())
        return 1


    def get_instances(self, concept: OWLClassExpression) -> Set[OWLNamedIndividual]:
        """Get all instances of a concept."""
        return self.kb.individuals_set(concept)

    def evaluate(self, concept: OWLClassExpression, pos: Set, neg: Set):
        """
        Evaluate a concept on given examples.

        Returns:
            (f1, precision, recall, covered_positives, covered_negatives)
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

        return f1, precision, recall, covered_pos, covered_neg


    def evaluate_with_confusion_matrix(self, concept: OWLClassExpression, pos: Set, neg: Set):

        # Special case: owl:Thing may not be explicitly asserted as rdf:type owl:Thing
        if concept == OWLThing:
            tp = len(pos)
            fp = len(neg)
            fn = 0
            tn = 0
            confusion_matrix = {
                "tp": tp,
                "fp": fp,
                "fn": 0,
                "tn": 0,
            }

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = compute_f1_score_from_confusion_matrix(confusion_matrix=confusion_matrix)

        else:
            sparql_query = owl_expression_to_sparql_with_confusion_matrix(expression=concept,
                                                                          positive_examples=pos,
                                                                          negative_examples=neg)
            bindings = self.kb.query(sparql_query).json()["results"]["bindings"]
            assert len(bindings) == 1
            bindings = bindings.pop()

            confusion_matrix = {k: v["value"] for k, v in bindings.items()}
            tp = tp=int(confusion_matrix["tp"])
            fp = fp=int(confusion_matrix["fp"])
            fn = fn=int(confusion_matrix["fn"])
            tn = tn=int(confusion_matrix["tn"])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = compute_f1_score_from_confusion_matrix(confusion_matrix=confusion_matrix)

        return f1, precision, recall, tp, fp


    def beam_search(self, pos: Set, neg: Set, start, negated):

        # Keep the beam incrementally ordered to avoid re-sorting the full list
        # on every loop. Entry shape: (-refinement_score, -F1, sequence, payload).
        beam = []
        beam_sequence = 0

        def push_beam(item):
            nonlocal beam_sequence
            # Sort primarily by F1 (highest first), then by R-score (highest first) as tie-breaker
            insort(beam, (-item[2], -item[1], beam_sequence, item))
            # insort(beam, (-item[2], beam_sequence, item))
            beam_sequence += 1
            if len(beam) > self.max_beam_size:
                beam.pop()

        f1, precision, recall, covered_pos, covered_neg = self.evaluate_with_confusion_matrix(OWLThing, pos, neg)
        refinement_score = self.refinement_score(OWLThing, f1, covered_pos)
        best_concept = (OWLThing, f1, refinement_score, precision, recall, covered_pos, covered_neg)
        push_beam(best_concept)

        visited = set()
        visited.add(str(OWLThing))

        best_f1 = f1

        print(f"\nStarting beam search with {len(pos)} positives and {len(neg)} negatives")
        print('pos:'+str(pos))
        print('neg'+str(neg))
        print("=" * 80)

        # check if the time limit has been reached before starting the loop
        while (time.time() - start) < self.time_limit:

            if not beam:
                break

            # Always pick the concept has highest refinement score (not necessarily highest F1)
            current = beam.pop(0)[3]  # remove best payload
            # current = beam.pop(0)[2]  # remove best payload

            concept, f1, refinement_score, precision, recall, covered_pos, covered_neg = current


            print(f"\nRefining Best R-score concept: {self.renderer.render(concept)} Best F1 for now: {best_f1:.3f}")

            # Refine the concept has highest refinement score (not necessarily highest F1)
            refinements, total_refinement_explored = self.operator.refine(concept)
            self.total_refinements_explored += total_refinement_explored

            # simiply negated all class expression to double the amount

            if negated:
                negated_refinements = []
                for child in refinements:
                    concept_expr, f1, refinement_score, precision, recall, pos_hit, neg_hit = child

                    # Create negation
                    negated_concept = OWLObjectComplementOf(concept_expr)

                    # For negated concept ¬C
                    neg_tp = len(pos) - pos_hit  # ¬C covers positives that C didn't cover
                    neg_fp = len(neg) - neg_hit  # ¬C covers negatives that C didn't cover
                    neg_fn = pos_hit  # ¬C doesn't cover positives that C did cover

                    neg_precision = neg_tp / (neg_tp + neg_fp) if (neg_tp + neg_fp) > 0 else 0.0
                    neg_recall = neg_tp / (neg_tp + neg_fn) if (neg_tp + neg_fn) > 0 else 0.0
                    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (
                                                                                                          neg_precision + neg_recall) > 0 else 0.0

                    neg_refinement_score = self.refinement_score(negated_concept, neg_f1, neg_tp)

                    negated_refinements.append(
                        (negated_concept, neg_f1, neg_refinement_score, neg_precision, neg_recall, neg_tp, neg_fp))

                refinements.extend(negated_refinements)
                self.total_refinements_explored += total_refinement_explored

            for child in refinements:

                child_str = str(child[0])
                if child_str in visited:
                    continue
                visited.add(child_str)

                # get the concept evaluation metrics
                f1, refinement_score, precision, recall, pos_hit = child[1],child[2],child[3],child[4],child[5]

                # print("concept:"+self.renderer.render(child[0])+"precision:", precision, "recall:", recall)
                # Check if necessary to start recursive

                if (precision >= self.operator.precision_threshold and
                        int(pos_hit) >= int(self.operator.min_positive_coverage) and
                        int(pos_hit) >= int(self.operator.min_positive_percentage * len(pos))):
                    print("\n" + "-" * 70)
                    print("🔹 High-Precision / Low-Recall Fragment Found")
                    print("-" * 70)
                    print(f"Concept: {self.renderer.render(child[0])}")
                    print(f"F1 score: {f1:.3f}")
                    print(f"Refinement score: {refinement_score:.3f}")
                    print(f"Precision: {precision:.3f}")
                    print(f"Recall: {recall:.3f}")
                    print(f"Covers: {pos_hit}/{len(pos)} positives")
                    print(f"Explored refinements so far: {total_refinement_explored}")
                    print("-" * 70 + "\n")

                    return child

                # If F1 is perfect, stop immediately
                if f1 == 1.0:
                    print("\n PERFECT F1 FOUND!")
                    print(f"Explored refinements so far: {total_refinement_explored}")
                    return child

                # if newly generated concept has better F1 than current best, update best
                if f1 > best_f1:
                    best_f1 = f1
                    best_concept = child

                # add the newly generated concept to the beam
                push_beam(child)

            print(f"Beam size: {len(beam)} Total refinements explored: {self.total_refinements_explored}")

        print(f"\nSearch completed")
        print(f"Best F1: {best_f1:.3f}")

        return best_concept

    def learn_recursive(self, pos, neg):

        # recursive learning

        remaining_pos = set(pos)
        fragments = []

        start = time.time()
        while remaining_pos:

            self.operator.set_input_examples(frozenset(remaining_pos), frozenset(neg))
            print(f"len(pos): {len(remaining_pos)} len(neg): {len(neg)}")

            if (time.time() - start) >= self.time_limit:
                print(f"\n⏱ Time limit reached!")
                break

            fragment = self.beam_search(
                remaining_pos,
                neg,
                start,
                negated=False
            )

            f1, precision, recall, _, _ = self.evaluate_with_confusion_matrix(
                fragment[0],
                remaining_pos,
                neg
            )

            instances = self.get_instances(fragment[0])
            covered = instances & remaining_pos

            # Only add if precise enough
            if precision >= 0.3 and len(fragments) > 0:
                fragments.append(fragment[0])
                remaining_pos -= covered
            elif len(fragments) == 0:
                fragments.append(fragment[0])
                remaining_pos -= covered
            else:
                print(f"SKIPPING imprecise fragment (precision={precision:.3f})")
                break

        if len(fragments) > 1:
            final_concepts = OWLObjectUnionOf(fragments)
            f1, _, _, _, _ = self.evaluate_with_confusion_matrix(final_concepts, pos, neg)
            print(f"\n{'=' * 80}")
            print(f"FINAL RESULT")
            print(f"{'=' * 80}")
            print(f"{str(self.renderer.render(final_concepts))}")
            print(f"\nF1: {f1:.3f}")
            print(f"{'=' * 80}")
            return OWLObjectUnionOf(fragments)

        else:
            f1, _, _, _, _ = self.evaluate_with_confusion_matrix(fragments[0], pos, neg)
            print(f"\n{'=' * 80}")
            print(f"FINAL RESULT")
            print(f"{'=' * 80}")
            print(f"{str(self.renderer.render(fragments[0]))}")
            print(f"\nF1: {f1:.3f}")
            print(f"{'=' * 80}")
            return fragments[0]



def main():

    # load the knowledge base via triple store, give a endpoint
    print("Loading knowledge base...")
    ts = 'http://example:9070/sparql'
    kb = TripleStore(url=ts)

    # load the learning problems
    print("Loading learning problems...")
    with open('lps/Exp II/QALD9WK/QALD9_wk_TandF_MST5.json', 'r') as f:
        lps = json.load(f)


    for str_target_concept, examples in lps['problems'].items():
        pos = frozenset({OWLNamedIndividual(IRI.create(iri)) for iri in examples['positive_examples']})
        neg = frozenset({OWLNamedIndividual(IRI.create(iri)) for iri in examples['negative_examples']})

        print(f"\nLearning problem: {str_target_concept}")
        print(f"Positives: {len(pos)}")
        print(f"Negatives: {len(neg)}")

        # Create refinement operator
        print("\nInitializing refinement operator...")
        operator = PruneCELBasedRefinement(
            knowledge_base=kb,    # knowledge base endpoint
            sparql_endpoint=ts,    # knowledge base endpoint
            length_penalty=0.01,    # length penalty for heuristic
            enable_negation=True,    # negation
            enable_inverse_roles=True,    # inverse roles (I)
            enable_data_properties=True,    # Domain (D)
            enable_qualified_cardinality=True    # Quantifier cardinality restrictions(Q)
        )

        operator.set_input_examples(pos, neg)

        # Configure fragment collection thresholds
        if isinstance(operator, PruneCELBasedRefinement):
            operator.precision_threshold = 1.0    # Precision
            operator.min_positive_coverage = 2    # minimum number of positive examples covered by a fragment
            operator.min_positive_percentage = 0.1


        # Create learner
        learner = BeamSearchLearner(
            kb=kb,    # knowledge base endpoint
            operator=operator,    # Refinement operator
            time_limit=600,    # time in second
            length_penalty=0.01   # length penalty for heuristic
        )

        # Run learning with timing
        lp_start_time = time.time()
        best_concept = learner.learn_recursive(pos=set(pos), neg=set(neg))
        lp_end_time = time.time()
        lp_runtime = lp_end_time - lp_start_time

        print(f"\n{'=' * 80}")
        print(f"Learning problem '{str_target_concept}' completed in {lp_runtime:.2f} seconds")
        print(f"{'=' * 80}")

if __name__ == "__main__":

    #profiler = cProfile.Profile()
    #profiler.enable()

    main()

    #profiler.disable()

    #stats = pstats.Stats(profiler)
    #stats.strip_dirs().sort_stats("cumtime").print_stats(50)
