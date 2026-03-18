"""
Covering-based Concept Learning Algorithm with Fragment Collection

The refinement operator automatically collects high-precision, low-recall fragments
during refinement. This script just runs beam search and retrieves the collected
fragments at the end.
"""
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
                 time_limit: float = 300.0):
        """
        Initialize the learner.
        
        Args:
            kb: Knowledge base
            operator: Refinement operator (collects fragments automatically)
            beam_width: Number of top concepts to refine each iteration (default: 10)
            max_depth: Maximum refinement depth (default: 5)
            time_limit: Time limit in seconds (default: 300)
        """
        self.kb = kb
        self.operator = operator
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.time_limit = time_limit
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
    
    def beam_search(self, pos: Set, neg: Set, start) -> Optional[OWLClassExpression]:
        """
        Beam search for best concept.
        
        The refinement operator automatically collects high-precision fragments
        during the search.
        
        Returns:
            Best F1 concept found
        """

        
        # Initialize beam with ⊤
        beam = [OWLThing]
        # beam = [OWLClass(IRI.create('http://www.benchmark.org/family#Female'))]
        
        best_concept = None
        best_f1 = 0.0
        
        visited = set()
        visited.add(str(OWLThing))
        
        print(f"\nStarting beam search (beam width={self.beam_width}, max depth={self.max_depth})")
        print("="*80)
        
        for depth in range(self.max_depth):

            if (time.time() - start) >= self.time_limit:
                elapsed = time.time() - start
                print(f"\nTime reached in {elapsed:.1f}s")
                print(f"Total refinements explored: {self.total_refinements_explored}")
                break
            
            print(f"\nDepth {depth}: Refining top {len(beam)} concepts...")
            
            # Generate all refinements of current beam
            all_refinements = []
            
            for concept in beam:
                try:
                    for child, f1, _, precision, recall, tp in self.operator.refine(concept):
                    # for child in self.operator.refine(concept):

                        # print(self.renderer.render(child))
                        self.total_refinements_explored += 1
                        
                        # Skip if already visited
                        child_str = str(child)
                        if child_str in visited:
                            continue
                        visited.add(child_str)
                        
                        # # Evaluate child
                        # f1, precision, recall, covered_pos = self.evaluate(child, pos, neg)
                        # tp = len(covered_pos)

                        # Check if necessary to start recursive
                        if precision >= self.operator.precision_threshold and recall <= self.operator.recall_threshold:

                            print("\n" + "-" * 70)
                            print("🔹 High-Precision / Low-Recall Fragment Found")
                            print("-" * 70)

                            print(f"Concept: {self.renderer.render(child)}")
                            print(f"Depth: {depth}")
                            print(f"F1: {f1:.3f}")
                            print(f"Precision: {precision:.3f}")
                            print(f"Recall: {recall:.3f}")
                            print(f"Covers: {tp}/{len(pos)} positives")
                            print(f"Explored refinements so far: {self.total_refinements_explored}")
                            print("-" * 70 + "\n")

                            return child
                        
                        all_refinements.append((child, f1, precision, recall, tp))

                        # Check for perfect F1 - stop immediately!
                        if f1 == 1.0:
                            print(f"\n PERFECT F1 FOUND!")
                            print(f"  {self.renderer.render(child)}")
                            print(f"  F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                            return child
                        
                        # Update global best
                        if f1 > best_f1:
                            best_concept = child
                            best_f1 = f1
                
                except Exception as e:
                    print(f"  Exception: {e}")
                    break

            if not all_refinements:
                print(f"  No new refinements generated")
                break
            
            # Sort by F1 and keep top beam_width
            all_refinements.sort(key=lambda x: x[1], reverse=True)
            
            # Print top concepts at this depth
            print(f"  Generated {len(all_refinements)} unique refinements")
            print(f"  Top 5:")
            for i, (concept, f1, prec, rec, _) in enumerate(all_refinements[:5], 1):
                concept_str = self.renderer.render(concept)
                if len(concept_str) > 70:
                    concept_str = concept_str[:70] + "..."
                print(f"    {i}. F1={f1:.3f} P={prec:.3f} R={rec:.3f} | {concept_str}")
            
            # Update beam with top concepts
            beam = [concept for concept, _, _, _, _ in all_refinements[:self.beam_width]]
            
            print(f"  Best F1 so far: {best_f1:.3f}")
        
        elapsed = time.time() - start
        print(f"\nSearch completed in {elapsed:.1f}s")
        print(f"Total refinements explored: {self.total_refinements_explored}")
        
        return best_concept


    def learn_recursive(self, pos, neg):

        # recursive learning

        remaining_pos = set(pos)
        fragments = []

        start = time.time()
        while remaining_pos:
            
            self.operator.set_input_examples(frozenset(remaining_pos), frozenset(neg))

            if (time.time() - start) >= self.time_limit:
                print(f"\n⏱ Time limit reached!")
                break

            fragment = self.beam_search(
                remaining_pos,
                neg,
                start
            )

            f1, precision, recall, covered = self.evaluate(
                fragment,
                remaining_pos,
                neg
            )

            fragments.append(fragment)
            remaining_pos -= covered

        if len(fragments) > 1:
            final_concepts = OWLObjectUnionOf(fragments)
            f1, precision, recall, _ = self.evaluate(final_concepts, pos, neg)
            print(f"\n{'=' * 80}")
            print(f"FINAL RESULT")
            print(f"{'=' * 80}")
            print(f"{self.renderer.render(final_concepts)}")
            print(f"\nF1: {f1:.3f}")
            print(f"{'=' * 80}")
            return OWLObjectUnionOf(fragments)

        else:
            f1, precision, recall, _ = self.evaluate(fragments[0], pos, neg)
            print(f"\n{'=' * 80}")
            print(f"FINAL RESULT")
            print(f"{'=' * 80}")
            print(f"{self.renderer.render(fragments[0])}")
            print(f"\nF1: {f1:.3f}")
            print(f"{'=' * 80}")
            return fragments[0]



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

    # Configure fragment collection thresholds
    if isinstance(operator, PruneCELBasedRefinement):
        operator.precision_threshold = 1.
        operator.recall_threshold = 0.6

    # Create learner
    learner = BeamSearchLearner(
        kb=kb,
        operator=operator,
        beam_width=5,
        max_depth=15,
        time_limit=60.0
    )
    
    # Run learning
    best_concept = learner.learn_recursive(pos=set(pos), neg=set(neg))
    
    print(f"\n✓ Learning complete!")
    # print(f"Best concept found: {learner.renderer.render(best_concept)}")


if __name__ == "__main__":
    main()