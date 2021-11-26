import sys
import time
from typing import Literal, Iterable, Set, Tuple, Dict, List, Final, Generator

import numpy as np

from owlapy.model import OWLClassExpression, OWLOntologyManager, OWLOntology, AddImport, OWLImportsDeclaration, \
    OWLClass, OWLEquivalentClassesAxiom, IRI, OWLNamedIndividual, OWLAnnotationAssertionAxiom, OWLAnnotation, \
    OWLAnnotationProperty, OWLLiteral
from .knowledge_base import KnowledgeBase
from .refinement_operators import LengthBasedRefinement
from .search import Node, RL_State
from .utils import balanced_sets

SearchAlgos = Literal['dfs', 'strict-dfs']


class LearningProblemGenerator:
    """ Learning problem generator. """

    def __init__(self, knowledge_base: KnowledgeBase, refinement_operator=None, num_problems=10_000, num_diff_runs=100,
                 min_num_instances=None, max_num_instances=sys.maxsize, min_length=3, max_length=5, depth=3,
                 search_algo='strict-dfs'):
        """
        Generate concepts via search algorithm to satisfy constraints.
         strict-dfs considers (min_length, max_length, min_num_ind, num_problems) as hard constraints.
         dfs- considers (min_length, max_length, min_num_ind) as hard constraints and soft (>=num_problems).

         Trade-off between num_diff_runs and num_problems.
        """
        if refinement_operator is None:
            refinement_operator = LengthBasedRefinement(knowledge_base=knowledge_base)

        if max_num_instances and min_num_instances:
            assert max_num_instances >= min_num_instances, f'max_num_instances should be greater or equal than ' \
                                                           f'min_num_instances but ' \
                                                           f'min_num_instances={min_num_instances}, ' \
                                                           f'min_num_instances={min_num_instances}'

        if max_length and min_length:
            assert max_length >= min_length, f'max_length should be greater or equal than min_num_instances ' \
                                             f'but max_length={max_length}, min_length={min_length}'

        self.kb = knowledge_base
        self.rho = refinement_operator
        self.search_algo = search_algo
        self.min_num_instances = min_num_instances
        self.max_num_instances = max_num_instances
        self.min_length = min_length
        self.max_length = max_length
        self.valid_learning_problems = []
        self.depth = depth
        self.num_diff_runs = num_diff_runs
        self.num_problems = num_problems // self.num_diff_runs

    def export_concepts(self, concepts: List[Node], path: str):
        """Serialise the given concepts to a file

        Args:
            concepts: list of Node objects
            path: filename base (extension will be added automatically)
        """
        SNS: Final = 'https://dice-research.org/predictions-schema/'
        NS: Final = 'https://dice-research.org/predictions/' + str(time.time()) + '#'
        # NS: Final = 'https://dice-research.org/problems/' + str(time.time()) + '#'

        from ontolearn import KnowledgeBase
        assert isinstance(self.kb, KnowledgeBase)

        from owlapy.owlready2 import OWLOntologyManager_Owlready2
        manager: OWLOntologyManager = OWLOntologyManager_Owlready2()

        ontology: OWLOntology = manager.create_ontology(IRI.create(NS))
        manager.load_ontology(IRI.create(self.kb.path))
        kb_iri = self.kb.ontology().get_ontology_id().get_ontology_iri()
        manager.apply_change(AddImport(ontology, OWLImportsDeclaration(kb_iri)))
        for ith, h in enumerate(concepts):
            cls_a: OWLClass = OWLClass(IRI.create(NS, "Pred_" + str(ith)))
            equivalent_classes_axiom = OWLEquivalentClassesAxiom(cls_a, h.concept)
            manager.add_axiom(ontology, equivalent_classes_axiom)

            count = None
            try:
                count = h.individuals_count
            except AttributeError:
                if isinstance(h, RL_State):
                    inst = h.instances
                    if inst is not None:
                        count = len(inst)

            if count is not None:
                num_inds = OWLAnnotationAssertionAxiom(cls_a.get_iri(), OWLAnnotation(
                    OWLAnnotationProperty(IRI.create(SNS, "covered_inds")), OWLLiteral(count)))
                manager.add_axiom(ontology, num_inds)

        manager.save_ontology(ontology, IRI.create('file:/' + path + '.owl'))

    def concept_individuals_to_string_balanced_examples(self, concept: OWLClassExpression) -> Dict[str, Set]:

        string_all_pos = set(self.kb.individuals(concept))

        string_all_neg = set(self.kb.ontology().individuals_in_signature()).difference(string_all_pos)

        string_balanced_pos, string_balanced_neg = balanced_sets(set(string_all_pos), set(string_all_neg))
        assert len(string_balanced_pos) >= self.min_num_instances, f"String Representation " \
                                                                   f"of all positive individuals should be greater " \
                                                                   f"than min_num_instances: " \
                                                                   f"|string_all_pos|={len(string_balanced_pos)}" \
                                                                   f"and |min_num_instances| = {self.min_num_instances}"

        assert len(string_balanced_neg) >= self.min_num_instances, f"String Representation " \
                                                                   f"of all positive individuals should be greater " \
                                                                   f"than min_num_instances: " \
                                                                   f"|string_balanced_neg|={len(string_balanced_neg)}" \
                                                                   f"and |min_num_instances| = {self.min_num_instances}"

        return {'string_balanced_pos': string_balanced_pos, 'string_balanced_neg': string_balanced_neg}

    def get_balanced_n_samples_per_examples(self, *, n=5, min_num_problems=None, max_length=None, min_length=None,
                                            num_diff_runs=None, min_num_instances=None,
                                            search_algo='strict-dfs') \
            -> Iterable[Tuple[RL_State, Set[OWLNamedIndividual], Set[OWLNamedIndividual]]]:
        """
        1. We generate min_num_problems number of concepts
        2. For each concept, we generate n number of positive and negative examples
        3. Each example contains
        @param n:
        @param min_num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo:
        @return:
        """
        assert max_length >= min_length

        def class_expression_sanity_checking(x):
            assert len(x.instances), f'Expected {x} has at least one instance {x.instances}'
            assert x.length >= min_length, f'Expected length >= {min_length} but got at least one instance {x.length}'
            assert x.length <= max_length, f'Expected length <= {max_length} but got at least one instance {x.length}'

            # Can we Reach T
            m = x
            while True:
                if m.concept.is_owl_thing() is False:
                    m = m.parent_node
                else:
                    break

        assert self.min_num_instances or min_num_instances
        res = []
        for valid_rl_state in self.generate_examples(num_problems=min_num_problems, max_length=max_length,
                                                     min_length=min_length, num_diff_runs=num_diff_runs,
                                                     min_num_instances=min_num_instances, search_algo=search_algo):
            class_expression_sanity_checking(valid_rl_state)
            for d in self.balanced_n_sampled_lp(n, valid_rl_state.instances):
                assert len(d['string_balanced_pos']) == len(d['string_balanced_neg']), \
                    f' Lengths of examples must match. ' \
                    f'|E^+|={len(d["string_balanced_pos"])} |E^-|={len(d["string_balanced_neg"])}'
                res.append((valid_rl_state, d['string_balanced_pos'], d['string_balanced_neg']))

            if len(res) > min_num_problems:
                break

        assert len(res) > 0, \
            f'No examples *** {len(res)} ***are created. ' \
            f'Please update the configurations for learning problem generator****'

        stats = np.array([[x.length, len(x.instances)] for (x, _, __) in res])
        print(f'\nNumber of generated concepts:{len(res)}')
        print(f'Number of generated learning problems via sampling: {len(res)}')
        print(f'Average length of generated concepts:{stats[:, 0].mean():.3f}\n'
              f'Average number of individuals belong to a generated example:{stats[:, 1].mean():.3f}\n')

        return res

    def balanced_n_sampled_lp(self, n: int, string_all_pos: set):

        string_all_neg = set(self.kb.individuals(self.kb.thing)).difference(string_all_pos)
        for i in range(n):
            string_balanced_pos, string_balanced_neg = balanced_sets(string_all_pos, string_all_neg)
            assert len(string_balanced_pos) >= self.min_num_instances
            assert len(string_balanced_neg) >= self.min_num_instances
            yield {'string_balanced_pos': string_balanced_pos, 'string_balanced_neg': string_balanced_neg}

    def get_balanced_examples(self, *, min_num_problems=None, max_length=None, min_length=None,
                              num_diff_runs=None, min_num_instances=None, search_algo='strict-dfs') -> list:
        """
        (1) Generate valid examples with input search algorithm.
        (2) Balance valid examples.

        @param min_num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as a hard constrain.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.

        """

        def output_sanity_check(y):
            try:
                assert len(y) >= min_num_problems
            except AssertionError:
                print('Not enough number of problems are generated')
                raise

        assert self.min_num_instances or min_num_instances

        res = []
        gen_examples = []
        for example_node in self.generate_examples(num_problems=min_num_problems, max_length=max_length,
                                                   min_length=min_length, num_diff_runs=num_diff_runs,
                                                   min_num_instances=min_num_instances, search_algo=search_algo):
            d = self.concept_individuals_to_string_balanced_examples(example_node.concept)
            res.append((str(example_node), d['string_balanced_pos'], d['string_balanced_neg']))
            gen_examples.append(example_node)

        output_sanity_check(res)
        stats = np.array([[x.length, len(x.instances)] for x in gen_examples])
        print(f'Average length of generated examples {stats[:, 0].mean():.3f}\n'
              f'Average number of individuals belong to a generated example {stats[:, 1].mean():.3f}')
        return res

    def get_examples(self, *, num_problems=None, max_length=None, min_length=None,
                     num_diff_runs=None, min_num_ind=None, search_algo=None) -> list:
        """
        (1) Get valid examples with input search algorithm.

        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_ind:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.

        """
        res = []
        for example_node in self.generate_examples(num_problems=num_problems,
                                                   max_length=max_length, min_length=min_length,
                                                   num_diff_runs=num_diff_runs, min_num_instances=min_num_ind,
                                                   search_algo=search_algo):
            assert len(example_node.concept.instances), f'{example_node}\nDoes not contain any instances. ' \
                                                        f'No instance to balance. Exiting.'
            string_all_pos = set(
                self.kb.convert_owlready2_individuals_to_uri_from_iterable(example_node.concept.instances))
            string_all_neg = set(self.kb.convert_owlready2_individuals_to_uri_from_iterable(
                self.kb.individuals.difference(example_node.concept.instances)))
            res.append((example_node.concept.str, string_all_pos, string_all_neg))
        return res

    def get_concepts(self, *, num_problems=None, max_length=None, min_length=None,
                     max_num_instances=None, num_diff_runs=None, min_num_instances=None, search_algo=None) -> Generator:
        """
        @param max_num_instances:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param num_diff_runs:
        @param min_num_instances:
        @param search_algo: 'dfs' or 'strict-'dfs=> strict-dfs considers num_problems as hard constriant.
        @return: A list of tuples (s,p,n) where s denotes the string representation of a concept,
        p and n denote a set of URIs of individuals indicating positive and negative examples.
        """
        return self.generate_examples(num_problems=num_problems,
                                      max_length=max_length, min_length=min_length,
                                      num_diff_runs=num_diff_runs,
                                      max_num_instances=max_num_instances,
                                      min_num_instances=min_num_instances,
                                      search_algo=search_algo)

    def generate_examples(self, *, num_problems=None, max_length=None, min_length=None,
                          num_diff_runs=None, max_num_instances=None, min_num_instances=None,
                          search_algo=None) -> Generator:
        """
        Generate examples via search algorithm that are valid examples w.r.t. given constraints

        @param num_diff_runs:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param min_num_instances:
        @param max_num_instances:
        @param search_algo:
        @return:
        """
        if num_problems and num_diff_runs:
            assert isinstance(num_problems, int)
            assert isinstance(num_diff_runs, int)
            self.num_problems = num_problems
            self.num_diff_runs = num_diff_runs
            self.num_problems //= self.num_diff_runs
        elif num_diff_runs:
            assert isinstance(num_diff_runs, int)
            self.num_diff_runs = num_diff_runs
            self.num_problems //= self.num_diff_runs
        elif num_problems:
            assert isinstance(num_problems, int)
            self.num_problems = num_problems // self.num_diff_runs

        if self.num_problems == 0:
            self.num_problems += 1

        if max_length:
            assert isinstance(max_length, int)
            self.max_length = max_length
        if min_length:
            assert isinstance(min_length, int)
            self.min_length = min_length
        if min_num_instances:
            assert isinstance(min_num_instances, int)
            self.min_num_instances = min_num_instances
            # Do not generate concepts that do not define enough individuals.
            self.rho.min_num_instances = self.min_num_instances

        if max_num_instances:
            assert isinstance(max_num_instances, int)
            self.max_num_instances = max_num_instances
        if search_algo:
            self.search_algo = search_algo

        if self.min_num_instances and self.max_num_instances == sys.maxsize:
            assert isinstance(self.min_num_instances, int)
            self.max_num_instances = self.kb.individuals_count() - self.min_num_instances

        if self.search_algo == 'dfs':
            valid_concepts = self._apply_dfs()
        elif self.search_algo == 'strict-dfs':
            valid_concepts = self._apply_dfs(strict=True)
        else:
            print(f'Invalid input: search_algo:{search_algo} must be in [dfs,strict-dfs]')
            raise ValueError
        yield from valid_concepts

    def _apply_dfs(self, strict=False) -> Generator:
        """
        Apply depth first search with backtracking to generate concepts.

        @return:
        """

        def f1(x):
            a = self.max_length >= x.length >= self.min_length
            if not a:
                return a
            b = self.max_num_instances >= len(x.instances) >= self.min_num_instances
            return b

        def f2(x):
            return self.max_length >= len(x.length) >= self.min_length

        rl_state = RL_State(self.kb.thing, parent_node=None, is_root=True)
        rl_state.length = self.kb.concept_len(self.kb.thing)
        rl_state.instances = set(self.kb.individuals(rl_state.concept))

        refinements_rl = self.apply_rho_on_rl_state(rl_state)
        if self.min_num_instances:
            constrain_func = f1
        else:
            constrain_func = f2

        valid_states_gate = set()
        while True:
            try:
                rl_state = next(refinements_rl)
            except StopIteration:
                print('All top concepts are refined.')
                break

            if rl_state.concept.is_owl_nothing():
                continue

            if constrain_func(rl_state):
                valid_states_gate.add(rl_state)
                yield rl_state
                if strict:
                    if len(valid_states_gate) >= self.num_problems * self.num_diff_runs:
                        print(f'|Valid Expressions|={len(valid_states_gate)}')
                        break

            temp_gate = set()
            for ref_rl_state in self._apply_dfs_on_state(state=rl_state,
                                                         apply_rho=self.apply_rho_on_rl_state,
                                                         constrain_func=constrain_func,
                                                         depth=self.depth,
                                                         patience_per_depth=(self.num_problems // 2)):
                if ref_rl_state not in valid_states_gate:
                    valid_states_gate.add(ref_rl_state)
                    temp_gate.add(ref_rl_state)
                    yield ref_rl_state
                    if strict:
                        if len(temp_gate) >= self.num_problems or (
                                len(valid_states_gate) >= self.num_problems * self.num_diff_runs):
                            break
            if strict:
                if len(valid_states_gate) >= self.num_problems * self.num_diff_runs:
                    break
        # sanity checking after the search.
        try:
            assert len(valid_states_gate) >= self.num_diff_runs * self.num_problems
        except AssertionError:
            print(f'Number of valid concepts generated:{len(valid_states_gate)}.\n'
                  f'Required number of concepts: {self.num_diff_runs * self.num_problems}.\n'
                  f'Please update the given constraints:'
                  f'Increase the max length (Currently {self.max_length}).\n'
                  f'Increase the max number of instances for concepts (Currently {self.max_num_instances}).\n'
                  f'Decrease the min length (Currently {self.min_length}).\n'
                  f'Decrease the max number of instances for concepts (Currently {self.min_num_instances}).\n')

    @staticmethod
    # @performance_debugger('_apply_dfs_on_state')
    def _apply_dfs_on_state(state, depth, apply_rho, constrain_func=None, patience_per_depth=None) -> set:
        """

        @param state:
        @param depth:
        @param apply_rho:
        @param num_problems:
        @param max_length:
        @param min_length:
        @param min_num_ind:
        @param max_num_ind:
        @return:
        """

        valid_examples = set()
        invalid_examples = set()

        for _ in range(depth):
            """ (1) Iterate over refinements of input state """
            for ref in apply_rho(state):
                """ (1.1.1) Check whether a refinement is a valid example """
                if constrain_func(ref):
                    """ (1.1.2) Remember a valid refinement """
                    valid_examples.add(ref)
                    """ (1.1.3) Yield a valid refinement """
                    yield ref
                else:
                    """ (1.1.2) Remember invalid refinement """
                    invalid_examples.add(ref)

            """ (2) Select next state to be refined from valid examples """
            if len(valid_examples) > 0:
                try:
                    state = next(iter(valid_examples))
                    valid_examples.discard(state)
                except StopIteration:
                    pass
                continue

            if len(invalid_examples) > 0:
                try:
                    state = next(iter(valid_examples))
                    invalid_examples.discard(state)
                except StopIteration:
                    pass
                continue

            print('Constraints are not fulfilled')
            # raise ValueError('We could not find ')

    def apply_rho_on_rl_state(self, rl_state):
        for i in self.rho.refine(rl_state.concept):
            next_rl_state = RL_State(i, parent_node=rl_state)
            next_rl_state.length = self.kb.concept_len(next_rl_state.concept)
            next_rl_state.instances = set(self.kb.individuals(next_rl_state.concept))
            yield next_rl_state
