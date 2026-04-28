import numpy as np

import os
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import concept_len
from ontolearn.lp_generator.helper_funcs import sample_examples, generate_data_description, \
    generate_concepts_parallel
import json


class BiasedRandomLPGenerator:
    def __init__(self, knowledge_base, max_iterations=1000, sample_ind_size=100,
                 max_example_percent=0.1, seed=42, generate_data_description=True, save_path=None, beyond_alc=False,
                 max_child_length=10, max_length=10, refinement_expressivity=0.2, second_level_refinement=False,
                 num_sub_roots=50, max_data_len=1000):
        self.kb = knowledge_base
        self.max_itter = max_iterations
        self.sample_ind_size = sample_ind_size
        self.dataset = dict()
        self.root = self.kb.generator.thing
        self.curr_itter = 0
        self.concept_dict = dict()
        self.concepts = None
        self.all_individuals = set(self.kb.individuals())
        self.all_individuals_str = set(i.str for i in self.all_individuals)

        self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length,
                                                         sample_fillers_count=5,
                                                         downsample=True, use_inverse=beyond_alc,
                                                         use_card_restrictions=beyond_alc,
                                                         use_numeric_datatypes=beyond_alc, use_time_datatypes=beyond_alc,
                                                         use_boolean_datatype=beyond_alc,
                                                         expressivity=refinement_expressivity)


        self.max_example_percent = max_example_percent
        self.seed = seed
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.generate_data_description = generate_data_description
        self.save_path = save_path
        self.max_length = max_length
        self.max_data_len = max_data_len
        self.num_sub_roots = num_sub_roots
        self.second_level_refinement = second_level_refinement
        self.curr_data_len = 0

    def get_concepts(self):
        self.concept_dict, self.concepts = generate_concepts_parallel(self.kb, self.root, self.refinement_operator,
                                                                      self.max_example_percent, self.max_length,
                                                                      self.num_sub_roots, self.second_level_refinement)

    def generate_lps(self):
        self.get_concepts()
        np.random.seed(self.seed)
        while self.curr_itter < self.max_itter and len(self.concepts) > 0:
            # take a random float r in [0, 1]
            lp = np.random.choice(self.concepts)
            r = np.random.rand()  # faster than random.random() because it is implemented in C
            # and optimized for performance.
            if r > 0.5:

                self.concepts.remove(lp)
                raw_pos_set = set(i.str for i in self.concept_dict[lp])
                raw_neg_set = self.all_individuals_str - raw_pos_set
                raw_pos = list(raw_pos_set)
                raw_neg = list(raw_neg_set)
                pos, neg = sample_examples(raw_pos, raw_neg, self.sample_ind_size)
                pos = [ind.split("/")[-1] for ind in pos]
                neg = [ind.split("/")[-1] for ind in neg]
                concept_name = self.dl_syntax_renderer.render(lp.get_nnf())
                self.dataset[concept_name] = {'positive examples': pos, 'negative examples': neg,
                                              'length': concept_len(lp)}
                self.curr_data_len = len(self.dataset)
                if self.curr_data_len >= self.max_data_len:
                    print(f"Reached maximum data length of {self.max_data_len}. Stopping generation.")
                    break

            self.curr_itter += 1
        if self.generate_data_description:
            description = generate_data_description(self.dataset)
            print(description)
        if self.save_path is not None:
            dataset = list(self.dataset.items())

            os.makedirs(self.save_path, exist_ok=True)  # Creates dir and ignores if it exists

            save_path_lp = os.path.join(self.save_path, 'generated_lps.json')

            with open(save_path_lp, 'w', encoding="utf-8") as f:
                json.dump(dataset, f, indent=3, ensure_ascii=False)


# Example usage
if __name__ == "__main__":
    content_root = '/home/psaha/PyCharmMiscProject/Ontolearn/'
    kb_path = 'KGs/Family/family-benchmark_rich_background.owl'
    kb_path = os.path.join(content_root, kb_path)

    kb = KnowledgeBase(path=kb_path)

    save_path = os.path.join(content_root, 'LPs/Family/generated_lps.json')

    # Initialize the generator
    generator = BiasedRandomLPGenerator(knowledge_base=kb, max_iterations=1000, probability=0.5,
                                        sample_ind_size=100, min_example_percent=0.1, seed=42,
                                        generate_data_description=True, save_path=save_path)

    # Generate learning problems
    generator.generate_lps()
