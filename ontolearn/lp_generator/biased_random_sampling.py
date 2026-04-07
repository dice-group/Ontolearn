import numpy as np

import os
from owlapy.class_expression import OWLThing
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import concept_len
from ontolearn.lp_generator.helper_funcs import generate_concepts, sample_examples, generate_data_description
import json


# Biased Random Search for based Generation
class BiasedRandomLPGenerator:
    def __init__(self, knowledge_base, max_iterations=1000, probability=0.5, sample_ind_size=100,
                 min_example_percent=0.1, seed=42, generate_data_description=True, save_path=None, beyond_alc=False):
        self.kb = knowledge_base
        self.max_itter = max_iterations
        self.probability = probability
        self.sample_ind_size = sample_ind_size
        self.dataset = dict()
        self.root = OWLThing
        self.curr_itter = 0
        self.concept_dict = dict()
        self.concepts = None
        self.all_individuals = set(self.kb.individuals())
        self.beyond_alc = beyond_alc
        if self.beyond_alc:
            self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=10,
                                    sample_fillers_count=5,
                                    downsample=True, use_inverse=True,
                                    use_card_restrictions=True,
                                    use_numeric_datatypes=True, use_time_datatypes=True,
                                    use_boolean_datatype=True,
                                    expressivity=0.8)
        else:
            self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=10,
                                    sample_fillers_count=5,
                                    downsample=True, use_inverse=False,
                                    use_card_restrictions=False,
                                    use_numeric_datatypes=False, use_time_datatypes=False,
                                    use_boolean_datatype=False,
                                    expressivity=0.8)

        self.min_example_percent = min_example_percent
        self.seed = seed
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.generate_data_description = generate_data_description
        self.save_path = save_path

    def get_concepts(self):
        self.concept_dict, self.concepts = generate_concepts(self.kb, self.root, self.refinement_operator,
                                                             self.min_example_percent)
    def generate_lps(self):
        self.get_concepts()
        np.random.seed(self.seed)
        while self.curr_itter < self.max_itter and len(self.concepts) > 0:
            # take a random float r in [0, 1]
            r = np.random.rand()  # faster than random.random() because it is implemented in C
                                  # and optimized for performance.
            if r < self.probability:
                lp = np.random.choice(self.concepts)
                self.concepts.remove(lp)
                raw_pos_set = self.concept_dict[lp]
                raw_neg_set = self.all_individuals - raw_pos_set
                raw_pos = list(raw_pos_set)
                raw_neg = list(raw_neg_set)
                pos, neg = sample_examples(raw_pos, raw_neg, self.sample_ind_size)
                pos = [ind.str.split("/")[-1] for ind in pos]
                neg = [ind.str.split("/")[-1] for ind in neg]
                concept_name = self.dl_syntax_renderer.render(lp.get_nnf())
                self.dataset[concept_name] = {'positive examples': pos, 'negative examples': neg,
                                              'length': concept_len(concept_name)}
            self.curr_itter += 1
        if self.generate_data_description:
            description = generate_data_description(self.dataset)
            print(description)
        if self.save_path is not None:
            dataset = list(self.dataset.items())
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
            with open(self.save_path, 'w', encoding="utf-8") as f:
                json.dump(dataset, f, indent=3, ensure_ascii=False)

#Example usage
if __name__ == "__main__":
    content_root = '/home/psaha/PyCharmMiscProject/Ontolearn/'
    kb_path = 'KGs/Family/family-benchmark_rich_background.owl'
    kb_path = os.path.join(content_root, kb_path)

    kb =  KnowledgeBase(path= kb_path)

    save_path = os.path.join(content_root, 'LPs/Family/generated_lps.json')

    # Initialize the generator
    generator = BiasedRandomLPGenerator(knowledge_base=kb, max_iterations=1000, probability=0.5,
                                        sample_ind_size=100, min_example_percent=0.1, seed=42,
                                        generate_data_description=True, save_path=save_path)

    # Generate learning problems
    generator.generate_lps()