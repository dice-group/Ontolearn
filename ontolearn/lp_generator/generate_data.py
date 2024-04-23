import random
from .helper_classes import RDFTriples, KB2Data

random.seed(42)


class LPGen:
    def __init__(self, kb_path, storage_dir=None, max_num_lps=1000, depth=3, max_child_length=20, refinement_expressivity=0.2,
                 downsample_refinements=True, sample_fillers_count=10, num_sub_roots=50, min_num_pos_examples=1):
        """
        Args
        - kb_path: path to the owl file representing the knowledge base/ontology
        - storage_dir: directory in which to store the data to be generated. Not the directory needs not to exists, it would be created automatically
        - max_num_lps: the maximum number of learning problems to store
        - depth, max_child_length, refinement_expressivity, sample_fillers_count, num_sub_roots all refer to the size of the data (learning problems) to be generated
        - downsample_refinements: whether to downsample refinements in ExpressRefinement. If refinement_expressivity<1, this must be set to True
        """
        self.triple_gen = RDFTriples(kb_path=kb_path, storage_dir=storage_dir)
        self.lp_gen = KB2Data(path=kb_path, storage_dir=storage_dir, max_num_lps=max_num_lps, depth=depth,
                      max_child_length=max_child_length, refinement_expressivity=refinement_expressivity,
                      downsample_refinements=downsample_refinements, sample_fillers_count=sample_fillers_count, num_sub_roots=num_sub_roots, min_num_pos_examples=min_num_pos_examples)
    def generate(self):
        self.triple_gen.export_triples()
        self.lp_gen.generate_descriptions().save_data()
