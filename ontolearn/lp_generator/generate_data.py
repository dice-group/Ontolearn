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

import random
from .helper_classes import KB2Data

random.seed(42)


class LPGen:
    def __init__(self, kb_path, storage_path=None, max_num_lps=1000, beyond_alc=False, depth=3, max_child_length=20, refinement_expressivity=0.2,
                 downsample_refinements=True, sample_fillers_count=10, num_sub_roots=50, min_num_pos_examples=1):
        """
        Args
        - kb_path: path to the owl file representing the knowledge base/ontology
        - storage_path: directory in which to store the data to be generated. Not the directory needs not to exists, it would be created automatically
        - max_num_lps: the maximum number of learning problems to store
        - beyond_alc: whether to generate learning problems in ALCHIQD
        - depth, max_child_length, refinement_expressivity, sample_fillers_count, num_sub_roots all refer to the size of the data (learning problems) to be generated
        - downsample_refinements: whether to downsample refinements in ExpressRefinement. If refinement_expressivity<1, this must be set to True
        """
        self.lp_gen = KB2Data(path=kb_path, storage_path=storage_path, max_num_lps=max_num_lps, beyond_alc=beyond_alc, depth=depth,
                      max_child_length=max_child_length, refinement_expressivity=refinement_expressivity,
                      downsample_refinements=downsample_refinements, sample_fillers_count=sample_fillers_count, num_sub_roots=num_sub_roots, min_num_pos_examples=min_num_pos_examples)
    def generate(self):
        self.lp_gen.generate_descriptions().save_data()
