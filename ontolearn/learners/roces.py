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

"""ROCES: Robust Class Expression Synthesis in Description Logics via Iterative Sampling."""

from typing import Optional

from ontolearn.abstracts import AbstractScorer
from ontolearn.learners.nces2 import NCES2


class ROCES(NCES2):
    """Robust Class Expression Synthesis in Description Logics via Iterative Sampling."""
    name = "ROCES"

    def __init__(self, knowledge_base, nces2_or_roces=True,
                 quality_func: Optional[AbstractScorer] = None, num_predictions=5, k=5,
                 path_of_trained_models=None, auto_train=True, proj_dim=128, rnn_n_layers=2, drop_prob=0.1,
                 num_heads=4, num_seeds=1, m=[32, 64, 128], ln=False, embedding_dim=128, sampling_strategy="p",
                 input_dropout=0.0, feature_map_dropout=0.1, kernel_size=4, num_of_output_channels=32,
                 learning_rate=1e-4, tmax=20, eta_min=1e-5, clip_value=5.0, batch_size=256, num_workers=4,
                 max_length=48, load_pretrained=True, verbose: int = 0, data=[], enforce_validity:Optional[bool]=None):
        self.k = k
        super().__init__(knowledge_base, nces2_or_roces,
                         quality_func, num_predictions, path_of_trained_models, auto_train, proj_dim, drop_prob,
                         num_heads, num_seeds, m, ln, embedding_dim, sampling_strategy, input_dropout,
                         feature_map_dropout, kernel_size, num_of_output_channels, learning_rate, tmax, eta_min,
                         clip_value, batch_size, num_workers, max_length, load_pretrained, verbose, enforce_validity)
        self.enforce_validity = enforce_validity
