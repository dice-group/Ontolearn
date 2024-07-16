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

from torch import nn


class Ensemble(nn.Module):
    """ ensemble through model averaging.
    """

    def __init__(self, modelA, modelB, modelC=None):
        super().__init__()
        self.name = modelA.name + '_' + modelB.name
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        if self.modelC:
            self.name = modelA.name + '_' + modelB.name + '_' + modelC.name

    def forward_head_batch(self, *, e1_idx, rel_idx):
        predictionsA = self.modelA.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx)
        predictionsB = self.modelB.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx)
        if self.modelC:
            predictionsC = self.modelC.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx)
            return (predictionsA + predictionsB + predictionsC) / 3
        else:
            return (predictionsA + predictionsB) / 2
