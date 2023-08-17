import torch
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
