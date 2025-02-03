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

"""NCES utils."""
import os
import random
import numpy as np
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from ontolearn.lp_generator import LPGen


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleSolution:

    def __init__(self, vocab, atomic_concept_names):
        self.name = 'SimpleSolution'
        self.atomic_concept_names = atomic_concept_names
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],show_progress=False)
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.train_from_iterator(vocab, trainer)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.tokenizer.pad_token = "[PAD]"

    def predict(self, expression: str):
        atomic_classes = [atm for atm in self.tokenizer.tokenize(expression) if atm in self.atomic_concept_names]
        if atomic_classes == []:
            # If no atomic class found, then randomly pick and use the first 3
            random.shuffle(self.atomic_concept_names)
            atomic_classes = self.atomic_concept_names[:3]
        return " ⊔ ".join(atomic_classes)


def sample_examples(pos, neg, num_ex):
    if min(len(pos), len(neg)) >= num_ex // 2:
        if len(pos) > len(neg):
            num_neg_ex = num_ex // 2
            num_pos_ex = num_ex - num_neg_ex
        else:
            num_pos_ex = num_ex // 2
            num_neg_ex = num_ex - num_pos_ex
    elif len(pos) + len(neg) >= num_ex and len(pos) > len(neg):
        num_neg_ex = len(neg)
        num_pos_ex = num_ex - num_neg_ex
    elif len(pos) + len(neg) >= num_ex and len(pos) < len(neg):
        num_pos_ex = len(pos)
        num_neg_ex = num_ex - num_pos_ex
    else:
        num_pos_ex = len(pos)
        num_neg_ex = len(neg)
    positive = np.random.choice(pos, size=min(num_pos_ex, len(pos)), replace=False)
    negative = np.random.choice(neg, size=min(num_neg_ex, len(neg)), replace=False)
    return positive.tolist(), negative.tolist()


def try_get_embs(pos, neg, embeddings, num_examples):
    """
    Depending on the KGE model, some individuals do not get assigned to any embedding during training. This function filters out such individuals from the provided positive/negative examples. It also
    """
    try:
        _ = embeddings.loc[pos]
    except Exception as e:
        # Some individuals do not appear in the embeddings
        new_pos = list(filter(lambda x: x in embeddings.index, pos))
        if new_pos and len(new_pos) >= len(pos)-len(new_pos):
            pos = new_pos + new_pos[:len(pos)-len(new_pos)]
        else:
            i = 0
            while not new_pos:
                new_pos, _ = sample_examples(pos, neg, num_examples)
                new_pos = list(filter(lambda x: x in embeddings.index, new_pos))
                i += 1
                if i > 3:
                    break
            if not new_pos:
                pos = np.random.choice(list(embeddings.index), num_examples//2).tolist()
                #if contains_prefix:
                #    pos = list(map(lambda x: x.split("/")[-1], pos))
            elif len(new_pos) > len(pos):
                pos = new_pos[:len(pos)]
            else:
                pos = new_pos + new_pos[:len(pos)-len(new_pos)]
        
    if len(pos) + len(neg) < num_examples:
        neg = neg + neg[:num_examples-len(pos)-len(neg)]
        
    elif len(pos) + len(neg) > num_examples:
        neg = neg[:num_examples-len(pos)]
        
    try:
        _ = embeddings.loc[neg]
    except Exception as e:
        # Some individuals do not appear in the embeddings
        new_neg = list(filter(lambda x: x in embeddings.index, neg))
        if new_neg and len(new_neg) >= len(neg)-len(new_neg):
            neg = new_neg + new_neg[:len(neg)-len(new_neg)]
        else:
            i = 0
            while not new_neg:
                _, new_neg = sample_examples(pos, neg, num_examples)
                new_neg = list(filter(lambda x: x in embeddings.index, new_neg))
                i += 1
                if i > 3:
                    break
            if not new_neg:
                neg = np.random.choice(list(embeddings.index), num_examples-len(pos)).tolist()
            elif len(new_neg) > len(neg):
                neg = new_neg[:len(neg)]
            else:
                neg = new_neg + new_neg[:len(neg)-len(new_neg)]

    return pos, neg


def generate_training_data(kb_path, max_num_lps=1000, refinement_expressivity=0.2, refs_sample_size=50,
                           beyond_alc=True, storage_path=None):
    if storage_path is None:
        storage_path = "./Training_Data"
    lp_gen = LPGen(kb_path=kb_path, max_num_lps=max_num_lps, refinement_expressivity=refinement_expressivity,
                   num_sub_roots=refs_sample_size,
                   beyond_alc=beyond_alc, storage_path=storage_path)
    lp_gen.generate()
    print("Loading generated data...")
    with open(f"{storage_path}/LPs.json") as file:
        lps = json.load(file)
        if isinstance(lps, dict):
            lps = list(lps.items())
        print("Number of learning problems:", len(lps))
    return lps
