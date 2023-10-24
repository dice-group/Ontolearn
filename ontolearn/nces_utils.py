"""NCES utils."""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleSolution:

    def __init__(self, vocab, atomic_concept_names):
        self.name = 'SimpleSolution'
        self.atomic_concept_names = atomic_concept_names
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
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
