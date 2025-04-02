import random
import re

import torch
import numpy as np
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser

from cnir.base import BaseDataset
from cnir.utils import sample_examples

class Dataset(BaseDataset):
    def __init__(self, data, all_individuals, concept_to_instance_set, embeddings, pma_mode=False, num_examples=100):
        super().__init__(data, all_individuals, embeddings)
        self.concept_to_instance_set = concept_to_instance_set
        self.pma_mode = pma_mode
        self.num_examples = num_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        expr = self.data[idx]
        dtype = torch.float32
        positives = list(self.concept_to_instance_set[expr])
        negatives = list(self.all_individuals-set(positives))
        
        examples = {"positive examples": positives, "negative examples": negatives}
        pos, neg = sample_examples(examples["positive examples"], examples["negative examples"], self.num_examples)
        try:
            pos_embs = torch.tensor(self.embeddings.loc[pos].values, dtype=dtype)
        except Exception as e:
            # Some individuals do not appear in the embeddings
            new_pos = list(filter(lambda x: x in self.embeddings.index, pos))
            if new_pos:
                pos = new_pos + new_pos[:len(pos)-len(new_pos)]
            else:
                i = 0
                while not new_pos:
                    new_pos, _ = sample_examples(examples["positive examples"], examples["negative examples"], self.num_examples)
                    new_pos = list(filter(lambda x: x in self.embeddings.index, new_pos))
                    i += 1
                    if i > 3:
                        break
                if not new_pos:
                    pos = np.random.choice(list(self.embeddings.index), self.num_examples//2).tolist()
                elif len(new_pos) > len(pos):
                    pos = new_pos[:len(pos)]
                else:
                    pos = new_pos + new_pos[:len(pos)-len(new_pos)]
            pos_embs = torch.tensor(self.embeddings.loc[pos].values, dtype=dtype)
            
        if len(pos) + len(neg) < self.num_examples:
            neg = neg + np.random.choice(neg, self.num_examples-len(pos)-len(neg), replace=True).tolist()
            
        elif len(pos) + len(neg) > self.num_examples:
            neg = neg[:self.num_examples-len(pos)]
            
        try:
            neg_embs = torch.tensor(self.embeddings.loc[neg].values, dtype=dtype)
        except Exception as e:
            # Some individuals do not appear in the embeddings
            new_neg = list(filter(lambda x: x in self.embeddings.index, neg))
            if new_neg:
                neg = new_neg + new_neg[:len(neg)-len(new_neg)]
            else:
                i = 0
                while not new_neg:
                    _, new_neg = sample_examples(examples["positive examples"], examples["negative examples"], self.num_examples)
                    new_neg = list(filter(lambda x: x in self.embeddings.index, new_neg))
                    i += 1
                    if i > 3:
                        break
                if not new_neg:
                    neg = np.random.choice(list(self.embeddings.index), self.num_examples-len(pos)).tolist()
                elif len(new_neg) > len(neg):
                    neg = new_neg[:len(neg)]
                else:
                    neg = new_neg + np.random.choice(new_neg, len(neg)-len(new_neg), replace=True).tolist()
            neg_embs = torch.tensor(self.embeddings.loc[neg].values, dtype=dtype)
        if self.pma_mode:
            rand_idx = list(range(len(pos)))
            random.shuffle(rand_idx)
            selected_pos_idx = rand_idx[:max(1,len(pos)//2)]
            remaining_idx = [idx for idx in rand_idx if idx not in selected_pos_idx]
            if not remaining_idx:
                remaining_idx = [0]
            if len(remaining_idx) + len(neg_embs) < self.num_examples:
                remaining_idx = remaining_idx + np.random.choice(remaining_idx, self.num_examples-len(remaining_idx)-len(neg_embs), replace=True).tolist()
            class_embs = pos_embs[selected_pos_idx]
            ind_embs = torch.cat([pos_embs[remaining_idx], neg_embs], dim=0)
            pos_labels = torch.ones(max(1,len(remaining_idx)), dtype=dtype)
            neg_labels = torch.zeros(neg_embs.size(0), dtype=dtype)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            return expr, class_embs, ind_embs, labels
        else:
            ind_embs = torch.cat([pos_embs, neg_embs], dim=0)
            pos_labels = torch.ones(len(pos_embs), dtype=dtype)
            neg_labels = torch.zeros(neg_embs.size(0), dtype=dtype)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            return expr, ind_embs, labels


class InferenceDataset(BaseDataset):
    def __init__(self, data, all_individuals, concept_to_instance_set, embeddings):
    #def __init__(self, data, all_individuals, concept_to_instance_set, embeddings):
        super().__init__(data, all_individuals, embeddings)
        #super().__init__(data, all_individuals, embeddings)
        self.concept_to_instance_set = concept_to_instance_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        expr = self.data[idx]
        dtype = torch.float32
        if re.search(r'\[(≤|≥)\s-?\d+(\.\d+)?\]', expr):
            constraint = (re.search(r'\[(≤|≥)\s-?\d+(\.\d+)?\]', expr).group())
            expr_modified = expr.replace(constraint, ' ⊤')
            expr_modified = re.sub(r'[ \(\)⊔.∃∀⊓¬⁻≤:\[]+', '|', expr_modified)
        else:
            expr_modified = re.sub(r'[ \(\)⊔.∃∀⊓¬⁻≤:\[]+', '|', expr)
        components = expr_modified.split('|')
        components = [comp for comp in components if comp]

        components = list(set([comp.strip() for comp in components if comp.strip()]))
        # TODO: emb(\bot) = NAND(emb(T), emb(T)) (during training)
        if '⊥' in components:
            components.remove('⊥')
            components.append('⊤')
        elif '{True}' in components:
            components.remove('{True}')
            components.append('⊤')
        elif '{False}' in components:
            components.remove('{False}')
            components.append('⊤')

        component_embeddings_dict = {
            comp:
                torch.tensor(self.embeddings.loc[self.embeddings.index == '⊤'].values,
                             dtype=dtype).squeeze()
                if comp == '⊤'
                else
                torch.tensor(self.embeddings[self.embeddings.index.str.match(
                    rf".*(?<![\-\s+])\b{re.escape(comp)}\b\s*$")].values, dtype=dtype).squeeze()
            for comp in components
        }
        for comp, tensor in component_embeddings_dict.items():
            if tensor.dim() > 1 and tensor.size(0) > 1:
                component_embeddings_dict[comp] = tensor[0]

        return expr,component_embeddings_dict


class CompositeDataset(BaseDataset):
    def __init__(self, data, all_individuals, concept_to_instance_set, embeddings,
                 num_examples=100):
        super().__init__(data, all_individuals, embeddings)
        self.concept_to_instance_set = concept_to_instance_set
        self.num_examples = num_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        expr = self.data[i]
        dtype = torch.float32
        positives = list(self.concept_to_instance_set[expr])
        negatives = list(self.all_individuals - set(positives))
        examples = {"positive examples": positives, "negative examples": negatives}
        pos, neg = sample_examples(examples["positive examples"], examples["negative examples"],
                                   self.num_examples)
        try:
            pos_embs = torch.tensor(self.embeddings.loc[pos].values, dtype=dtype)
        except Exception as e:
            # Some individuals do not appear in the embeddings
            new_pos = list(filter(lambda x: x in self.embeddings.index, pos))
            if new_pos:
                pos = new_pos + new_pos[:len(pos) - len(new_pos)]
            else:
                i = 0
                while not new_pos:
                    new_pos, _ = sample_examples(examples["positive examples"],
                                                 examples["negative examples"], self.num_examples)
                    new_pos = list(filter(lambda x: x in self.embeddings.index, new_pos))
                    i += 1
                    if i > 3:
                        break
                if not new_pos:
                    pos = np.random.choice(list(self.embeddings.index), self.num_examples // 2).tolist()
                elif len(new_pos) > len(pos):
                    pos = new_pos[:len(pos)]
                else:
                    pos = new_pos + new_pos[:len(pos) - len(new_pos)]
            pos_embs = torch.tensor(self.embeddings.loc[pos].values, dtype=dtype)

        if len(pos) + len(neg) < self.num_examples:
            #neg = neg + neg[:self.num_examples - len(pos) - len(neg)]
            neg = neg + np.random.choice(neg, self.num_examples-len(pos)-len(neg), replace=True).tolist()

        elif len(pos) + len(neg) > self.num_examples:
            neg = neg[:self.num_examples - len(pos)]

        try:
            neg_embs = torch.tensor(self.embeddings.loc[neg].values, dtype=dtype)
        except Exception as e:
            # Some individuals do not appear in the embeddings
            new_neg = list(filter(lambda x: x in self.embeddings.index, neg))
            if new_neg:
                neg = new_neg + new_neg[:len(neg) - len(new_neg)]
            else:
                i = 0
                while not new_neg:
                    _, new_neg = sample_examples(examples["positive examples"],
                                                 examples["negative examples"], self.num_examples)
                    new_neg = list(filter(lambda x: x in self.embeddings.index, new_neg))
                    i += 1
                    if i > 3:
                        break
                if not new_neg:
                    neg = np.random.choice(list(self.embeddings.index),
                                           self.num_examples - len(pos)).tolist()
                elif len(new_neg) > len(neg):
                    neg = new_neg[:len(neg)]
                else:
                    neg = new_neg + new_neg[:len(neg) - len(new_neg)]
            neg_embs = torch.tensor(self.embeddings.loc[neg].values, dtype=dtype)

        pos_labels = torch.ones(pos_embs.size(0), dtype=dtype)
        neg_labels = torch.zeros(neg_embs.size(0), dtype=dtype)
        ind_embs = torch.cat([pos_embs, neg_embs], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        if re.search(r'\[(≤|≥)\s-?\d+(\.\d+)?\]', expr):
            constraint = (re.search(r'\[(≤|≥)\s-?\d+(\.\d+)?\]', expr).group())
            expr_modified = expr.replace(constraint, ' ⊤')
            expr_modified = re.sub(r'[ \(\)⊔.∃∀⊓¬⁻≤:\[]+', '|', expr_modified)
        else:
            expr_modified = re.sub(r'[ \(\)⊔.∃∀⊓¬⁻≤:\[]+', '|', expr)
        components = expr_modified.split('|')
        components = [comp for comp in components if comp]

        components = list(set([comp.strip() for comp in components if comp.strip()]))
        # TODO: emb(\bot) = NAND(emb(T), emb(T)) (during training)
        if '⊥' in components:
            components.remove('⊥')
            components.append('⊤')
        elif '{True}' in components:
            components.remove('{True}')
            components.append('⊤')
        elif '{False}' in components:
            components.remove('{False}')
            components.append('⊤')

        component_embeddings_dict = {
            comp:
                torch.tensor(self.embeddings.loc[self.embeddings.index == '⊤'].values,
                             dtype=dtype).squeeze()
                if comp == '⊤'
                else
                torch.tensor(self.embeddings[self.embeddings.index.str.match(
                    rf".*(?<![\-\s+])\b{re.escape(comp)}\b\s*$")].values, dtype=dtype).squeeze()
            for comp in components
        }
        for comp, tensor in component_embeddings_dict.items():
            if tensor.dim() > 1 and tensor.size(0) > 1:
                component_embeddings_dict[comp] = tensor[0]
        return expr, ind_embs, labels, component_embeddings_dict