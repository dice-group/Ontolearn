import re

import numpy as np
import pandas as pd
import torch
import time
import glob
import math
import json
import copy
import functools
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import F1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def gen_pe(max_length, d_model, n):
  # generate an empty matrix for the positional encodings (pe)
    pe = np.zeros(max_length*d_model).reshape(max_length, d_model) 

    # for each position
    for k in np.arange(max_length):

        # for each dimension
        for i in np.arange(d_model//2):

          # calculate the internal value for sin and cos
            theta = k / (n ** ((2*i)/d_model))       

            # even dims: sin   
            pe[k, 2*i] = math.sin(theta) 

            # odd dims: cos               
            pe[k, 2*i+1] = math.cos(theta)
    return pe


def jaccard_index(actual, retrieved):
    if len(actual.union(retrieved)) == 0:
        print("Both empty")
        return 1
    score = len(actual.intersection(retrieved))/len(actual.union(retrieved))
    return score


def f1_score(actual, retrieved):
    tp = len(actual.intersection(retrieved))
    fn = len(actual.difference(retrieved))
    fp = len(retrieved.difference(actual))
    tn = 0 # Not used
    return F1().score2(tp, fn, fp, tn)[-1]


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
    return positive, negative

def score_all_inds(model, tokenizer, all_ind_embs, exprs, hidden_size, chunk_size=1024):
    outputs = []
    inputs = tokenizer(exprs, padding="max_length", truncation=True, max_length=model.max_length, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    for i in range(0, all_ind_embs.shape[0], chunk_size):
        ind_embs = all_ind_embs[i:i+chunk_size]
        ind_embs_reshaped = ind_embs.reshape(-1, hidden_size).repeat(len(exprs), 1)
        out = model(input_ids, attention_mask, ind_embs_reshaped)
        out = out.detach().cpu().numpy().squeeze()
        out = out.reshape(-1, ind_embs.shape[0])
        outputs.append(out)
    return np.concatenate(outputs, axis=1)
    
def score_all_inds_composite(model, exprs, all_ind_embs, component_embeddings_dict):
    outputs = []
    all_ind_embs_expanded = all_ind_embs.repeat(len(exprs),1,1)
    out = model(exprs, all_ind_embs_expanded, component_embeddings_dict)
    out = out.detach().cpu().numpy().squeeze()
    out = out.reshape(-1, all_ind_embs.shape[0])
    outputs.append(out)
    return np.concatenate(outputs, axis=1)

enable_log = True
def timeit(func):
    """From https://github.com/dice-group/dice-embeddings"""
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        print(f"\nRunning `{repr(func).split(' at ')[0].strip()+'>'}`...")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if enable_log:
            if args is not None:
                s_args = [type(i) for i in args]
            else:
                s_args = args
            if kwargs is not None:
                s_kwargs = {k: type(v) for k, v in kwargs.items()}
            else:
                s_kwargs = kwargs
            print(f'Function {func.__name__} with  Args:{s_args} | Kwargs:{s_kwargs} took {total_time:.4f} seconds')
        else:
            print(f'Took {total_time:.4f} seconds')

        return result

    return timeit_wrapper

def get_short_name(name):
    _name = str(name)
    if "/" in _name:
        return _name.split("/")[-1]
    return _name

@timeit
def read_embs(base_path, merge=False):
    if not merge:
        df = pd.read_csv(glob.glob(f'{base_path}/embeddings/*entity_embeddings.csv')[0], index_col=0).drop_duplicates(subset="0")
        df.index = df.index.map(get_short_name)
        return df
    files = glob.glob(f'{base_path}/embeddings/*entity_embeddings.csv') + glob.glob(f'{base_path}/embeddings/*relation_embeddings.csv')
    dfs = pd.concat([pd.read_csv(file, index_col=0) for file in files], axis=0).drop_duplicates(subset="0")
    dfs.index = dfs.index.map(get_short_name)
    return dfs

@timeit
def read_embs_and_apply_agg(base_path, merge=False, nn_agg=None, device="cpu"):
    if torch.cuda.is_available():
        device = "cuda"
    if nn_agg is not None:
        nn_agg.to(device)
        print(f"\nBuilding atomic concept embeddings with PMA: {nn_agg}\n(A permutation-invariant architecture)\n")
    kb = KnowledgeBase(path=f'{base_path}/kb/ontology.owl') 
    embs = read_embs(base_path, merge)
    embs.index = embs.index.map(get_short_name)
    classes = [c.str.split("/")[-1] for c in kb.ontology.classes_in_signature()]
    classtoinstance = {C.str.split("/")[-1]: [ind.str.split("/")[-1] for ind in kb.individuals(C) if
                                              ind.str.split("/")[-1] in embs.index] for C in
                       kb.ontology.classes_in_signature()}
    emb_copy = embs.copy()
    all_inds = [ind.str.split("/")[-1] for ind in kb.individuals()]
    valid_inds = [ind for ind in all_inds if ind in embs.index]
    for idx in range(len(emb_copy)):
        if emb_copy.index[idx] in classes and len(classtoinstance[emb_copy.index[idx]]) > 0:
            if nn_agg:
                with torch.no_grad():
                    emb_copy.loc[emb_copy.index[idx], :] = nn_agg.encode(torch.FloatTensor(emb_copy.loc[classtoinstance[emb_copy.index[idx]]].values).to(device).unsqueeze(0)).cpu().numpy()
            else:
                emb_copy.loc[emb_copy.index[idx], :] = emb_copy.loc[classtoinstance[emb_copy.index[idx]]].mean(0)
    if nn_agg:
        with torch.no_grad():
            emb_copy.loc['⊤'] = nn_agg.encode(torch.FloatTensor(embs.loc[valid_inds].values).to(device).unsqueeze(0)).cpu().numpy().squeeze()
    else:
        emb_copy.loc['⊤'] = embs.loc[valid_inds].mean(0)
    
    return kb, valid_inds, emb_copy

def read_and_prepare_pma_data(base_path):
        kb = KnowledgeBase(path=f'{base_path}/kb/ontology.owl')
        emb_ent = pd.read_csv(f'{base_path}/embeddings/DeCaL_entity_embeddings.csv',
                              index_col=0)
        emb_rel = pd.read_csv(f'{base_path}/embeddings/DeCaL_relation_embeddings.csv',
                              index_col=0)
        print("Before drop duplicates: ", emb_ent.shape[0] + emb_rel.shape[0], "\n")
        emb = pd.concat([emb_ent, emb_rel], axis=0).drop_duplicates(subset='0')
        print("After drop duplicates: ", emb.shape[0], "\n")
        new_index = emb.index.map(get_short_name)
        print("Example new indices...")
        print(new_index[:3])
        emb.index = new_index
        emb_copy = emb.copy()

        classes = [c.to_string_id().split("/")[-1] for c in kb.ontology.classes_in_signature()]
        classtoinstance = {
            C.str.split("/")[-1]: [ind.str.split("/")[-1] for ind in kb.individuals(C) if
                                   ind.str.split("/")[-1] in new_index] for C in
            kb.ontology.classes_in_signature()}

        valid_inds = set([ind for ind in [ind.str.split("/")[-1] for ind in kb.individuals()] if
                          ind in emb.index])
        class_emb = {}
        data = []
        for c in classes:
            individuals = emb[emb.index.isin(classtoinstance[c])]
            class_emb[c] = {
                'embedding': [individuals.iloc[i].values for i in range(len(individuals))]}
            if len(individuals) > 0:
                data.append(c)
        print(len(data))
        return emb_copy, data, classtoinstance, valid_inds, class_emb, kb
@timeit
def read_training_data(base_path, remove_atomic_concepts=False):
    with open(f"{base_path}/training_data/training_data.json", "r") as f:
        data = []
        for item in json.load(f):
            if isinstance(item, str):
                data.append(item)
            else:
                data.append(item[0])
    if remove_atomic_concepts:
        for i in data:
            if not re.search(r'[⊔.∃∀⊓¬]', i):
                data.remove(i)
    return data

def read_test_data(base_path):
    with open(f"{base_path}/test_data/test_data.json", "r") as f:
        data = []
        for item in json.load(f):
            if isinstance(item, str):
                data.append(item)
            else:
                data.append(item[0])
    return data