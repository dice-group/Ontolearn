"""
visualize_vnet_tree.py
======================
Load the recorded beam-search tree for a given LP (e.g. Aunt),
score every node with the trained V-Net (predicted best-reachable F1),
and render a pruned tree with graphviz.

Each node box shows:
  - Concept string (shortened)
  - actual F1      (what beam search measured)
  - V-Net pred     (predicted best-reachable F1 from this node)
  - GT             (ground-truth best-reachable F1 via bottom-up DP)

The tree is pruned to TOP_K children per node (highest V-Net score).

Metadata (scores + tree structure) is saved to JSON so the plot can be
regenerated instantly without reloading the model.

Usage:
  # First run — scores nodes and saves metadata:
  conda run -n owlapy311 python visualize_vnet_tree.py \\
      --lp Aunt --top_k 4 --max_depth 4

  # Re-plot from saved metadata (no model needed):
  conda run -n owlapy311 python visualize_vnet_tree.py \\
      --lp Aunt --top_k 4 --max_depth 4 --load_meta
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
import graphviz

sys.path.insert(0, os.path.dirname(__file__))
from train_vocell_v_net import (
    ConceptVNet,
    get_mean_emb,
    compute_best_reachable_from_nodes,
)

# ── Defaults ────────────────────────────────────────────────────────────────
DATASET_FILE = 'vnet_search_data_difficult.json'
CHECKPOINT   = 'Family_mean/vocell_v_net_bootstrap_mean.pt'
EMBEDDINGS   = 'Experiments/embeddings/Keci_entity_embeddings.csv'
OUT_DIR      = 'results_loss_curve'


# ── Model helpers ────────────────────────────────────────────────────────────

def load_vnet(checkpoint_path: str, device: str = 'cpu') -> ConceptVNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    dim  = ckpt['embedding_dim']
    net  = ConceptVNet(dim, device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net


def score_nodes(nodes, pos_iris, neg_iris, df, net, device='cpu'):
    """Return dict: concept_str -> V-Net predicted best-reachable F1."""
    emb_pos = get_mean_emb(pos_iris, df).to(device)
    emb_neg = get_mean_emb(neg_iris, df).to(device)
    preds = {}
    with torch.no_grad():
        for node in nodes:
            iris  = node.get('instance_iris') or []
            emb_c = get_mean_emb(iris, df).to(device)
            x     = torch.cat([emb_c, emb_pos, emb_neg], dim=1)   # (1,3,dim)
            preds[node['concept_str']] = net(x).item()
    return preds


# ── Metadata save / load ─────────────────────────────────────────────────────

def meta_path(out_dir: str, lp_name: str) -> str:
    return os.path.join(out_dir, f'vnet_tree_meta_{lp_name}.json')


def save_meta(out_dir, lp_name, nodes, vnet_preds, dp_best):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'lp_name':    lp_name,
        'nodes':      nodes,          # full node dicts
        'vnet_preds': vnet_preds,     # concept_str -> float
        'dp_best':    dp_best,        # concept_str -> float
    }
    path = meta_path(out_dir, lp_name)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Metadata saved → {path}')


def load_meta(out_dir, lp_name):
    path = meta_path(out_dir, lp_name)
    with open(path) as f:
        d = json.load(f)
    print(f'Metadata loaded ← {path}')
    return d['nodes'], d['vnet_preds'], d['dp_best']


# ── Graphviz rendering ───────────────────────────────────────────────────────

def shorten(s: str, max_len: int = 36) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + '…'


def pred_color(p: float) -> str:
    """Interpolate dark green #0b7d0b (low) → #aad34b (high) for p in [0,1]."""
    # low:  #0b7d0b  (r=11,  g=125, b=11)
    # high: #aad34b  (r=170, g=211, b=75)
    r = int(11  + (170 - 11)  * p)
    g = int(125 + (211 - 125) * p)
    b = int(11  + (75  - 11)  * p)
    return f'#{r:02x}{g:02x}{b:02x}'


def render_tree(nodes, vnet_preds, dp_best, top_k, max_depth, lp_name, out_dir,
                exclude=None):
    """
    Render the pruned beam-search tree.

    At each node keep only the TOP_K children with the highest V-Net score.
    Traversal is BFS from root down to max_depth.
    Nodes whose concept_str contains any substring in `exclude` are hidden.
    """
    from collections import defaultdict

    exclude = exclude or []

    def is_excluded(cstr):
        return any(pat in cstr for pat in exclude)

    # Build full children map
    children_map: dict = defaultdict(list)
    for n in nodes:
        p = n.get('parent_str')
        if p is not None:
            children_map[p].append(n)

    node_map = {n['concept_str']: n for n in nodes}
    root     = next(n for n in nodes if n.get('parent_str') is None)

    # BFS: keep top_k children per node by V-Net score, skipping excluded nodes
    visible = {root['concept_str']}
    queue   = [root]
    while queue:
        cur = queue.pop(0)
        if cur['depth'] >= max_depth:
            continue
        kids = children_map.get(cur['concept_str'], [])
        kids_sorted = sorted(
            kids,
            key=lambda n: vnet_preds.get(n['concept_str'], 0.0),
            reverse=True,
        )
        added = 0
        for child in kids_sorted:
            if added >= top_k:
                break
            if is_excluded(child['concept_str']):
                continue
            visible.add(child['concept_str'])
            queue.append(child)
            added += 1

    # Also remove excluded nodes that made it in via root
    visible = {c for c in visible if not is_excluded(c)}

    # ── Build Digraph ────────────────────────────────────────────────────────
    dot = graphviz.Digraph(
        name=f'VNet Tree — {lp_name}',
        format='pdf',
        graph_attr={
            'rankdir':  'TB',
            'fontname': 'Helvetica',
            'fontsize': '18',
            'nodesep':  '0.6',
            'ranksep':  '0.8',
            'splines':  'ortho',
        },
        node_attr={
            'shape':    'box',
            'style':    'filled,rounded',
            'fontname': 'Helvetica',
            'fontsize': '20',
        },
        edge_attr={'arrowsize': '0.8'},
    )

    for cstr in visible:
        n     = node_map[cstr]
        pred  = vnet_preds.get(cstr, 0.0)
        gt    = dp_best.get(cstr, n['f1'])
        color = pred_color(pred)
        label = (
            f"{shorten(cstr)}\n"
            f"F1={n['f1']:.3f}  │  V-Net={pred:.3f}\n"
            f"GT={gt:.3f}"
        )
        dot.node(cstr, label=label, fillcolor=color, fontcolor='black')

    for n in nodes:
        if n['concept_str'] not in visible:
            continue
        p = n.get('parent_str')
        if p and p in visible:
            dot.edge(p, n['concept_str'])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'vnet_tree_{lp_name}_k{top_k}_d{max_depth}')
    dot.render(out_path, cleanup=True)
    print(f'Tree saved → {out_path}.pdf')
    return out_path + '.pdf'


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lp',         default='Aunt',
                        help='LP name in the dataset JSON.')
    parser.add_argument('--dataset',    default=DATASET_FILE)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--embeddings', default=EMBEDDINGS)
    parser.add_argument('--top_k',      type=int, default=4,
                        help='Max children kept per node (by V-Net score).')
    parser.add_argument('--max_depth',  type=int, default=4,
                        help='Max tree depth to render.')
    parser.add_argument('--out_dir',    default=OUT_DIR)
    parser.add_argument('--device',     default='cpu')
    parser.add_argument('--load_meta',  action='store_true',
                        help='Skip model inference; load saved metadata JSON instead.')
    parser.add_argument('--exclude', nargs='+', default=[],
                        metavar='SUBSTR',
                        help='Hide nodes whose concept string contains any of '
                             'these substrings (space-separated).')
    args = parser.parse_args()

    print(f'LP: {args.lp}')

    if args.load_meta:
        # ── Fast path: load pre-computed scores ──────────────────────────────
        nodes, vnet_preds, dp_best = load_meta(args.out_dir, args.lp)
    else:
        # ── Full path: score nodes with V-Net ────────────────────────────────
        print(f'Checkpoint: {args.checkpoint}')

        with open(args.dataset) as f:
            dataset = json.load(f)

        if args.lp not in dataset:
            print(f'ERROR: LP "{args.lp}" not found. '
                  f'Available: {list(dataset.keys())}')
            sys.exit(1)

        lp_data  = dataset[args.lp]
        nodes    = lp_data['nodes']
        pos_iris = lp_data['positive_examples']
        neg_iris = lp_data['negative_examples']

        print(f'Nodes in tree : {len(nodes)}')
        print(f'Pos / Neg     : {len(pos_iris)} / {len(neg_iris)}')

        df  = pd.read_csv(args.embeddings, index_col=0).astype('float32')
        net = load_vnet(args.checkpoint, args.device)

        print('Scoring nodes with V-Net…')
        vnet_preds = score_nodes(nodes, pos_iris, neg_iris, df, net, args.device)
        dp_best    = compute_best_reachable_from_nodes(nodes)

        save_meta(args.out_dir, args.lp, nodes, vnet_preds, dp_best)

    # ── Text summary ─────────────────────────────────────────────────────────
    node_map = {n['concept_str']: n for n in nodes}
    print(f'\n{"Concept":<55} {"F1":>6} {"V-Net":>6} {"GT":>6}')
    print('-' * 78)
    for n in sorted(nodes, key=lambda x: (x['depth'],
                                          -vnet_preds.get(x['concept_str'], 0))):
        if n['depth'] > args.max_depth:
            continue
        c = n['concept_str']
        print(f'{c:<55} {n["f1"]:>6.3f} '
              f'{vnet_preds.get(c, 0):>6.3f} {dp_best.get(c, 0):>6.3f}')

    render_tree(nodes, vnet_preds, dp_best,
                args.top_k, args.max_depth, args.lp, args.out_dir,
                exclude=args.exclude)


if __name__ == '__main__':
    main()
