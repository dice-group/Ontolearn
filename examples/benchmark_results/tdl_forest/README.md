Raw rows of the TDLForest benchmark (`examples/benchmark_tdl_forest.py`), one JSON object per (problem, repeat, fold, config). See issue #672 for the protocol and the results.

| file | configurations | produced by |
|---|---|---|
| `a_base.jsonl.gz` | tree, fs, rules, vote5, forest_tree, forest_vote, tree_t, fs_t, rules_t, forest_vote_t | `run ... --out a.jsonl` |
| `b_bagged_pruning.jsonl.gz` | best_tree, pruned, best_tree_t, pruned_t | `run --configs best_tree,pruned,best_tree_t,pruned_t` |
| `c_full_tree_pruning.jsonl.gz` | full_rules, pruned_full0, pruned_full_t, tree_l, pruned_full_l | `run --configs full_rules,pruned_full0,pruned_full_t,tree_l,pruned_full_l` |
| `d_retrieval_time.jsonl.gz` | tree, tree_t, tree_l, full_rules, pruned_full0, pruned_full_t, pruned_full_l, with `retrieval_time` | `run --workers 8 --configs ...` |

Report: `python examples/benchmark_tdl_forest.py report --out a.jsonl --out b.jsonl --out c.jsonl` (unzip first). `retrieval_time` is in seconds (minimum of 3 warm runs); `length` is the number of characters of the DL rendering.

**Fixed-matrix rerun.** Files a-d predate the fix in #673 (the tautological `∀ r⁻.⊤` column). `e_fixed_matrix_all.jsonl.gz` holds all configurations (with `retrieval_time`) rerun after the fix, with the same seeds and protocol; use it in place of a-d. The conclusions are the same (see the comment on #672).
