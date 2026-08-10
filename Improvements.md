# Ontolearn — Suggested Improvements

Findings from a code-quality/architecture pass over `ontolearn/` (core reasoning
modules, the `learners/` package) and the surrounding infra (tests, packaging,
lint, docs). Each item cites the file/line where the issue was observed.
Prioritized roughly high → low impact within each section.

## 1. Correctness bugs

These are actual defects, not style nits — worth fixing regardless of any
broader refactor.

- **`ontolearn/learners/tree_learner.py:441`** — the "no features extracted"
  error message references `self.use_data_properties`, an attribute
  `TDL.__init__` never sets (only `use_data_properties_boolean/string/date/numeric`
  exist). Hitting this branch raises `AttributeError` instead of the intended
  descriptive error, hiding the real problem from users. (`TDL_refinement`
  does define this attribute, so only the base `TDL` class is affected.)

## 2. Code duplication

- **`ontolearn/learners/tree_learner_refinement_inherit.py` vs
  `tree_learner.py`** — `TDL_refinement(TDL)` is ~70% copy-pasted rather than
  extended. `fit()` (tree_learner_refinement_inherit.py:604-761) duplicates
  `TDL.fit()` (tree_learner.py:661-736) almost line-for-line (grid search,
  classifier training, `report_classification`, plotting) with only the
  refinement loop spliced in; `extract_expressions_from_owl_individuals`,
  `create_training_data`, and `_extract_data_property_features` are likewise
  wholesale overrides. A template-method refactor — pulling
  `_train_classifier`/`_report`/`_plot` into the base `TDL` as hooks — would
  remove on the order of 300 duplicated lines. (`OCEL(CELOE)` in
  `ontolearn/learners/celoe.py`/`ocel.py` is a good counter-example of doing
  this inheritance correctly — worth using as the template.)
- **`ontolearn/knowledge_base.py:191-262`** — `abox()` repeats the same
  triple-traversal ~70 lines of near-identical branching, once per
  `mode in {"native", "iri", "axiom"}`, differing only in output formatting.
  Factor a single internal `(subject, predicate, object)` generator with
  per-mode formatters applied on top.
- **`ontolearn/search.py:645-662` and `:790-810`** — `get_top_n` in
  `SearchTreePriorityQueue` and `DRILLSearchTreePriorityQueue` duplicate the
  same `key == 'quality'/'heuristic'/'length'` dispatch. Extract a shared
  strategy map/helper.

## 3. Error handling & logging

- **Logging infra is built but essentially unused.** `ontolearn/utils/log_config.py`
  and `oplogging.py` provide a real fileConfig-based setup with a TRACE
  level, but only ~8 call sites across `ontolearn/` actually use
  `getLogger`/`oplogging`/`log_config`. Meanwhile `print(` appears **363
  times** across `ontolearn/**/*.py`. Worst offenders: `learners/drill.py`
  (37), `semantic_caching.py` (25), `learners/tree_learner_refinement_inherit.py`
  (25), `learners/spell_kit/structures.py` (18), `learners/nces.py` (17).
  `knowledge_base.py` even defines `logger = logging.getLogger(__name__)`
  (line 54) but still uses `print()` for warnings at lines 103-104 and 289.
  Routing these through the existing logger (with levels) would make output
  controllable/filterable instead of unconditionally printed.
- **Bare `except:`** in `ontolearn/data_struct.py:289,388` swallows
  everything including `KeyboardInterrupt`/`SystemExit`. Narrow to the
  specific expected exception type.
- **Broad `except Exception:` that just prints and continues**, masking real
  bugs as "no result found": `learners/tree_learner.py:525,547,592`,
  `learners/nces.py:101,144,177,195,234,311,319`,
  `learners/nces2.py:125,146,272,280,381`,
  `learners/sparql_query_learner.py:252`. Also
  `learners/celoe.py:183-185` has a bare `except ValueError: pass` next to an
  unresolved `# TODO:CD: We need to understand this` — worth actually
  chasing down before merging code that silently ignores it.
- **`learners/celoe.py:290-311`** — `_add_node_evald` is dead code, a
  near-duplicate of `_add_node` (255-288) with zero call sites anywhere in
  the repo. Remove it.
- **`learners/base.py:167-173`** — `BaseConceptLearner.train()` is a no-op
  `pass` stub, yet `drill.py:251` defines its own real `train()` outside that
  contract. The shared abstraction isn't actually shared; either give the
  base method real shared behavior or drop it and document `train()` as
  learner-specific.

## 4. Mutable default arguments / shared state

- **`learners/tree_learner.py:287`** and
  **`learners/tree_learner_refinement_inherit.py:59`** —
  `kwargs_grid_search: dict = {}` is a classic mutable-default-argument bug.
  The same dict is shared across every `TDL`/`TDL_refinement` instance
  created without explicitly passing this argument, and it's mutated in
  place (`.setdefault("cv", 10)` at tree_learner.py:317) — state can leak
  between unrelated learner instances. Use `None` as the default and
  construct a fresh dict inside `__init__`.

## 5. Reproducibility

- **`learners/drill.py`** uses unseeded `random.random()`/`random.choice()`/
  `random.sample()` (lines 648-649, 744-784) with no `random_state`/`seed`
  constructor parameter and no `torch.manual_seed()` anywhere in
  `Drill.__init__`. Runs aren't reproducible, and there's no way to make
  them so via the public API. `EvoLearner`'s DEAP-driven GA has the same
  gap — no seed parameter exposed. Worth adding a `random_state` param
  threaded through to `random`/`numpy`/`torch`/DEAP, consistent with the
  sklearn convention this codebase otherwise follows (e.g. its
  `fit`/`predict` naming).
- GPU/CUDA handling itself is fine — `clip.py:138`, `nero.py:107`,
  `drill.py:92` all correctly guard with
  `torch.device("cuda" if torch.cuda.is_available() else "cpu")`; no
  hardcoded `.cuda()` calls found.

## 6. Testing gaps

- **Two test files are 100% commented out** and run zero assertions while
  still being collected by pytest: `tests/test_semantic_cache.py` (68/68
  lines) and `tests/test_example_concept_learning_neural_evaluation.py`
  (168/168 lines). Either restore them or delete them — as-is they give a
  false sense of coverage.
- The committed coverage report (`docs/usage/09_further_resources.md:191-244`,
  v0.10.0, 82% overall) shows specific weak spots worth targeted tests:
  `ontolearn/incomplete_kb.py` 8% (73/79 statements missed),
  `ontolearn/quality_funcs.py` 31%, `ontolearn/nces_utils.py` 39%,
  `ontolearn/triple_store.py` 53% (237/501 missed),
  `ontolearn/data_struct.py` 60%.
- **`README.md:3`** badge claims 86% coverage; the committed report in
  `docs/usage/09_further_resources.md:243` says 82%. Pick one source of
  truth and keep the badge in sync with it (ideally generated by CI rather
  than hand-updated).
- CI (`.github/workflows/test.yml`) scopes `ruff check` to
  `ontolearn/learners` only (excluding `spell_kit`), not the whole package —
  most of `ontolearn/` (core reasoning modules, `utils/`, etc.) isn't linted
  in CI at all.

## 7. Packaging

- No `pyproject.toml` — packaging is still pure legacy `setup.py`. Migrating
  to `pyproject.toml` (PEP 621) would be a larger, separate effort, but is
  worth planning for since `setup.py`-only packaging is increasingly
  unsupported by newer tooling.
- Dependency pinning is inconsistent with no explanation: most deps are
  `>=` range-pinned, but `owlapy==1.6.6`, `dicee==0.3.2`, `lxml==5.3.0`,
  `python-sat==0.1.7.dev23`, and `shap==0.49.1` are exact-pinned
  (`setup.py:42-70`) with no comment on why those five specifically need
  exact pins. A short comment per exact-pin (e.g. "pinned: breaking change
  in X") would keep this from looking accidental.
- `setup.py:122` sets `python_requires='>=3.11'` but the trove classifier at
  `setup.py:119` still says `"Programming Language :: Python :: 3.10"` —
  stale, should read 3.11 (CI only tests 3.11.14 anyway).

## 8. Lint configuration

- `ruff.toml` is pinned to a minimal ruleset (`E4`, `E7`, `E9`, `F`) with a
  200-char line length and no line-length rule, no import-sort (`I`), and no
  complexity/naming checks. The comment in the file explains this was to
  avoid the 0.15→0.16 default-ruleset jump (~60→~415 rules) — reasonable as
  a stopgap, but worth revisiting deliberately (e.g. opt into `I` for import
  sorting and `UP` for pyupgrade) rather than leaving it at the bare minimum
  indefinitely.

## 9. Dead / orphaned code

- **`ontolearn/incomplete_kb.py`** has zero real imports anywhere in
  `ontolearn/`, `tests/`, or `examples/` (only mentioned in a docstring in
  `ontolearn/utils/_lazy_owlready2.py:29`), and 8% test coverage. Effectively
  orphaned — either wire it into a test/example or remove it, similar to the
  now-deleted `tentris.py`.
- **`ontolearn/logging_tentris.conf`** is a leftover config file from the
  just-removed `tentris.py` (PR #602) and is now unused — safe to delete in
  a follow-up.
- **`ontolearn/binders.py`** and **`ontolearn/executor.py`** are each used by
  exactly one example script and nowhere else in `ontolearn/` or `tests/` —
  worth confirming they're still meant to be supported public surface rather
  than vestigial.

## Suggested priority order

1. Fix the remaining correctness bug in §1 (cheap, high-value, likely a
   one-liner).
2. Fix the two mutable-default-argument bugs in §4 (cheap, real risk).
3. Delete or restore the two fully-commented-out test files in §6.
4. Tackle the `tree_learner_refinement_inherit.py` duplication in §2 as a
   dedicated refactor (largest single win, but the biggest single change).
5. Everything else (logging migration, packaging modernization, lint
   tightening) as incremental follow-ups, not blocking.
