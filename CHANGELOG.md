# Changelog

All notable changes to Ontolearn are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

When cutting a release, move the relevant `[Unreleased]` entries under a new
`## [x.y.z] - YYYY-MM-DD` heading (matching the PyPI/GitHub release version and
date) and use that section as the basis for the GitHub release notes.

## [Unreleased]

### Fixed
- `save_owl_class_expressions` called `exit(1)` after printing debug output when an
  `OWLEquivalentClassesAxiom` could not be added, terminating the whole interpreter from library
  code. It now logs the details and re-raises the original `AttributeError`.
- `PrepareBatchOfTraining` (`data_struct.py`) batching errors (`invalid input detected during batching in
  X`/`Y`) now carry their message in the raised `ValueError` instead of printing it beforehand.
- `NERO` crashed whenever CUDA was available: the model was moved to the GPU for training, but the
  index tensors for `search`, `fit` and the embedding getters were built on the CPU
  (`RuntimeError: Expected all tensors to be on the same device`), and the predicted indices were
  converted with `.numpy()` straight from the GPU. Indices are now created on the model's device
  (`NERO._index_tensor`), `forward` moves its inputs there, and results are moved back with
  `.cpu()`. CI has no GPU and never saw this.
- `LearningProblemGenerator` raised a bare `ValueError` for an invalid `search_algo`; the message
  was printed beforehand. It is now the exception message.
- `NCESTrainer` and `CLIPTrainer` raised a bare `ValueError` for an unsupported optimizer; the
  message was a `print` placed after the `raise` and never ran. It is now the exception message.
- `TripleStoreReasoner.instances(ce, direct=False)` only expanded subclasses for the root
  variable's type triple, so class expressions with fillers (e.g.
  `Grandfather ⊓ ∃hasChild.Grandparent`) missed individuals typed only by a subclass of the
  filler. Every type triple is now expanded, each with its own variable, and `instances_count` and
  `is_subset_of_individuals` share the same helper (`with_subclass_closure`), which also removes
  the shared-variable scoping bug of #575 from those two. (#571)
- `ontolearn.semantic_caching.retrieve` called `NeuralOntology(path_of_kb=...)`,
  an argument removed from owlapy, so the neural-reasoner path raised
  `TypeError` whenever no KGE model path was given. It now trains/loads via
  `path_neural_embedding` and caches the reasoner per KG instead of rebuilding
  it for every concept. (#515)

### Changed
- `NCES`, `NCES2` (and so `ROCES`) report status, warnings and verbose predictions through
  `logging` instead of `print`, without the ANSI colour codes, so the output can be filtered or
  silenced with the standard logging configuration. A leftover debug `print(num_workers)` in
  `NCES2.train` is now a debug log.
- `TDL` / `TDL_refinement` (`learners/tree_learner.py`) report progress (feature extraction,
  training, grid-search parameters, classification report) through `logging` instead of `print`.
  The duplicate verbose `print` after each "Error extracting ... features" warning is removed,
  as the same error was already logged.
- `NERO` (`verbose` levels map to `info`/`debug`; the "queue exceeded max" notice is now an
  unconditional `warning`), `NCESTrainer` and `CLIPTrainer` report through `logging` instead of
  `print`; the decorative `#`/`*` banners are gone.
- `LearningProblemGenerator` and the `lp_generator` data generator (`KB2Data`) report progress and
  constraint problems through `logging` instead of `print` (the insufficient-concepts and
  unfulfilled-constraints notices are warnings).
- `binders` (DL-Learner output parsing), `utils.static_funcs` (`save_owl_class_expressions`,
  `verbalize`), `data_struct` and the diagnostic messages of `search` report through `logging`
  instead of `print`. The tree-display methods of `search` still print on purpose.
- Re-enabled `tests/test_semantic_cache.py` (previously 100% commented out), exercising only the
  symbolic (HermiT) reasoner; the neural (EBR) path is left out as in-process KGE training
  alongside the JVM segfaulted CI (#515).
- `tests/test_semantic_cache.py` now also covers the neural (EBR) retrieval path, using the
  a KGE trained on a temporary copy of `father.owl`, in a child process so a torch + JVM
  crash fails that one test instead of killing the suite. (#515)
- Removed `tests/test_example_concept_learning_neural_evaluation.py`, which was
  100% commented out and imported the long-gone `ontolearn.concept_learner`.
- `Improvements.md` pruned of items already resolved (`CELOE._add_node_evald`
  removal, `BaseConceptLearner.train()` documented as an optional hook,
  unit-test coverage for previously weak modules) and its priority list
  refreshed.
- NCES/NCES2/ROCES tests (`tests/test_nces*.py`, `tests/test_roces.py`) run in
  ~20s instead of ~100s: `test_nces.py` and `test_nces_trainer.py` now use the
  shipped pretrained family embeddings/model instead of training embeddings
  via a `dicee` subprocess and the synthesizer for 5 epochs on every run;
  NCES2/ROCES (no shipped weights) train briefly on a few LPs. Quality
  thresholds raised (0.1 -> 0.4 pretrained NCES, 0.3 NCES2; ROCES stays at 0.1 because its
  trained quality varies run to run), and the
  tests run in a temporary working directory so they no longer leave
  `*-Experiment-*`, `temp_embeddings/` and `temp_triples/` behind. (#559)
- CI (`.github/workflows/test.yml`): `push` now only triggers on `develop`/`main`
  (PRs run via `pull_request`), so branches with an open PR are no longer tested
  twice; superseded runs are cancelled; and on PRs only the tests of the touched
  learner(s) run unless shared code changed (`.github/scripts/select_tests.py`;
  unmapped files fall back to the full suite, docs-only PRs skip the test job). (#645)
- CI (`.github/workflows/test.yml`): the test suite is split into `learners`,
  `neural` and `triplestore` matrix jobs that run in parallel
  (`.github/scripts/split_tests.py`; new test files land in `learners` unless
  matched). Only `triplestore` starts Fuseki, every job restores the data
  cache, and the PR test selection still applies (groups with nothing selected
  are not started). (#651)

## [0.11.0] - 2026-10-15

### Removed
- `ontolearn/logging_tentris.conf`, a leftover logging config from the
  removed `tentris.py`, unused anywhere in the package.
- `CELOE._add_node_evald` (`ontolearn/learners/celoe.py`), dead code with no
  call sites — a near-duplicate of `_add_node`.
- `ontolearn/tentris.py`, a stale, self-marked-TODO script unused anywhere
  else in the codebase. It imported `ontolearn.base`, a module removed back
  in the owlapy 1.1.0 migration, so it hadn't been importable for a long
  time. (#599)

### Added
- `best_hypotheses(..., return_info=True)` on `CELOE`/`OCEL`, `EvoLearner`,
  `Drill`, `NCES`, `NCES2` and `ROCES` returns `(hypotheses, info)`, where
  `info` holds the `quality` and `length` of each hypothesis. `fit` and the
  default return value are unchanged. `TDL`, the SAT learners and `NERO` don't
  have a node-based `best_hypotheses` and are not covered yet. (#498)
- `ontolearn.lp_generator.GoalDirectedLPGenerator`
  (`ontolearn/lp_generator/complexity_lp_generator.py`), a goal-directed,
  complexity-controlled learning-problem generator. Unlike `LPGen`/`KB2Data`,
  which refine `owl:Thing` bottom-up and keep whatever falls out, it samples
  target concepts top-down from an explicit DL-construct grammar
  (`ComplexityProfile` / `DLConstruct`: intersection, union, ∃, ∀,
  cardinality, has-value, negation, plus length/depth/arity bounds),
  retrieves E+/E- via the reasoner and discards trivially-solvable problems
  (a cheap one-step baseline hypothesis already separates E+/E-).
  `generate_benchmark()` runs a curriculum of profiles and `save_benchmark()`
  persists it.
  - Difficulty-aware: `DifficultyWeights`/`difficulty_score` combine target
    length/depth, (1 - baseline F1) and near-miss-negative density;
    `generate()` oversamples and keeps the hardest `num_lps`, biasing
    negatives towards near-misses (`hard_negative_ratio`).
  - Works on any knowledge base, including ones with no object properties
    (relation-based constructs are dropped from the grammar).
  - Example sweeping several KGs with TDL:
    `examples/benchmark_complexity_lp_generator_with_tdl.py`.
- Unit tests for previously weakly-covered modules: `quality_funcs`
  (31% -> 100%), `nces_utils` (39% -> 87%), `incomplete_kb` (8% -> 91%),
  `data_struct` (~60% -> 97%), `value_splitter` incl. `EntropyValueSplitter`
  (33% -> 100%) and `utils/static_funcs` (37% -> 92%).
- `Drill.__init__` (`ontolearn/learners/drill.py`) accepts a `random_state`
  param, seeding a dedicated `random.Random` instance (used for exploration/
  example sampling instead of the global `random` module) plus `torch`/CUDA,
  so runs are reproducible given the same seed. Previously unseeded with no
  way to make runs reproducible via the public API. `EvoLearner`'s
  DEAP-driven GA has the same gap, still open. (#624)
- Example for generating learning problems with numeric data properties
  (`examples/generate_numeric_lps.py`). (#593)
- Explicit `ruff.toml` pinning the enforced lint rule set (`E4`, `E7`, `E9`,
  `F`) instead of relying on ruff's shipped default. (#594)

### Changed
- CI caches the `KGs`/`LPs`/`NCESData`/`CLIPData` archives with `actions/cache`,
  so runs no longer download them from `files.dice-research.org` every time. (#637)
- Version bumped to 0.11.0 (`setup.py`, `ontolearn/__init__.py`, README
  badges, docs). The `setup.py` trove classifier and the README/installation
  docs now state Python 3.11, matching `python_requires='>=3.11'` (they still
  said 3.10).
- `print()` calls in `knowledge_base.py`, `semantic_caching.py`,
  `learners/tree_learner_refinement_inherit.py`,
  `learners/spell_kit/structures.py`, and `learners/drill.py` now go through
  module-level `logging.getLogger(__name__)` loggers (`info`/`warning`/
  `debug`/`error`, matched to the message), so their output is controllable
  and filterable via the existing logging config instead of always printing
  unconditionally to stdout. `structures.py`'s `structure_to_dot`, whose
  `print()`s produce actual DOT-graph output rather than log messages, is
  left as-is. Remaining offenders across the rest of `ontolearn/` are noted
  in `Improvements.md`.
- `BaseConceptLearner.train()` (`ontolearn/learners/base.py`) is now documented
  as a learner-specific optional hook (overridden by Drill, NCES, NCES2, ROCES,
  CLIP with their own signatures) and returns `self` as its docstring always
  claimed, instead of being a misleadingly-documented "Train RL agent" stub
  returning `None`.
- Upgraded to owlapy 1.6.6, which makes owlready2 an optional install extra
  rather than a hard dependency; `ontolearn/incomplete_kb.py` and
  `ontolearn/semantic_caching.py` switched from module-load-time
  `from owlready2 import *` to a lazy `ontolearn.utils.import_owlready2()`
  helper so importing them no longer requires owlready2 unless
  owlready2-backed functionality is actually used. (#600)
- `TripleStore.individuals_count()` and `TripleStore.most_general_object_properties()`
  (`ontolearn/triple_store.py`) fetched and materialized the full SPARQL result set
  over HTTP just to compute a count or test a domain/range subset relationship,
  transferring and parsing far more data than needed on a large or remote endpoint
  (e.g. DBpedia). `individuals_count` now issues a server-side `COUNT` query
  (`TripleStoreOntology.individuals_in_signature_count`/`TripleStoreReasoner.instances_count`).
  `most_general_object_properties` now checks each object property's domain/range
  against the target domain via a single short-circuiting `ASK` query
  (`TripleStoreReasoner.is_subset_of_individuals`) instead of fetching every
  property's full instance set per iteration; this also fixes the subset check
  itself, which previously compared individuals against the raw (unconsumed)
  domain/range generator instead of the individuals it would yield, so it could
  only ever match when the domain was `owl:Thing`. (#613)
- `tests/test_example_concept_learning_evaluation.py::TestConceptLearning::test_learning`
  intermittently failed DRILL's mean-F1 assertion (`assert x > 0.9`) in CI. Now
  passes `random_state=42` to `Drill` and gives it a larger dedicated runtime
  budget; its threshold is lowered to `0.85` since the search is still
  wall-clock time-boxed, so outcome still varies somewhat with runner load
  even with a fixed seed. See #624.
- `KnowledgeBase.abox()` (`ontolearn/knowledge_base.py`) repeated the same
  rdf:type/data-property/object-property triple traversal once per
  `mode in {"native", "iri", "axiom"}`, differing only in output formatting.
  Factored into a shared `_abox_native_triples` generator with per-mode
  formatter methods applied on top.
- `ModifiedCELOERefinement._setup()` (`ontolearn/refinement_operators.py`, used by
  `CELOE`/`OCEL`/`ExpressRefinement`) eagerly swept every object property against
  every individual in the knowledge base at construction time, purely to precompute
  `max_nr_fillers` for cardinality-restriction refinements — an O(properties x
  individuals) reasoner cost paid up front regardless of whether cardinality
  refinements on a given property were ever explored during search. `max_nr_fillers`
  is now a lazily-populated mapping (`_LazyMaxNrFillers`) that computes and caches a
  property's value on first lookup instead. (#612)
- `TDL_refinement._merge_binary_feature_matrices` re-rendered each candidate
  feature's DL syntax string from scratch inside the per-example loop
  (`owl_expression_to_dl` scales with expression complexity), giving
  O(examples × features × expression-size) work instead of O(features ×
  expression-size); now precomputed once per feature before the example
  loop. (#611)
- `TDL_refinement.fit()` (`ontolearn/learners/tree_learner_refinement_inherit.py`)
  duplicated `TDL.fit()` (`ontolearn/learners/tree_learner.py`) almost
  line-for-line for grid search, classifier training, classification
  reporting, and plotting, with only the SHAP-based refinement loop spliced
  in; likewise `create_training_data` was a byte-for-byte copy of `TDL`'s.
  Extracted the shared steps into `_fit_decision_tree`/`_report_and_plot`/
  `_finalize_predictions`/`_maybe_plot_embeddings` hooks on the base `TDL`
  class, and `TDL_refinement` now inherits `create_training_data` instead of
  redefining it. (#616)
- `quality_funcs.f1`/`acc` reimplemented the tp/tn/fp/fn and precision/recall
  arithmetic already implemented in `metrics.py`'s `F1`/`Accuracy` classes; now
  delegate to those classes instead of duplicating the math.
- `SearchTreePriorityQueue.get_top_n` and `DRILLSearchTreePriorityQueue.get_top_n`
  (`ontolearn/search.py`) duplicated the same `key == 'quality'/'heuristic'/'length'`
  dispatch; extracted into a shared `_get_top_n_nodes` helper that both now call,
  parameterized by how each tree obtains a node's length.
- Ruff upgraded to `>=0.16.0`. (#594)
- `TDL`/`TDL_refinement` data-property filtering extended to exclude boolean
  data properties, with a toggle to enable/disable boolean data properties.
  (#588)
- `max_runtime` for TDL feature matrix creation/extraction changed to `None`
  by default, matching the other learners. (#585)

### Fixed
- `LPGen`/`KB2Data` relied on a module-level `random.seed(42)` that reseeded
  the global `random` module on every import, so the same seed did not
  reproduce the same learning problems once anything else touched global
  random state. `LPGen`, `KB2Data`, `ConceptDescriptionGenerator` and
  `ExpressRefinement` now take a `random_seed` and use their own
  `random.Random` instance; the import-time seeding is gone. (#595)
- `TripleStore` instance retrieval returned too few individuals for class
  expressions with a `⊤` filler such as `∃r.⊤` on a triplestore without
  reasoning (#581): owlapy <= 1.6.5 rendered the filler as an explicit
  `?s a owl:Thing` pattern that raw triples never satisfy, so such branches
  matched nothing (e.g. 127 instead of 259 instances on Carcinogenesis).
  Resolved by the owlapy 1.6.6 upgrade (#600); a regression test now guards
  the generated SPARQL.
- `TDL`/`TDL_refinement` (`ontolearn/learners/tree_learner.py`,
  `tree_learner_refinement_inherit.py`) declared `kwargs_grid_search: dict = {}`,
  a mutable default argument that was mutated in place via
  `.setdefault("cv", 10)`, so state leaked between unrelated learner
  instances (and into caller-supplied dicts). The default is now `None` and
  each instance gets its own copy.
- `CLIPDatasetInference.__getitem__` (`ontolearn/data_struct.py`) built the
  negative-example tensor from the *positive* examples (`embeddings.loc[pos]`
  instead of `embeddings.loc[neg]`), so `CLIP`'s length predictor ignored the
  negatives at inference time even though it was trained with them. (#638)
- Two bare `except:` clauses swallowed everything, including
  `KeyboardInterrupt`/`SystemExit`: `NCESBaseDataset.decompose`'s numeric-token
  check (`ontolearn/data_struct.py:289`) narrowed to `ValueError` (the only
  thing `int(char)` can raise there), and `NCESDatasetInference.__getitem__`'s
  embedding lookup (`data_struct.py:388`) narrowed to `KeyError` (the only
  thing a missing-label `.loc[]` lookup can raise).
  `SPARQLQueryLearner._compute_f1_score` (`ontolearn/learners/sparql_query_learner.py`)
  caught bare `Exception` for the same reason its own `_http_request` raised a
  bare `Exception("Query failed")` on a non-2xx response; `_http_request` now
  raises `requests.exceptions.HTTPError`, so the catch narrows to
  `(requests.exceptions.RequestException, KeyError, IndexError, ValueError)` -
  the actual failure modes of a malformed candidate SPARQL query (built from
  `_powerset_of_filters` combinations) or its response.
- `TDL.extract_expressions_from_owl_individuals`'s three per-individual feature
  extractors (`ontolearn/learners/tree_learner.py`) caught `Exception` around
  both the reasoner lookups *and* the OWL-object construction that follows
  them, so a real bug in the construction code (not just a reasoner miss)
  would silently reduce a feature set instead of surfacing. Narrowed
  `_extract_inverse_property_features`'s inner `try` to wrap only the
  `get_object_property_values` reasoner call; all three now always log the
  caught exception (previously only printed when `verbose > 0`, so a bug
  hitting the default-quiet path left no trace at all).
- `CELOE.updating_node`'s `except ValueError: pass` around
  `heuristic_queue.discard(node)` carried an unresolved
  `# TODO:CD: We need to understand this`. `heuristic_queue` is a `SortedSet`
  keyed by the node's `heuristic` value; `discard()` looks a node up by its
  *current* key, so the `ValueError` means the node's queue membership and
  its `heuristic` attribute have gone out of sync (e.g. mutated somewhere
  other than through this context manager). Documented the actual mechanism
  and replaced the silent `pass` with a `logger.warning` so a future
  occurrence is visible instead of invisible.
- `TripleStoreReasoner.instances` (`ontolearn/triple_store.py`), used for indirect
  (`direct=False`) instance retrieval on a `TripleStore`, rewrote every `?x a `
  occurrence in the generated SPARQL to the same `?some_cls` variable. For class
  expressions with more than one such occurrence in different scopes — e.g.
  `(¬Daughter) ⊓ Female`, which places one inside a `FILTER NOT EXISTS` block and
  one outside it — the shared variable name let the two occurrences bind to each
  other instead of being scoped independently, producing incorrect results. Each
  occurrence now gets its own uniquely numbered `?some_cls_N` variable. Covered by
  `tests/test_triplestore_indirect_instances_sparql.py`, which mocks the HTTP layer
  so it doesn't need a live triplestore. (#575)
- `TripleStoreOntology.object_property_domain_axioms`/`data_property_domain_axioms`/
  `object_property_range_axioms` (`ontolearn/triple_store.py`) crashed with
  `TypeError: cannot unpack non-iterable NoneType object` for any property with no
  explicit `rdfs:domain`/`rdfs:range` triple, because the `peek()` helper they rely on
  returned a bare `None` for an empty generator instead of the `(None, generator)` tuple
  its callers unpack. Previously unreachable in practice since nothing exercised these
  axioms for a property lacking a domain/range; surfaced by the `#613` fix above once
  `most_general_object_properties` started actually calling into them for every object
  property in the signature. `peek()` now returns `(None, generator)` on an empty
  generator, matching its documented contract.
- `ALCSAT`/`SPELL` (`SATBaseLearner._convert_kb_to_structure`,
  `ontolearn/learners/sat_base.py`) crashed with
  `AttributeError: 'Or' object has no attribute 'iri'` while initializing
  over ontologies that illegally pun an entity as multiple property types
  (e.g. `KGs/Biopax/biopax.owl`'s `DELTA-G`, declared as both
  `ObjectProperty` and `AnnotationProperty`), because owlready2's load-time
  punning repair returned internal class-expression nodes as object-property
  values. Resolved as a side effect of the owlapy 1.6.6 upgrade above
  (`#600`), which routes `StructuralReasoner.object_property_values()`
  through a helper that skips such malformed values with a warning instead
  of crashing; not previously credited for this fix. (#598)
- `TDL_refinement.classification_report` always raised `AttributeError`:
  `TDL_refinement.fit()` set `self.__classification_report` from within its
  own class body, which Python name-mangles to a different attribute than
  the one the `classification_report` property (defined on `TDL`) reads.
  Fixed as part of deduplicating `fit()` into shared `TDL` hooks, above; the
  report is now always set from the base class's `_report_and_plot`, so both
  classes agree on the mangled name. `tests/test_tdl_refinement.py`'s
  `test_classification_report` re-enabled. (#616)
- `NCESTrainer.train()` created a fresh `DataLoader(num_workers=...)` worker
  pool per architecture in a loop; the existing `os.cpu_count()`-based cap on
  `num_workers` only engaged when `cpu_count <= num_workers`, so it silently
  did nothing on machines with more cores than the default of 8, and repeated
  worker-pool creation at that count could deadlock under `coverage run`
  (`NCES`/`NCES2`/`ROCES` training, e.g. `test_nces_trainer.py`). Now capped
  unconditionally to `min(requested, cpu_count-1, 4)`. (#610)
- `tests/test_nces.py`, `test_nces2.py` (and `ROCES`, which subclasses
  `NCES2`) intermittently hung for 1h+ in CI (#622), stalling PRs until
  manually cancelled. Root cause: several `DataLoader` construction sites in
  `ontolearn/learners/nces.py`/`nces2.py` still built a multiprocessing
  worker pool (`num_workers=self.num_workers`, `NCES2` even doing so once
  per architecture in a loop) — the same fresh-worker-pool-under-
  `coverage run` deadlock pattern already found and fixed for
  `NCESTrainer.train()` in #610, just missed elsewhere: the four
  inference-only call sites (`NCES`/`NCES2` `fit_one`/`fit_from_iterable`,
  no throughput benefit from multiprocessing when fetching one or a few
  batches) and the "quick, low-fidelity" `auto_train` convenience path in
  both classes' `_set_prerequisites` (5-epoch training on a tiny dataset,
  where #610's capped-but-nonzero worker count still wasn't safe under
  coverage). All six now use `num_workers=0`. Also added a `timeout=600`
  to the untimed `dicee` embedding-training subprocess in
  `NCES._set_prerequisites` as defense in depth, and a job-level
  `timeout-minutes: 90` safety net in `.github/workflows/test.yml` so any
  future hang fails fast instead of running for hours. The three NCES test
  files, temporarily excluded from CI while this was diagnosed (#623), are
  re-enabled.
- `ConceptAbstractSyntaxTreeBuilder._fix_mid_tokens_errors`/`_postprocess_tail_fix`/
  `_enforce` repaired malformed token sequences via `random.choice(...)`, making
  parser output non-deterministic across runs on identical input; replaced with
  a deterministic (sorted) tie-break.
- `EvoLearner` registered its DEAP `Fitness`/`Quality`/`Individual` types under
  fixed names in DEAP's process-global `creator` module and tore them down in
  `clean()` behind a bare `except AttributeError: pass`; concurrent
  `EvoLearner` instances (e.g. `fit()` calls interleaved across threads) could
  silently clobber each other's DEAP type definitions. Each instance now
  registers its own uniquely-named types. (#609)
- `TDL`'s "no features extracted" error message referenced `self.use_data_properties`,
  an attribute `TDL.__init__` never sets, raising `AttributeError` instead of the
  intended descriptive error; now reports the actual
  `use_data_properties_boolean/string/date/numeric` flags.
- `DRILLSearchTreePriorityQueue.get_top_n()` raised `AttributeError`/`TypeError`
  on the never-assigned `self.refined_nodes` attribute; now builds its node
  list from `self.nodes.values()`. (#605)
- `LengthBasedRefinement.refine_atomic_concept`'s disjointness check was dead
  logic — both branches of the `if`/`else` built the same expression, so
  refinements into a class disjoint with the current concept were never
  actually skipped. (#604)
- `LearningProblemGenerator` was not working; `tests/test_learning_problem_generator.py`
  re-enabled. (#589)
- TDL++ integer-typed expression handling; two more intervals added to range
  generation; range generation skipped when the value distribution can't be
  fitted. (#593)
- Malformed range/domain axioms are skipped instead of aborting the whole
  learner run. (#587)

## [0.10.0] - 2026-02-10

### Added
- SAT-based learners **SPELL** and **ALCSAT** integrated.
- **NERO** integrated.
- GPU execution support for Drill, with stricter stopping conditions.

### Changed
- Concept and neural learners extended and refactored, with better feature
  construction and validation. All learners now live under
  `ontolearn.learners`.
- Triple store: caching and stability improvements.
- Web UI extended for a smoother experience.
- TDL received multiple fixes and refinements.
- Upgraded to owlapy 1.6.3.

### Removed
- Deprecated components such as `EBR` (moved to owlapy).

## [0.9.2] - 2025-06-12

### Fixed
- Issue in the `triple_store` module from the 0.9.1 release.

### Changed
- `executor` module now supports both TDL and NCES.
- `verbalizer` module improvements.

## [0.9.1] - 2025-05-26

### Added
- Caching for OWL reasoners.

### Changed
- Upgraded to owlapy 1.5.1, resolving key dependency issues; sync (Java)
  reasoner integration tested and stabilized.

## [0.9.0] - 2025-03-03

### Added
- **ROCES** and **NCES2** learners.
- Semantic caching algorithm for OWL reasoners.

### Changed
- Major refactor of knowledge base classes, especially `TripleStore`:
  - Removed some `KnowledgeBase` hyperparameters and caching operations.
  - `TripleStore` and `TripleStoreKnowledgeBase` merged into a single class.
  - `TripleStore` knowledge base can now be used by all learners except
    NCES, NCES2, and ROCES.
- Drill: bug fixes and optimizations.

## [0.8.1] - 2024-11-19

### Changed
- Upgraded to owlapy 1.3.3, bringing several fundamental changes and
  accompanying fixes.

### Fixed
- Neural reasoner fixes, prediction caching (LRU), `test_owl_neural_retrieval`
  ABox method fix.
- Learning SPARQL queries using string literals.

## [0.8.0] - 2024-10-28

### Removed
- `ModelAdapter` removed.

### Changed
- Code adjustments for owlapy's newest versions (up to 1.3.1) and
  refactoring for the Enexa project.
- NCES integrated into the `ontolearn-webservice`.
- DL-Learner binding fixes.

## [0.7.3] - 2024-08-01

### Added
- ALCQIO retrieval with neural nets.

### Changed
- TDL refactoring.
- `OWLLiteral` with string value support; triples based on literals can now
  be parsed.

## [0.7.2] - 2024-07-11

### Changed
- **Breaking:** modules under `ontolearn/base` removed from Ontolearn;
  the classes are now provided by [owlapy](https://github.com/dice-group/owlapy)
  (owlapy 1.1.0). Documentation guides moved to owlapy's docs accordingly.
- Triplestore improvements; `ontolearn-webservice` API changes; nominals
  support in Drill; default cardinality restriction handling.

## [0.7.1] - 2024-05-09

### Changed
- DRILL refactored.
- `TripleStore` reworked on top of `rdflib.graph`; TDL over triplestore.
- Adapted to owlapy 1.0.1.

### Added
- LLM-based verbalizer.
- TDL class expression learning example over a DBpedia SPARQL endpoint.

## [0.7.0] - 2024-03-07

### Added
- **Drill**, **Tree-based DL Learner (TDL)**, and **CLIP** learners
  available via `ontolearn.learners` / `ontolearn.concept_learner`.
- Triple Store Knowledge Base: `TripleStoreOntology`, `TripleStoreReasoner`,
  `TripleStoreKnowledgeBase`, usable with just a SPARQL endpoint.
- `KnowledgeBase.abox()` / `.tbox()` / `.triples()` for triple retrieval, in
  `'native'`, `'iri'`, or `'axiom'` modes.
- [Ontosample](https://ontolearn-docs-dice-group.netlify.app/autoapi/ontosample/)
  integrated (available via `pip install ontolearn[full]`).
- Learning problem generator as a Python module.

### Changed
- `KnowledgeBase`: `kb.ontology()`/`kb.reasoner()` are now properties
  (`kb.ontology`/`kb.reasoner`); optional `include_implicit_individuals` flag
  for type retrieval.
- Dependencies split into `ontolearn[min]` (default) and `ontolearn[full]`
  (extras).

### Removed
- Triplestore logic removed from `OWLOntology_Owlready2` /
  `OWLReasoner_Owlready2`, moved to `ontolearn.triple_store`.

### Fixed
- Reusing the same `EvoLearner` model to fit more than one learning problem
  no longer causes a quality drop.

## [0.6.1] - 2023-12-03

Maintenance release.

## [0.5.4] - 2023-08-17

Maintenance release.

## [0.5.3] - 2023-02-10

Maintenance release.

## [0.5.2] - 2022-10-17

### Added
- Domain inclusion checking in `ConceptGenerator`.
- Top-level CNF/DNF conversion.

### Fixed
- OCEL fixes (not yet fully equivalent to the DL-Learner implementation).
- `DLSyntaxParser` correctly parses `Thing`/`Nothing`.
- Multiple `fit()` calls on `EvoLearner` for datasets with data properties.
- Super-property filtering in `OWLReasoner_Owlready2`.

### Changed
- Closed-world behaviour used by default for negation
  (`FastInstanceChecker`).

[Unreleased]: https://github.com/dice-group/Ontolearn/compare/0.11.0...develop
[0.11.0]: https://github.com/dice-group/Ontolearn/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/dice-group/Ontolearn/compare/0.9.2...0.10.0
[0.9.2]: https://github.com/dice-group/Ontolearn/compare/0.9.1...0.9.2
[0.9.1]: https://github.com/dice-group/Ontolearn/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/dice-group/Ontolearn/compare/0.8.1...0.9.0
[0.8.1]: https://github.com/dice-group/Ontolearn/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/dice-group/Ontolearn/compare/0.7.3...0.8.0
[0.7.3]: https://github.com/dice-group/Ontolearn/compare/0.7.2...0.7.3
[0.7.2]: https://github.com/dice-group/Ontolearn/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/dice-group/Ontolearn/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/dice-group/Ontolearn/compare/0.6.1...0.7.0
[0.6.1]: https://github.com/dice-group/Ontolearn/compare/0.5.4...0.6.1
[0.5.4]: https://github.com/dice-group/Ontolearn/compare/0.5.3...0.5.4
[0.5.3]: https://github.com/dice-group/Ontolearn/compare/0.5.2...0.5.3
[0.5.2]: https://github.com/dice-group/Ontolearn/releases/tag/0.5.2
