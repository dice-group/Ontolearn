# Changelog

All notable changes to Ontolearn are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

When cutting a release, move the relevant `[Unreleased]` entries under a new
`## [x.y.z] - YYYY-MM-DD` heading (matching the PyPI/GitHub release version and
date) and use that section as the basis for the GitHub release notes.

## [Unreleased]

### Removed
- `ontolearn/tentris.py`, a stale, self-marked-TODO script unused anywhere
  else in the codebase. It imported `ontolearn.base`, a module removed back
  in the owlapy 1.1.0 migration, so it hadn't been importable for a long
  time. (#599)

### Added
- Example for generating learning problems with numeric data properties
  (`examples/generate_numeric_lps.py`). (#593)
- Explicit `ruff.toml` pinning the enforced lint rule set (`E4`, `E7`, `E9`,
  `F`) instead of relying on ruff's shipped default. (#594)

### Changed
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
- Ruff upgraded to `>=0.16.0`. (#594)
- `TDL`/`TDL_refinement` data-property filtering extended to exclude boolean
  data properties, with a toggle to enable/disable boolean data properties.
  (#588)
- `max_runtime` for TDL feature matrix creation/extraction changed to `None`
  by default, matching the other learners. (#585)

### Fixed
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

[Unreleased]: https://github.com/dice-group/Ontolearn/compare/0.10.0...develop
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
