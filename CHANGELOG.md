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
- Ruff upgraded to `>=0.16.0`. (#594)
- `TDL`/`TDL_refinement` data-property filtering extended to exclude boolean
  data properties, with a toggle to enable/disable boolean data properties.
  (#588)
- `max_runtime` for TDL feature matrix creation/extraction changed to `None`
  by default, matching the other learners. (#585)

### Fixed
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
