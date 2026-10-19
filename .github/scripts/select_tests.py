"""Map files changed in a pull request to the pytest targets that need to run.

Reads changed file paths (one per line) from stdin and prints either
``ALL`` (run the full suite), ``NONE`` (nothing test-relevant changed) or a
space-separated list of pytest targets.

Only modules that are isolated leaves (a single learner and its private
helpers) are mapped to a subset of tests. Any changed file that is not
explicitly mapped or ignored is treated as shared code and triggers ``ALL``,
so a missing rule can only cost time, never skip needed tests.
"""
import fnmatch
import sys

# Files whose change cannot affect any test.
IGNORED = ["docs/*", "*.md", "LICENSE", ".github/workflows/docs.yml", ".gitignore"]

# (path pattern, test file patterns). fnmatch's "*" also matches "/".
RULES = [
    (["ontolearn/learners/nces.py", "ontolearn/learners/nces2.py", "ontolearn/learners/roces.py",
      "ontolearn/base_nces.py", "ontolearn/nces_*.py"], ["tests/test_nces*.py", "tests/test_roces.py"]),
    (["ontolearn/learners/clip.py", "ontolearn/clip_*.py"], ["tests/test_clip*.py"]),
    (["ontolearn/learners/nero.py", "ontolearn/nero_*.py"], ["tests/test_nero_simple.py"]),
    (["ontolearn/learners/drill.py"], ["tests/test_drill.py"]),
    (["ontolearn/learners/evolearner.py", "ontolearn/ea_*.py", "ontolearn/fitness_functions.py"],
     ["tests/test_evolearner.py"]),
    (["ontolearn/learners/tree_learner*.py"], ["tests/test_tdl*.py"]),
    (["ontolearn/learners/alcsat.py", "ontolearn/learners/sat_base.py", "ontolearn/learners/spell.py",
      "ontolearn/learners/spell_kit/*"], ["tests/test_sat_based_learners.py"]),
    (["ontolearn/semantic_caching.py"], ["tests/test_semantic_cache.py"]),
    (["ontolearn/verbalizer.py"], ["tests/test_verbalizer.py"]),
    (["ontolearn/incomplete_kb.py"], ["tests/test_incomplete_kb.py"]),
]

TEST_FILE = "tests/test_*.py"


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def select(changed, all_tests):
    targets = set()
    for path in changed:
        if matches(path, IGNORED):
            continue
        if matches(path, [TEST_FILE]):
            targets.add(path)  # a changed test file always runs itself
            continue
        for sources, tests in RULES:
            if matches(path, sources):
                for pattern in tests:
                    targets.update(t for t in all_tests if fnmatch.fnmatch(t, pattern))
                break
        else:
            return "ALL"
    existing = sorted(t for t in targets if t in all_tests)  # deleted tests are skipped
    return " ".join(existing) if existing else "NONE"


if __name__ == "__main__":
    import glob

    changed = [line.strip() for line in sys.stdin if line.strip()]
    print(select(changed, glob.glob(TEST_FILE)))
