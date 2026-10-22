"""Split the test files across CI matrix jobs.

Takes the output of ``select_tests.py`` (``ALL``, ``NONE`` or a space-separated
list of test files) as argv and prints a JSON list of ``{"group", "tests"}``
entries, one per group that has tests to run, for use as a GitHub Actions
matrix. Every test file belongs to exactly one group: ``neural`` and
``triplestore`` are matched by pattern, ``learners`` is everything else, so a
new test file is never left out.
"""
import fnmatch
import glob
import json
import sys

# Groups that are matched by pattern, in priority order. Tests of the
# triplestore group need the Fuseki service; the neural group is the slow,
# torch-based one.
GROUPS = [
    ("triplestore", ["tests/test_triplestore*.py"]),
    ("neural", ["tests/test_nces*.py", "tests/test_roces.py", "tests/test_clip*.py",
                "tests/test_nero*.py", "tests/test_drill.py"]),
]
DEFAULT_GROUP = "learners"


def group_of(test):
    for name, patterns in GROUPS:
        if any(fnmatch.fnmatch(test, p) for p in patterns):
            return name
    return DEFAULT_GROUP


def split(selection, all_tests):
    if selection == "NONE":
        return []
    tests = sorted(all_tests) if selection == "ALL" else selection.split()
    grouped = {}
    for test in tests:
        grouped.setdefault(group_of(test), []).append(test)
    return [{"group": g, "tests": " ".join(t)} for g, t in sorted(grouped.items())]


if __name__ == "__main__":
    print(json.dumps(split(" ".join(sys.argv[1:]).strip() or "ALL", glob.glob("tests/test_*.py"))))
