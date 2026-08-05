# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""Optional owlready2 dependency shim.

owlapy no longer depends on owlready2 at install time (it's an optional extra,
``pip install owlapy[owlready2]`` -- see dice-group/owlapy#205), and Ontolearn doesn't
declare a direct dependency on it either. ``ontolearn.incomplete_kb`` and
``ontolearn.semantic_caching`` still use owlready2 directly, so they import it lazily through
``import_owlready2()`` instead of at module load, raising a clear, actionable error only when
code actually tries to use it.
"""


def import_owlready2():
    """Import ``owlready2`` if installed, else raise a clear, actionable ``ImportError``."""
    try:
        import owlready2
        return owlready2
    except ImportError as e:
        raise ImportError(
            "owlready2 is required for this operation but is not installed. owlready2 is an "
            "optional dependency of owlapy -- install it with `pip install owlready2>=0.40` or "
            "`pip install owlapy[owlready2]`."
        ) from e
