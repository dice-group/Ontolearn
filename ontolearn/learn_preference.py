# surrogate_pref.py
import re
import math
import json
import joblib
import numpy as np
from typing import List, Tuple, Callable, Dict
from sklearn.ensemble import RandomForestRegressor

# ===============================
# Utilities for robust parsing
# ===============================

def _normalize_symbols(s: str) -> str:
    """
    Normalize common DL unicode symbols to simple ASCII placeholders
    where useful, but ALSO keep the originals so both regex styles match.
    We mainly standardize whitespace.
    """
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _max_paren_depth(s: str) -> int:
    depth = 0
    max_depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth = max(0, depth - 1)
    return max_depth

def _numbers_in(s: str) -> List[float]:
    """
    Find all floats/ints inside a string (for thresholds, counts, etc.)
    Matches '.5', '0.5', '5', '10.0'
    """
    nums = re.findall(r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)", s)
    return [float(x) for x in nums]

# ===============================
# Feature extraction for DL strings
# ===============================

FEATURE_NAMES = [
    # Constructor counts
    "count_exists",        # ∃
    "count_forall",        # ∀
    "count_and",           # ⊓
    "count_or",            # ⊔
    "count_not",           # ¬
    "count_inverse_role",  # R⁻

    # Number restrictions (≥ n, ≤ n before roles)
    "count_ge_num_restr",
    "count_le_num_restr",
    "mean_num_restr",
    "max_num_restr",
    "min_num_restr",

    # Datatype facet thresholds inside [...] (e.g., xsd:double[≥ 4.25])
    "count_datatype_facets",
    "mean_datatype_thresh",
    "max_datatype_thresh",
    "min_datatype_thresh",

    # Role usage
    "unique_roles",
    "total_roles",

    # Structure/shape
    "max_paren_depth",
    "length_chars",
]

_ROLE_PAT = re.compile(r"\bhas[A-Za-z0-9_]*\b")
_ROLE_INV_PAT = re.compile(r"\bhas[A-Za-z0-9_]*\s*[\-⁻]")  # matches 'hasXxx -' or 'hasXxx⁻'

def encode_concept_dl(concept_str: str) -> np.ndarray:
    """
    Encode DL-like concept strings into a fixed-length numeric vector.

    Handles:
      - ∃, ∀, ⊓, ⊔, ¬
      - role inverses 'R⁻'
      - number restrictions: '≥ n R.C', '≤ n R.C'
      - datatype facets: 'xsd:double[≥ 4.25]' (collect thresholds)
      - role tokens: 'hasSomething'
    """
    s = _normalize_symbols(concept_str)

    # --- Constructors (unicode) ---
    count_exists = s.count('∃')
    count_forall = s.count('∀')
    count_and    = s.count('⊓')
    count_or     = s.count('⊔')
    count_not    = s.count('¬')

    # --- Inverse roles ---
    # Count explicit inverse marker '⁻' (unicode superscript minus) or ASCII '-' attached to role
    count_inverse = len(re.findall(r"[A-Za-z0-9_]\s*⁻", s)) + len(re.findall(r"\bhas[A-Za-z0-9_]*\s*-\b", s))

    # --- Number restrictions before roles: '≥ n ...' or '≤ n ...'
    # We'll extract numbers that appear immediately after ≥ or ≤, which are usually the cardinalities.
    ge_nums = [float(x) for x in re.findall(r"≥\s*([0-9]+(?:\.[0-9]+)?)", s)]
    le_nums = [float(x) for x in re.findall(r"≤\s*([0-9]+(?:\.[0-9]+)?)", s)]

    all_num_restr = ge_nums + le_nums
    mean_num = float(np.mean(all_num_restr)) if all_num_restr else 0.0
    max_num  = float(np.max(all_num_restr)) if all_num_restr else 0.0
    min_num  = float(np.min(all_num_restr)) if all_num_restr else 0.0

    # --- Datatype facets inside [...] e.g. xsd:double[≥ 4.25]
    # Extract bracket contents and then any numbers within.
    bracket_contents = re.findall(r"\[([^\]]+)\]", s)
    dt_numbers: List[float] = []
    for bc in bracket_contents:
        dt_numbers.extend(_numbers_in(bc))

    count_dt = len(bracket_contents)
    mean_dt  = float(np.mean(dt_numbers)) if dt_numbers else 0.0
    max_dt   = float(np.max(dt_numbers)) if dt_numbers else 0.0
    min_dt   = float(np.min(dt_numbers)) if dt_numbers else 0.0

    # --- Roles (heuristic: tokens starting with 'has')
    roles = _ROLE_PAT.findall(s)
    unique_roles = len(set(roles))
    total_roles  = len(roles)

    # --- Structure
    depth = _max_paren_depth(s)
    length_chars = len(s)

    feats = np.array([
        count_exists,
        count_forall,
        count_and,
        count_or,
        count_not,
        count_inverse,

        len(ge_nums),
        len(le_nums),
        mean_num,
        max_num,
        min_num,

        count_dt,
        mean_dt,
        max_dt,
        min_dt,

        unique_roles,
        total_roles,

        depth,
        length_chars,
    ], dtype=float)

    return feats

def batch_encode_dl(concepts: List[str]) -> np.ndarray:
    return np.vstack([encode_concept_dl(c) for c in concepts])

# ===============================
# Training data creation
# ===============================

def build_training_data(
    concepts: List[str],
    expensive_pref_fn: Callable[[str], float],
    skip_errors: bool = True
) -> Tuple[List[str], List[float]]:
    """
    Use the current (expensive) preference function ONCE to label concepts.
    """
    X_concepts: List[str] = []
    y_ratings: List[float] = []
    for c in concepts:
        try:
            y = float(expensive_pref_fn(c))
            if math.isnan(y) or math.isinf(y):
                if skip_errors:
                    continue
                else:
                    y = 0.0
            X_concepts.append(c)
            y_ratings.append(y)
        except Exception as e:
            if not skip_errors:
                raise
            # skip problematic concept
            continue
    return X_concepts, y_ratings

# ===============================
# Train + save + load surrogate
# ===============================

def train_surrogate_from_concepts(
    concepts: List[str],
    expensive_pref_fn: Callable[[str], float],
    model_path: str = "surrogate_pref_model.pkl",
    n_estimators: int = 200,
    random_state: int = 42
):
    """
    End-to-end: label concepts with the expensive function, train RF regressor, save model.
    """
    Xc, y = build_training_data(concepts, expensive_pref_fn)
    if not Xc:
        raise ValueError("No valid (concept, rating) pairs were produced for training.")

    X = batch_encode_dl(Xc)
    y = np.array(y, dtype=float)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_path)
    print(f"[surrogate] Trained on {len(Xc)} concepts. Saved to {model_path}")
    return model

def load_surrogate(model_path: str = "surrogate_pref_model.pkl"):
    bundle = joblib.load(model_path)
    return bundle["model"]

class PreferenceSurrogate:
    """
    Fast drop-in replacement for the preference function.
    Usage:
        surrogate = PreferenceSurrogate("surrogate_pref_model.pkl")
        score = surrogate(concept_str)
    """
    def __init__(self, model_path: str = "surrogate_pref_model.pkl"):
        self.model = load_surrogate(model_path)

    def __call__(self, concept_str: str) -> float:
        x = encode_concept_dl(concept_str).reshape(1, -1)
        y = self.model.predict(x)[0]
        try:
            return float(y)
        except Exception:
            return float(np.nan)

# ===============================
# (Optional) tiny demo / test
# ===============================

if __name__ == "__main__":
    # Replace this stub with your real expensive function.
    # For demo, we synthesize a "hidden" function that correlates with features:
    def expensive_pref_fn_stub(c: str) -> float:
        # pretend true rating correlates with:
        # +0.2 per ∃, +0.1 per ≤ number, +0.05 per datatype threshold avg, -0.1 per ¬
        feats = encode_concept_dl(c)
        count_exists = feats[0]
        count_le = feats[7]
        mean_dt  = feats[12]
        count_not = feats[4]
        score = 0.2*count_exists + 0.1*count_le + 0.05*mean_dt - 0.1*count_not
        # clamp to [0,10] to mimic IMDb-ish scale
        return max(0.0, min(10.0, score + 6.0))

    # Example concepts (you should provide thousands from your generator)
    concepts = [
        "∃ hasRatingValue.xsd:double[≥ 4.25]",
        "≤ 9 hasPrincipal.(¬Episode)",
        "¬Movie",
        "≤ 8 hasPrincipal.(¬Series)",
        "∃ hasGenre.(Comedy ⊔ Drama)",
        "∀ hasWriter.⊤",
        "∃ hasDirector.⊤ ⊓ ¬Series",
        "≥ 2 hasPrincipal.(Person ⊓ ¬Rating)",
        "∃ hasRatingValue.xsd:double[≥ 8.3]",
        "≤ 3 hasGenre.(¬Person)",
    ]

    # Train surrogate on demo
    train_surrogate_from_concepts(concepts, expensive_pref_fn_stub, model_path="surrogate_pref_model.pkl")

    # Use surrogate
    surrogate = PreferenceSurrogate("surrogate_pref_model.pkl")
    test = "∃ hasRatingValue.xsd:double[≥ 7.5] ⊓ ≤ 2 hasGenre.(Comedy)"
    print("Predicted preference:", surrogate(test))
