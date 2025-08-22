from owlapy import owl_expression_to_sparql, owl_expression_to_dl
from owlapy.class_expression import OWLClassExpression
import requests
import random
import os
import pickle
import json
from pathlib import Path



def preference_score_cached(concept):
    PREF_CACHE_FILE = Path("preference_cache.json")

    # Try to load cache
    preference_cache = {}
    if PREF_CACHE_FILE.exists():
        try:
            with open(PREF_CACHE_FILE, "r") as f:
                preference_cache = json.load(f)
        except json.JSONDecodeError:
            # File exists but empty or corrupt → start fresh
            preference_cache = {}

    concept_str = owl_expression_to_dl(concept)

    # Check if cached
    if concept_str in preference_cache:
        # print(f"concept {concept_str} found in cache")
        return preference_cache[concept_str]

    # Otherwise compute and store
    print(f"concept {concept_str} not found in cache")

    score = preference_score_utility_based(concept)
    preference_cache[concept_str] = score

    # Persist
    with open(PREF_CACHE_FILE, "w") as f:
        json.dump(preference_cache, f, indent=2)

    return score



def preference_score_utility_based(concept: OWLClassExpression, url:str="http://localhost:3030/imdb_10000/sparql") -> float:
    """
    Compute preference score as the average imdb:hasRatingValue of individuals
    in the extension of the OWL class expression `concept`.
    """
    # Step 1: Translate OWL concept to SPARQL filter block (this function should exist already)
    subquery = owl_expression_to_sparql(concept)  # full SPARQL query: SELECT DISTINCT ?x WHERE { ... }

    query = f"""
        PREFIX imdb: <http://example.org/imdb/>

        SELECT ?x ?rating
        WHERE {{
            {{
                {subquery}
            }}
            ?x imdb:hasRatingValue ?rating .
        }}
        """

    try:
        response = requests.Session().post(url, data={"query": query})
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
    except Exception as e:
        print(f"[ERROR] SPARQL query failed: {e}")
        print("Query was:\n", query)
        exit(0)
        return 0.0

    # Step 3: Extract ratings
    ratings = []
    for row in bindings:
        try:
            rating_str = row["rating"]["value"]
            rating = float(rating_str)
            ratings.append(rating)
        except (KeyError, ValueError):
            continue

    # Step 4: Aggregate
    if ratings:
        return sum(ratings) / len(ratings)
    else:
        return 0.0