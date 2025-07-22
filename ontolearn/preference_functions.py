from owlapy import owl_expression_to_sparql, owl_expression_to_dl
from owlapy.class_expression import OWLClassExpression
import requests

def preference_score_utility_based(concept: OWLClassExpression, url:str="http://localhost:3030/imdb_1000/sparql") -> float:
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
        return sum(ratings) / (len(ratings) + len(query)) #Penalize long concepts
    else:
        return 0.0