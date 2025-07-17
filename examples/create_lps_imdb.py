from SPARQLWrapper import SPARQLWrapper, JSON
import json

endpoint_url = "http://localhost:3030/imdb_1000/sparql"
rating_property = "http://example.org/imdb/hasRatingValue"
sample_size = 20  # Number of positive and negative examples

# Set up SPARQL endpoint
sparql = SPARQLWrapper(endpoint_url)
sparql.setReturnFormat(JSON)

learning_problems = []

def run_query(query):
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return [r["movie"]["value"] for r in results["results"]["bindings"]]
    except Exception as e:
        print("⚠️ Failed to parse JSON response")
        print("Query was:\n", query)
        print("Exception:", e)
        raw = sparql.query().response.read()
        print("Raw response (first 500 chars):\n", raw[:500])
        return []

# Iterate over thresholds from 9.0 to 8.6
for i in range(90, 65, -1):
    t = i / 10.0
    print(f"Generating learning problem for threshold t = {t:.1f}")

    # Positive examples (rating == t)
    query_pos = f"""
    SELECT DISTINCT ?movie WHERE {{
      ?movie <{rating_property}> ?rating .
      FILTER (?rating = {t})
    }}
    LIMIT {sample_size}
    """
    positives = run_query(query_pos)

    # Negative examples (rating < t)
    query_neg = f"""
    SELECT DISTINCT ?movie WHERE {{
      ?movie <{rating_property}> ?rating .
      FILTER (?rating < {t})
    }}
    LIMIT {sample_size}
    """
    negatives = run_query(query_neg)

    # Only accept if both lists are full and balanced
    if len(positives) == sample_size and len(negatives) == sample_size:
        learning_problems.append({
            "threshold": round(t, 1),
            "positives": positives,
            "negatives": negatives
        })
        print(f"✅ Added LP with threshold {t:.1f}")
    else:
        print(f"❌ Skipped t={t:.1f} | Found {len(positives)} positives and {len(negatives)} negatives")

# Save all problems to a JSON file
with open("LPs/IMDB/learning_problems.json", "w") as f:
    json.dump({"problems": learning_problems}, f, indent=2)

print("✅ Learning problems saved to 'learning_problems.json'")





# import json
# from SPARQLWrapper import SPARQLWrapper, JSON
#
#
#
#
# # CONFIGURATION
# endpoint_url = "http://localhost:3030/imdb_1000/sparql"      #"http://localhost:3030/imdb/sparql"
# rating_property = "http://example.org/imdb/hasRatingValue"     #"https://www.imdb.com/averageRating"
# sample_size = 30 # Number of postive and negative examples to be sampled
#
# # Set up SPARQL endpoint
# sparql = SPARQLWrapper(endpoint_url)
# sparql.setReturnFormat(JSON)
#
# learning_problems = []
#
# def run_query(query):
#     sparql.setQuery(query)
#     try:
#         results = sparql.query().convert()
#         return [r["movie"]["value"] for r in results["results"]["bindings"]]
#     except Exception as e:
#         print("⚠️ Failed to parse JSON response")
#         print("Query was:\n", query)
#         print("Exception:", e)
#         raw = sparql.query().response.read()
#         print("Raw response (first 500 chars):\n", raw[:500])
#         return []
#
# for i in range(90, 85, -1):  # Average rating From 2.0 to 1.1
#     t = i / 10.0
#     print(f"Generating learning problem for threshold t = {t:.1f}")
#
#     # Positive examples (rating ≥ t)
#     query_pos = f"""
#     SELECT DISTINCT ?movie WHERE {{
#       ?movie <{rating_property}> ?rating .
#       FILTER (?rating = {t})
#     }}
#     LIMIT {sample_size}
#     """
#     positives = run_query(query_pos)
#
#     # Negative examples (rating < t)
#     query_neg = f"""
#     SELECT DISTINCT ?movie WHERE {{
#       ?movie <{rating_property}> ?rating .
#       FILTER (?rating < {t})
#     }}
#     LIMIT {sample_size}
#     """
#     negatives = run_query(query_neg)
#
#     if positives and negatives:
#         learning_problems.append({
#             "threshold": round(t, 1),
#             "positives": positives,
#             "negatives": negatives
#         })
#
#     # Save to JSON
#     problem = {
#         "threshold": t,
#         "positives": positives,
#         "negatives": negatives
#     }
#
#   # Save to a single JSON file
# with open("LPs/IMDB/learning_problems.json", "w") as f:
#     json.dump({"problems": learning_problems}, f, indent=2)
#
# print("✅ Learning problems saved to 'learning_problems.json'")
