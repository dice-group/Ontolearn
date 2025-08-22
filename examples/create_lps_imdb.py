from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from sentence_transformers import SentenceTransformer, util
import  numpy as np
import os
import json
from ontolearn.lp_generator import generate_data
from pathlib import Path

kb_path = "/home/dice/Downloads/IMDB/imdb_10000.owl"

# where to store the generated LPs (it will create the folder if it does not exist)
storage_path = "LPs/IMDB/generated_lps"

# initialize
lp_gen = generate_data.LPGen(
    kb_path=kb_path,
    storage_path=storage_path,
    max_num_lps=50,        # how many learning problems you want
    beyond_alc=True,       # if True → more expressive concepts (ALCHIQD)
    depth=4,                # refinement depth
    refinement_expressivity=0.5,  # how "rich" refinements are (between 0 and 1)
    num_sub_roots=250,       # number of starting classes for LP generation
    min_num_pos_examples=2  # ensures at least 2 positives per LP
)

lp_gen.generate() # Uncomment to generate


def add_namespace_and_downsample(input_file, output_file, namespace="http://example.org/imdb/", sample_size=None):
    # Load JSON data
    with open(input_file, "r") as f:
        data = json.load(f)

    updated_data = []
    for concept, examples_dict in data:
        new_examples_dict = {}
        for key, examples in examples_dict.items():
            # Downsample if requested
            if sample_size:
                examples = examples[:sample_size]

            # Add namespace to each example
            examples = [namespace + ex for ex in examples]
            new_examples_dict[key] = examples

        updated_data.append([concept, new_examples_dict])

    # Save updated data
    with open(output_file, "w") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Updated file saved at {output_file}")


if __name__ == "__main__":
    input_path = Path("/home/dice/Desktop/Ontolearn/LPs/IMDB/generated_lps/LPs.json")  # replace with your actual file
    output_path = Path("/home/dice/Desktop/Ontolearn/LPs/IMDB/LPs.json")

    add_namespace_and_downsample(
        input_file=input_path,
        output_file=output_path,
        namespace="http://example.org/imdb/",
        sample_size=25  # set to None if you want all
    )










# Generate LPs using personas
# ----------------------
# CONFIG
# ----------------------
# SPARQL_ENDPOINT = "http://localhost:3030/imdb_10000/sparql"
# OMDB_API_KEY = "df1522cd"
# PERSONA_DESCRIPTION = """A skilled machinist or toolmaker who specializes in the use of high carbon steel,
# particularly in the formation of hardened steel articles. They are proficient in the use of heat treatments
# such as quenching and tempering to achieve the desired properties of the finished articles.
# They are also knowledgeable about the formation of metastable Martensite during quenching and the reduction
# of the fraction of this to the desired amount during tempering.
# They are likely to be interested in articles such as tools, machine parts, and forming and machining processes.
# """
#
# # ----------------------
# # 1. Query ontology for candidates
# # ----------------------
# def get_candidate_movies():
#     sparql = SPARQLWrapper(SPARQL_ENDPOINT)
#     sparql.setQuery("""
#         PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#         PREFIX imdb: <http://example.org/imdb/>
#
#         SELECT DISTINCT ?movie ?title
#         WHERE {
#           ?movie rdf:type imdb:Movie .
#           ?movie imdb:hasTitle ?title .
#         }
#     """)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
#     a = [r["title"]["value"] for r in results["results"]["bindings"]]
#     return a
#
#
# def get_movie_uri_by_title(title):
#     """Query KB for the movie URI matching a given title."""
#     sparql = SPARQLWrapper(SPARQL_ENDPOINT)
#     sparql.setQuery(f"""
#         PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#         PREFIX imdb: <http://example.org/imdb/>
#
#         SELECT DISTINCT ?movie
#         WHERE {{
#           ?movie rdf:type imdb:Movie .
#           ?movie imdb:hasTitle ?title .
#           FILTER (lcase(str(?title)) = lcase("{title}"))
#         }}
#         LIMIT 1
#     """)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
#     bindings = results["results"]["bindings"]
#     return bindings[0]["movie"]["value"] if bindings else None
#
# # ----------------------
# # 2. Get metadata from OMDb
# # ----------------------
# def get_movie_info(title):
#     url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
#     r = requests.get(url)
#     if r.status_code == 200:
#         data = r.json()
#         if data.get("Response") == "True":
#             return {
#                 "title": data["Title"],
#                 "plot": data.get("Plot", ""),
#                 "genre": data.get("Genre", "")
#             }
#     return None
#
#
#
# # ----------------------
# # 3. Filter using embeddings
# # ----------------------
#
# def compute_and_save_embeddings(movies, emb_path="movie_embeddings.npy", texts_path="movie_texts.npy", titles_path="title_texts.npy"):
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#
#     all_text_to_compare = []
#     all_title = []
#     for movie in movies:
#         if not movie:
#             continue
#         text_to_compare = f"{movie['title']} {movie.get('genre', '')} {movie.get('plot', '')}"
#         if "N/A" in text_to_compare:
#             continue
#         all_text_to_compare.append(text_to_compare)
#         all_title.append(movie["title"])
#
#     all_movie_embeddings = model.encode(all_text_to_compare, convert_to_tensor=False)  # numpy array
#
#     # Save embeddings and texts
#     np.save(emb_path, all_movie_embeddings)
#     np.save(texts_path, np.array(all_text_to_compare))
#     np.save(titles_path, np.array(all_title))
#
#     print(f"Saved embeddings to {emb_path} and texts to {texts_path}")
#
#
# def load_embeddings(emb_path="movie_embeddings.npy", texts_path="movie_texts.npy", titles_path="title_texts.npy"):
#     all_movie_embeddings = np.load(emb_path)
#     all_text_to_compare = np.load(texts_path, allow_pickle=True)
#     all_titles = np.load(titles_path, allow_pickle=True)
#     return all_text_to_compare, all_titles, all_movie_embeddings
#
# def filter_movies_by_persona_loaded(movies_texts, movies_embeddings, persona_desc, threshold=0.3):
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     persona_embedding = model.encode(persona_desc, convert_to_tensor=False)
#
#     scores = util.cos_sim(persona_embedding, movies_embeddings).flatten()
#     filtered = [(text, float(score)) for text, score in zip(movies_texts, scores) if score >= threshold]
#     filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
#
#     return filtered
#
#
# # ----------------------
# # Create LPs
# # ----------------------
#
# def create_persona_problem(persona_name, persona_description, all_titles, all_movie_embeddings, top_n=10):
#     """Compute scores, pick top/bottom, return LPS problem entry."""
#     filtered_results = filter_movies_by_persona_loaded(
#         all_titles, all_movie_embeddings, persona_description, threshold=-1.
#     )
#     print(f"{persona_name}")
#     for text, score in filtered_results[:10]:
#         print(f"{score:.2f} — {text}")
#
#     # Top N
#     positives = [get_movie_uri_by_title(text) for text, _ in filtered_results[:top_n]]
#     # Bottom N
#     negatives = [get_movie_uri_by_title(text) for text, _ in filtered_results[-top_n:]]
#
#     positives = [p for p in positives if p]
#     negatives = [n for n in negatives if n]
#
#     return {
#         persona_name: [persona_description.strip()],
#         "positives": positives,
#         "negatives": negatives
#     }
#
# def build_lps_for_personas(personas, movies, top_n=10):
#     """
#     personas: list of tuples (persona_name, persona_description)
#     movies: [(title, score), ...] for each persona (can vary)
#     """
#     problems = []
#     for persona_name, persona_desc, filtered_results in personas:
#         problems.append(create_persona_problem(
#             persona_name, persona_desc, filtered_results, top_n=top_n
#         ))
#     return {"problems": problems}
#
# # ----------------------
# # MAIN
# # ----------------------
# if __name__ == "__main__":
#     emb_path = "movie_embeddings.npy"
#     texts_path = "movie_texts.npy"
#     titles_path = "title_texts.npy"
#
#     if os.path.exists(emb_path) and os.path.exists(texts_path):
#         print("Loading saved embeddings and texts...")
#         all_text_to_compare, all_titles, all_movie_embeddings = load_embeddings(emb_path, texts_path, titles_path)
#     else:
#         print("Computing embeddings for the first time...")
#         titles = get_candidate_movies()
#         enriched = [get_movie_info(t) for t in titles]
#         compute_and_save_embeddings(enriched, emb_path, texts_path)
#         all_text_to_compare, all_titles, all_movie_embeddings = load_embeddings(emb_path, texts_path, titles_path)
#
#     # filtered_results = filter_movies_by_persona_loaded(all_titles, all_movie_embeddings, PERSONA_DESCRIPTION,
#     #                                                    threshold=-1.)
#     #
#     # for text, score in filtered_results[:10]:
#     #     print(f"{score:.2f} — {text}")
#
#     persona_list = [
#         ("Persona_1", """
#        A Russian economist who specializes in macroeconomics, particularly the analysis of the Russian economy and its response to economic sanctions and oil price fluctuations. """),
#         ("Persona_2", """
#          A film critic passionate about historical documentaries focusing on civil rights movements
#          and underrepresented voices.
#          """),
#         ("Persona_3", """ A knowledgeable and experienced gemologist who specializes in the study and identification of sapphires, particularly those from Sri Lanka.
#         They have a deep understanding of the mineral properties, colors, and characteristics of sapphires, and can accurately identify them. They are also knowledgeable
#          about the history and cultural significance of sapphires, and can provide insights into their origins and uses.
#         They are likely to be involved in the business of gemstone trading, and have a keen eye for quality and authenticity.""")
#     ]
#
#
#         # Step 3: Build problems
#     problems = []
#     for persona_name, persona_description in persona_list:
#         problems.append(
#             create_persona_problem(persona_name, persona_description, all_titles, all_movie_embeddings, top_n=50)
#         )
#
#     # Step 4: Save JSON
#     lps_data = {"problems": problems}
#     with open("LPs/IMDB/lps_personas.json", "w", encoding="utf-8") as f:
#         json.dump(lps_data, f, indent=2, ensure_ascii=False)
#
#     print("✅ Saved all personas to lps_personas.json")












































# from SPARQLWrapper import SPARQLWrapper, JSON
# import json
# from collections import defaultdict
#
# endpoint_url = "http://localhost:3030/imdb_10000/sparql"
# rating_property = "http://example.org/imdb/hasRatingValue"
# sample_size = 1  # Number of positive and negative examples
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
#         return results["results"]["bindings"]  #[r["movie"]["value"] for r in results["results"]["bindings"]]
#     except Exception as e:
#         print("⚠️ Failed to parse JSON response")
#         print("Query was:\n", query)
#         print("Exception:", e)
#         raw = sparql.query().response.read()
#         print("Raw response (first 500 chars):\n", raw[:500])
#         return []
#
#
# def get_movies_by_rating_cap(rating_property, comparison_op, bound, cap=10):
#     # Comparison_op: either ">=" or "<"
#     query = f"""
#     SELECT DISTINCT ?movie ?rating WHERE {{
#       ?movie <{rating_property}> ?rating .
#       FILTER (?rating {comparison_op} {bound})
#     }}
#     """
#     results = run_query(query)
#
#     # Group by rounded rating
#     rating_groups = defaultdict(list)
#     for res in results:
#         movie_uri = res["movie"]["value"]
#         rating_val = float(res["rating"]["value"])
#         rounded_rating = round(rating_val, 1)
#         rating_groups[rounded_rating].append(movie_uri)
#
#     # Cap the number per rating bin
#     capped_movies = []
#     for rating_val in sorted(rating_groups.keys(), reverse=(comparison_op == ">=")):
#         selected = rating_groups[rating_val][:cap]
#         print(f"Rating {rating_val}: selected {len(selected)} out of {len(rating_groups[rating_val])} movies")
#         capped_movies.extend(selected)
#
#     return capped_movies
#
#
# learning_problems = []
# j=0
# for i in [90, 75, 55, 40, 30]:
#     t = i / 10.0
#     j+=1
#     print(f"🔍 Generating learning problem for threshold t = {t:.1f}")
#
#     if i>=55:
#         positives = get_movies_by_rating_cap(rating_property, ">=", t, cap=10)
#         negatives = get_movies_by_rating_cap(rating_property, "<", t, cap=3)
#     else:
#         positives = get_movies_by_rating_cap(rating_property, ">=", t, cap=3)
#         negatives = get_movies_by_rating_cap(rating_property, "<", t, cap=10)
#     # Make sure there's a reasonable balance
#     if len(positives) >= 20 and len(negatives) >= 20:
#         learning_problems.append({
#             "threshold": round(t, 1),
#             "positives": positives,
#             "negatives": negatives
#         })
#         print(f"✅ Added LP with threshold {t:.1f} | Positives: {len(positives)} | Negatives: {len(negatives)}")
#     else:
#         print(f"❌ Skipped t={t:.1f} | Only {len(positives)} positives and {len(negatives)} negatives")
#
# # Save to file
# with open("LPs/IMDB/learning_problems.json", "w") as f:
#     json.dump({"problems": learning_problems}, f, indent=2)
#
#
# print(f"✅ {len(learning_problems)} Learning problems saved to 'learning_problems.json'")
#
# import json
# from SPARQLWrapper import SPARQLWrapper, JSON
#
#
#
#
# # CONFIGURATION
# endpoint_url = "http://localhost:3040/imdb_1000/sparql"      #"http://localhost:3030/imdb/sparql"
# rating_property = "http://example.org/imdb/hasRatingValue"     #"https://www.imdb.com/averageRating"
# sample_size = 1000 # Number of postive and negative examples to be sampled
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
# good_ratings = [9.0, 8.5, 8.0, 7.8, 7.5]  #{9.0: 11 movies, 8.5: 37 movies, 8.0: 87 movies, 7.8: 117 movies, 7.5: 185 movies}
# bad_ratings = [2.7, 3.4, 4.2, 4.5 ,5.0]  #{2.7: 10 movies, 3.4: 38 movies, 4.2: 84 movies, 4.5: 119 movies, 5.0: 187 movies}
#
# for i in range(len(good_ratings)):
#
#     print(f"Generating learning problem for threshold t = {good_ratings[i]:.1f}")
#
#     # Positive examples (rating ≥ t)
#     query_pos = f"""
#     SELECT DISTINCT ?movie WHERE {{
#       ?movie <{rating_property}> ?rating .
#       FILTER (?rating >= {good_ratings[i]})
#     }}
#     LIMIT {sample_size}
#     """
#     positives = run_query(query_pos)
#
#     # Negative examples (rating < t)
#     query_neg = f"""
#     SELECT DISTINCT ?movie WHERE {{
#       ?movie <{rating_property}> ?rating .
#       FILTER (?rating <= {bad_ratings[i]})
#     }}
#     LIMIT {sample_size}
#     """
#
#     negatives = run_query(query_neg)
#
#
#
#     if positives and negatives:
#         learning_problems.append({
#             "threshold": round(good_ratings[i], 1),
#             "positives": positives,
#             "negatives": negatives
#         })
#
#     # Save to JSON
#     problem = {
#         "threshold": good_ratings[i],
#         "positives": positives,
#         "negatives": negatives
#     }
#
#   # Save to a single JSON file
# with open("LPs/IMDB/learning_problems.json", "w") as f:
#     json.dump({"problems": learning_problems}, f, indent=2)
#
# print("✅ Learning problems saved to 'learning_problems.json'")
