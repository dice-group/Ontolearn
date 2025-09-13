from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from sentence_transformers import SentenceTransformer, util
import  numpy as np
import os
import json
from ontolearn.lp_generator import generate_data
from pathlib import Path

# kb_path = "/home/dice/Downloads/Spotify/spotify_owl/music_updated.owl"
# # kb_path = "/home/dice/Downloads/IMDB/imdb_100.owl"
#
# # where to store the generated LPs (it will create the folder if it does not exist)
# storage_path = "LPs/Music/generated_lps"
#
# # initialize
# lp_gen = generate_data.LPGen(
#     kb_path=kb_path,
#     storage_path=storage_path,
#     max_num_lps=150,        # how many learning problems you want
#     beyond_alc=True,       # if True → more expressive concepts (ALCHIQD)
#     depth=4,                # refinement depth
#     refinement_expressivity=1,  # how "rich" refinements are (between 0 and 1)
#     num_sub_roots=500,       # number of starting classes for LP generation
#     min_num_pos_examples=2,  # ensures at least 2 positives per LP
#     downsample_refinements=False
# )
#
# lp_gen.generate() # Uncomment to generate
#
#
# def add_namespace_and_downsample(input_file, output_file, namespace="http://example.org/music/", sample_size=None):
#     # Load JSON data
#     with open(input_file, "r") as f:
#         data = json.load(f)
#
#     updated_data = []
#     for concept, examples_dict in data:
#         new_examples_dict = {}
#         for key, examples in examples_dict.items():
#             # Downsample if requested
#             if sample_size:
#                 examples = examples[:sample_size]
#
#             # Add namespace to each example
#             examples = [namespace + ex for ex in examples]
#             new_examples_dict[key] = examples
#
#         updated_data.append([concept, new_examples_dict])
#
#     # Save updated data
#     with open(output_file, "w") as f:
#         json.dump(updated_data, f, indent=2)
#
#     print(f"Updated file saved at {output_file}")
#
#
# if __name__ == "__main__":
#     input_path = Path("/home/dice/Desktop/Ontolearn/LPs/Music/generated_lps/LPs.json")  # replace with your actual file
#     output_path = Path("/home/dice/Desktop/Ontolearn/LPs/Music/LPs_1.json")
#
#     add_namespace_and_downsample(
#         input_file=input_path,
#         output_file=output_path,
#         namespace="http://example.org/music/",
#         sample_size=15  # set to None if you want all
#     )









import os
import requests
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# ----------------------
# CONFIG
# ----------------------
SPARQL_ENDPOINT = "http://localhost:3030/music_10000/sparql"
BASE_URL = "http://127.0.0.1:8000"   # Persona API

# ----------------------
# 1. Query ontology for candidates
# ----------------------

def get_candidate_songs():
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX music: <http://example.org/music/>

        SELECT DISTINCT ?song ?desc ?track ?album ?genre
        WHERE {
          ?song rdf:type music:Track .
          OPTIONAL { ?song music:hasDescription ?desc . }
          OPTIONAL { ?song music:hasTrack ?track . }
          OPTIONAL { ?song music:hasAlbum ?album . }
          OPTIONAL { ?song music:hasGenre ?genre . }
        }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    songs = []
    for r in results["results"]["bindings"]:
        desc = r.get("desc", {}).get("value", "")
        track = r.get("track", {}).get("value", "")
        album = r.get("album", {}).get("value", "")
        genre = r.get("genre", {}).get("value", "")
        uri = r.get("song", {}).get("value", "")

        songs.append({
            "uri": uri,
            "track": track.split("/")[-1] if track else "",
            "album": album.split("/")[-1] if album else "",
            "genre": genre.split("/")[-1] if genre else "",
            "description": desc
        })
    return songs


def get_song_uri(track, album):
    """Query KB for the song URI given track+album (more reliable than title)."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(f"""
        PREFIX music: <http://example.org/music/>
        SELECT DISTINCT ?song
        WHERE {{
          ?song rdf:type music:Song .
          ?song music:hasTrack ?t .
          ?song music:hasAlbum ?a .
          FILTER(str(?t) = "{track}" && str(?a) = "{album}")
        }}
        LIMIT 1
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    return bindings[0]["song"]["value"] if bindings else None


# ----------------------
# 2. Compute embeddings
# ----------------------

def compute_and_save_embeddings(songs, emb_path="song_embeddings.npy",
                                texts_path="song_texts.npy", uris_path="song_uris.npy"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_texts = []
    all_uris = []
    for s in songs:
        text_to_compare = f"{s['track']} {s['album']} {s['genre']} {s['description']}"
        if not text_to_compare.strip():
            continue
        all_texts.append(text_to_compare)
        all_uris.append(s["uri"])

    all_embeddings = model.encode(all_texts, convert_to_tensor=False)

    np.save(emb_path, all_embeddings)
    np.save(texts_path, np.array(all_texts))
    np.save(uris_path, np.array(all_uris))

    print(f"Saved embeddings to {emb_path} and texts to {texts_path}")


def load_embeddings(emb_path="song_embeddings.npy",
                    texts_path="song_texts.npy",
                    uris_path="song_uris.npy"):
    embeddings = np.load(emb_path)
    texts = np.load(texts_path, allow_pickle=True)
    uris = np.load(uris_path, allow_pickle=True)
    return texts, uris, embeddings


def filter_songs_by_persona_loaded(song_texts, song_embeddings, persona_desc, song_uris, threshold=0.3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    persona_embedding = model.encode(persona_desc, convert_to_tensor=False)

    scores = util.cos_sim(persona_embedding, song_embeddings).flatten()
    filtered = [(text, uri, float(score)) for text, uri, score in zip(song_texts, song_uris, scores) if score >= threshold]
    filtered = sorted(filtered, key=lambda x: x[2], reverse=True)
    return filtered


# ----------------------
# 3. Create LPs
# ----------------------
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

def create_persona_problem(persona_name, persona_description, song_texts, song_uris, song_embeddings, top_n=10):
    filtered_results = filter_songs_by_persona_loaded(song_texts, song_embeddings, persona_description, song_uris, threshold=-1.0)

    print(f"=== {persona_name} ===")
    for text, uri, score in filtered_results[:5]:
        print(f"{score:.2f} — {text}")

    positives = [uri for _, uri, _ in filtered_results[:top_n]]
    negatives = [uri for _, uri, _ in filtered_results[-top_n:]]

    return {
        persona_name: [persona_description.strip()],
        "positives": positives,
        "negatives": negatives
    }


def build_lps_for_personas(personas, song_texts, song_uris, song_embeddings, top_n=10):
    problems = []
    for persona_name, persona_desc in personas:
        problems.append(
            create_persona_problem(persona_name, persona_desc, song_texts, song_uris, song_embeddings, top_n=top_n)
        )
    return {"problems": problems}


# ----------------------
# 4. Fetch personas
# ----------------------

def get_persona_list(n=50):
    resp = requests.get(f"{BASE_URL}/personas_random?limit={n}")
    resp.raise_for_status()
    data = resp.json()
    return [(f"Persona_{i}", p["description"]) for i, p in enumerate(data, start=1)]


# ----------------------
# MAIN
# ----------------------

if __name__ == "__main__":
    emb_path, texts_path, uris_path = "song_embeddings.npy", "song_texts.npy", "song_uris.npy"

    if os.path.exists(emb_path) and os.path.exists(texts_path):
        print("Loading saved embeddings and texts...")
        all_texts, all_uris, all_embeddings = load_embeddings(emb_path, texts_path, uris_path)
    else:
        print("Computing embeddings for the first time...")
        songs = get_candidate_songs()
        compute_and_save_embeddings(songs, emb_path, texts_path, uris_path)
        all_texts, all_uris, all_embeddings = load_embeddings(emb_path, texts_path, uris_path)

    personas = get_persona_list(50)  # e.g. 5 personas
    # personas = [
    #     ("Persona_1", """
    #    A Russian economist who specializes in macroeconomics, particularly the analysis of the Russian economy and its response to economic sanctions and oil price fluctuations. """),
    #     ("Persona_2", """
    #      A film critic passionate about historical documentaries focusing on civil rights movements
    #      and underrepresented voices.
    #      """),
    #     ("Persona_3", """ A knowledgeable and experienced gemologist who specializes in the study and identification of sapphires, particularly those from Sri Lanka.
    #     They have a deep understanding of the mineral properties, colors, and characteristics of sapphires, and can accurately identify them. They are also knowledgeable
    #      about the history and cultural significance of sapphires, and can provide insights into their origins and uses.
    #     They are likely to be involved in the business of gemstone trading, and have a keen eye for quality and authenticity.""")
    # ]
    lps = build_lps_for_personas(personas, all_texts, all_uris, all_embeddings, top_n=10)

    with open("LPs/Music/lps_personas.json", "w", encoding="utf-8") as f:
        json.dump(lps, f, indent=2, ensure_ascii=False)

    print("✅ Saved all personas to lps_personas.json")

    print(f"Generated {len(lps)} LPs:")
    # print(lps)
#
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
# #
#     # persona_list = [
#     #     ("Persona_1", """
#     #    A Russian economist who specializes in macroeconomics, particularly the analysis of the Russian economy and its response to economic sanctions and oil price fluctuations. """),
#     #     ("Persona_2", """
#     #      A film critic passionate about historical documentaries focusing on civil rights movements
#     #      and underrepresented voices.
#     #      """),
#     #     ("Persona_3", """ A knowledgeable and experienced gemologist who specializes in the study and identification of sapphires, particularly those from Sri Lanka.
#     #     They have a deep understanding of the mineral properties, colors, and characteristics of sapphires, and can accurately identify them. They are also knowledgeable
#     #      about the history and cultural significance of sapphires, and can provide insights into their origins and uses.
#     #     They are likely to be involved in the business of gemstone trading, and have a keen eye for quality and authenticity.""")
#     # ]
#
#     persona_list = get_persona_list(50)
#
#
#         # Step 3: Build problems
#     problems = []
#     for persona_name, persona_description in persona_list:
#         problems.append(
#             create_persona_problem(persona_name, persona_description, all_titles, all_movie_embeddings, top_n=10)
#         )
#
#     # Step 4: Save JSON
#     lps_data = {"problems": problems}
#     with open("LPs/IMDB/lps_personas_10.json", "w", encoding="utf-8") as f:
#         json.dump(lps_data, f, indent=2, ensure_ascii=False)
#
#     print("✅ Saved all personas to lps_personas_50.json")





#
# from ontolearn.knowledge_base import KnowledgeBase
# from ontolearn.triple_store import TripleStore
# from ontolearn.utils import jaccard_similarity, f1_set_similarity, concept_reducer, concept_reducer_properties
# from owlapy.class_expression import (
#     OWLObjectUnionOf,
#     OWLObjectIntersectionOf,
#     OWLObjectSomeValuesFrom,
#     OWLObjectAllValuesFrom,
#     OWLObjectMinCardinality,
#     OWLObjectMaxCardinality,
#     OWLObjectOneOf,
# )
# import time
# from typing import Tuple, Set
# import pandas as pd
# from owlapy import owl_expression_to_dl
# from itertools import chain
# from argparse import ArgumentParser
# import os
# import json
# from tqdm import tqdm
# import random
# import itertools
# import ast
#
# # Set pandas options to ensure full output
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.colheader_justify', 'left')
# pd.set_option('display.expand_frame_repr', False)
#
#
# def execute(args):
#     # (1) Initialize knowledge base.
#     assert os.path.isfile(args.path_kg)
#     if args.endpoint_triple_store:
#         symbolic_kb = TripleStore(url="http://localhost:3030/family")
#     else:
#         symbolic_kb = KnowledgeBase(path=args.path_kg)
#     # (2) Initialize Neural OWL Reasoner.
#     # if args.path_kge_model:
#     #     neural_owl_reasoner = TripleStoreNeuralReasoner(path_neural_embedding=args.path_kge_model, gamma=args.gamma)
#     # else:
#     #     neural_owl_reasoner = TripleStoreNeuralReasoner(path_of_kb=args.path_kg, gamma=args.gamma)
#     # Fix the random seed.
#     random.seed(args.seed)
#     ###################################################################
#     # GENERATE DL CONCEPTS TO EVALUATE RETRIEVAL PERFORMANCES
#     # (3) R: Extract object properties.
#     object_properties = sorted({i for i in symbolic_kb.get_object_properties()})
#
#     # (3.1) Subsample if required.
#     if args.ratio_sample_object_prop and len(object_properties) > 0:
#         object_properties = {i for i in random.sample(population=list(object_properties),
#                                                       k=max(1, int(len(
#                                                           object_properties) * args.ratio_sample_object_prop)))}
#
#     object_properties = set(object_properties)
#
#     # (4) R⁻: Inverse of object properties.
#     object_properties_inverse = {i.get_inverse_property() for i in object_properties}
#
#     # (5) R*: R UNION R⁻.
#     object_properties_and_inverse = object_properties.union(object_properties_inverse)
#     # (6) NC: Named owl concepts.
#     nc = sorted({i for i in symbolic_kb.get_concepts()})
#
#     if args.ratio_sample_nc and len(nc) > 0:
#         # (6.1) Subsample if required.
#         nc = {i for i in random.sample(population=list(nc), k=max(1, int(len(nc) * args.ratio_sample_nc)))}
#
#     nc = random.sample(nc, k=min(len(nc), 10))
#     nc = set(nc)  # return to a set
#     # (7) NC⁻: Complement of NC.
#     nnc = {i.get_object_complement_of() for i in nc}
#
#     # (8) NC*: NC UNION NC⁻.
#     nc_star = nc.union(nnc)
#     # (9) Retrieve 10 random Nominals.
#     if len(symbolic_kb.individuals()) > args.num_nominals:
#         nominals = set(random.sample(symbolic_kb.individuals(), args.num_nominals))
#     else:
#         nominals = symbolic_kb.individuals()
#     # (10) All combinations of 3 for Nominals, e.g. {martin, heinz, markus}
#     nominal_combinations = set(OWLObjectOneOf(combination) for combination in itertools.combinations(nominals, 3))
#
#     # (11) NC UNION NC.
#     unions = concept_reducer(nc, opt=OWLObjectUnionOf)
#     unions = set(random.sample(unions, 5))
#
#     # (12) NC INTERSECTION NC.
#     intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
#     intersections = set(random.sample(intersections, 5))
#     # (13) NC* UNION NC*.
#     unions_nc_star = concept_reducer(nc_star, opt=OWLObjectUnionOf)
#     unions_nc_star = random.sample(unions_nc_star, 10)
#     # (14) NC* INTERACTION NC*.
#     intersections_nc_star = concept_reducer(nc_star, opt=OWLObjectIntersectionOf)
#     intersections_nc_star = set(random.sample(intersections_nc_star, 10))
#     # (15) \exist r. C s.t. C \in NC* and r \in R* .
#     exist_nc_star = concept_reducer_properties(
#         concepts=nc_star,
#         properties=object_properties_and_inverse,
#         cls=OWLObjectSomeValuesFrom,
#     )
#     # exist_nc_star = set(random.sample(exist_nc_star,k=30))
#
#     # (16) \forall r. C s.t. C \in NC* and r \in R* .
#     for_all_nc_star = concept_reducer_properties(
#         concepts=nc_star,
#         properties=object_properties_and_inverse,
#         cls=OWLObjectAllValuesFrom,
#     )
#     # for_all_nc_star = set(random.sample(for_all_nc_star,k=10))
#     # (17) >= n r. C  and =< n r. C, s.t. C \in NC* and r \in R* .
#     min_cardinality_nc_star_1, min_cardinality_nc_star_2, min_cardinality_nc_star_3 = (
#         concept_reducer_properties(
#             concepts=nc_star,
#             properties=object_properties_and_inverse,
#             cls=OWLObjectMinCardinality,
#             cardinality=i,
#         )
#         for i in [1, 2, 3])
#
#     min_cardinality_nc_star_1, min_cardinality_nc_star_2, min_cardinality_nc_star_3 = set(
#         random.sample(min_cardinality_nc_star_1, 5)), random.sample(min_cardinality_nc_star_2, 5), random.sample(
#         min_cardinality_nc_star_3, 5)
#
#     max_cardinality_nc_star_1, max_cardinality_nc_star_2, max_cardinality_nc_star_3 = (
#         concept_reducer_properties(
#             concepts=nc_star,
#             properties=object_properties_and_inverse,
#             cls=OWLObjectMaxCardinality,
#             cardinality=i,
#         )
#         for i in [1, 2, 3]
#     )
#     max_cardinality_nc_star_1, max_cardinality_nc_star_2, max_cardinality_nc_star_3 = random.sample(
#         max_cardinality_nc_star_1, 5), random.sample(max_cardinality_nc_star_2, 5), random.sample(
#         max_cardinality_nc_star_3, 5)
#
#     # (18) \exist r. Nominal s.t. Nominal \in Nominals and r \in R* .
#     exist_nominals = concept_reducer_properties(
#         concepts=nominal_combinations,
#         properties=object_properties_and_inverse,
#         cls=OWLObjectSomeValuesFrom,
#     )
#     exist_nominals = random.sample(exist_nominals, 20)
#
#     ###################################################################
#
#     # Retrieval Results
#     def concept_retrieval(retriever_func, c) -> Tuple[Set[str], float]:
#         start_time = time.time()
#         return {i.str for i in retriever_func.individuals(c)}, time.time() - start_time
#
#     # () Collect the data.
#     data = []
#     # () Converted to list so that the progress bar works.
#     concepts = list(
#         chain(
#             nc,  # named concepts          (C)
#             nnc,  # negated named concepts  (\neg C)
#             unions_nc_star,  # A set of Union of named concepts and negat
#             intersections_nc_star,  #
#             exist_nc_star,
#             for_all_nc_star,
#             min_cardinality_nc_star_1, min_cardinality_nc_star_1, min_cardinality_nc_star_3,
#             max_cardinality_nc_star_1, max_cardinality_nc_star_2, max_cardinality_nc_star_3,
#             exist_nominals))
#     print("\n")
#     print("#" * 50)
#     print("Description of generated Concepts")
#     print(f"NC denotes the named concepts\t|NC|={len(nc)}")
#     print(f"NNC denotes the negated named concepts\t|NNC|={len(nnc)}")
#     print(f"|NC UNION NC|={len(unions)}")
#     print(f"|NC Intersection NC|={len(intersections)}")
#
#     print(f"NC* denotes the union of named concepts and negated named concepts\t|NC*|={len(nc_star)}")
#     print(f"|NC* UNION NC*|={len(unions_nc_star)}")
#     print(f"|NC* Intersection NC*|={len(intersections_nc_star)}")
#     print(f"|exist R* NC*|={len(exist_nc_star)}")
#     print(f"|forall R* NC*|={len(for_all_nc_star)}")
#
#     print(
#         f"|Max Cardinalities|={len(max_cardinality_nc_star_1) + len(max_cardinality_nc_star_2) + len(max_cardinality_nc_star_3)}")
#     print(
#         f"|Min Cardinalities|={len(min_cardinality_nc_star_1) + len(min_cardinality_nc_star_1) + len(min_cardinality_nc_star_3)}")
#     print(f"|exist R* Nominals|={len(exist_nominals)}")
#     print("#" * 50, end="\n\n")
#
#     # () Shuffled the data so that the progress bar is not influenced by the order of concepts.
#
#     random.shuffle(concepts)
#
#     # check if csv arleady exists and delete it cause we want to override it
#     if os.path.exists(args.path_report):
#         os.remove(args.path_report)
#     file_exists = False
#     # () Iterate over single OWL Class Expressions in ALCQIHO
#     data = dict()
#     for expression in (tqdm_bar := tqdm(concepts, position=0, leave=True)):
#         retrieval_y: Set[str]
#         runtime_y: Set[str]
#         # () Retrieve the true set of individuals and elapsed runtime.
#         retrieval_y, _ = concept_retrieval(symbolic_kb, expression)
#         positives = list(retrieval_y)
#         if not positives:
#             continue
#         positives = random.sample(positives, min(len(positives), 25))
#
#         # Negatives = all individuals - positives
#         all_individuals = symbolic_kb.individuals()
#         negatives = list(all_individuals - retrieval_y)
#
#         negatives = random.sample(negatives, min(len(negatives), 25))
#         negatives = [i.iri.str for i in negatives]
#
#         data[owl_expression_to_dl(expression)] = {
#             "positive examples": positives,
#             "negative examples": negatives
#         }
#
#     with open("LPs/Music/generated_lps/LPs.json", "w") as f:
#         json.dump(data, f, indent=2)
#
#     print(f"Built dataset with {len(data)} learning problems.")
#     print(data)
#     return data
#
#
# def get_default_arguments():
#     parser = ArgumentParser()
#     parser.add_argument("--path_kg", type=str,
#                         default="/home/dice/Downloads/Spotify/spotify_owl/music_10000_descriptions.owl")
#     parser.add_argument("--path_kge_model", type=str, default=None)
#     parser.add_argument("--endpoint_triple_store", type=str, default=None)
#     parser.add_argument("--gamma", type=float, default=0.9)
#     parser.add_argument("--seed", type=int, default=1)
#     parser.add_argument("--ratio_sample_nc", type=float, default=0.2, help="To sample OWL Classes.")
#     parser.add_argument("--ratio_sample_object_prop", type=float, default=0.1, help="To sample OWL Object Properties.")
#     parser.add_argument("--min_jaccard_similarity", type=float, default=0.0,
#                         help="Minimum Jaccard similarity to be achieve by the reasoner")
#     parser.add_argument("--num_nominals", type=int, default=10, help="Number of OWL named individuals to be sampled.")
#
#     # H is obtained if the forward chain is applied on KG.
#     parser.add_argument("--path_report", type=str, default="ALCQHI_Retrieval_Results.csv")
#     return parser.parse_args()
#
#
# if __name__ == "__main__":
#     execute(get_default_arguments())