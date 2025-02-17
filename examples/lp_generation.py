import json
from ontolearn.lp_generator import LPGen

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
STORAGE_PATH = 'Family_LPs'

def generate_lps():
    lp_gen = LPGen(kb_path=PATH_FAMILY, storage_path=STORAGE_PATH, rho_name="ELRefinement", max_num_lps=500) # max_num_lps is the maximum number of concepts you will generate.
    lp_gen.generate()
    with open(f"{STORAGE_PATH}/LPs.json") as file:
        lps = json.load(file)
        print("Number of learning problems:", len(lps))
    print(f"Leaning problems stored at {STORAGE_PATH}")

if __name__ == '__main__':
    generate_lps()