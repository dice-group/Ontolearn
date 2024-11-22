from ontolearn.lp_generator import LPGen

PATH = 'https://dbpedia.data.dice-research.org/sparql'
STORAGE_DIR = 'DBpedia_LPs'

def generate_lps():
    lp_gen = LPGen(kb_path=PATH, storage_dir=STORAGE_DIR, refinement_expressivity=1e-7, use_triple_store=True, sample_fillers_count=1, num_sub_roots=1)
    lp_gen.generate()

if __name__ == '__main__':
    generate_lps()