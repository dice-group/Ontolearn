import unittest
import json
from ontolearn.lp_generator import LPGen
from ontolearn.utils import setup_logging
setup_logging("ontolearn/logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
STORAGE_DIR = 'KGs/Family/new_dir'

class LPGen_Test(unittest.TestCase):
    def test_generate_load(self):
        lp_gen = LPGen(kb_path=PATH_FAMILY, storage_dir=STORAGE_DIR)
        lp_gen.generate()
        print("Loading generated data...")
        with open(f"{STORAGE_DIR}/triples/train.txt") as file:
            triples_data = file.readlines()
            print("Number of triples:", len(triples_data))
        with open(f"{STORAGE_DIR}/LPs.json") as file:
            lps = json.load(file)
            print("Number of learning problems:", len(lps))
        self.assertGreaterEqual(lp_gen.lp_gen.max_num_lps, len(lps))

if __name__ == '__main__':
    unittest.main()