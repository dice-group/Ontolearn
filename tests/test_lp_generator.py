import unittest
import json
from ontolearn.lp_generator import LPGen
from ontolearn.utils import setup_logging
setup_logging("ontolearn/logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
STORAGE_PATH = 'KGs/Family/new_dir'

class LPGen_Test(unittest.TestCase):
    def test_generate_load(self):
        lp_gen = LPGen(kb_path=PATH_FAMILY, storage_path=STORAGE_PATH)
        lp_gen.generate()
        with open(f"{STORAGE_PATH}/LPs.json") as file:
            lps = json.load(file)
            print("Number of learning problems:", len(lps))
        self.assertGreaterEqual(lp_gen.lp_gen.max_num_lps, len(lps))

    def test_random_seed_reproducibility(self):
        storage_path_1 = f"{STORAGE_PATH}_seed_1"
        storage_path_2 = f"{STORAGE_PATH}_seed_2"
        LPGen(kb_path=PATH_FAMILY, storage_path=storage_path_1, random_seed=1).generate()
        LPGen(kb_path=PATH_FAMILY, storage_path=storage_path_2, random_seed=1).generate()
        with open(f"{storage_path_1}/LPs.json") as file:
            lps_1 = json.load(file)
        with open(f"{storage_path_2}/LPs.json") as file:
            lps_2 = json.load(file)
        self.assertEqual(lps_1, lps_2)

if __name__ == '__main__':
    unittest.main()