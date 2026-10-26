import json
import shutil
import subprocess
import sys
import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from ontolearn.semantic_caching import run_semantic_cache, run_non_semantic_cache


def check_cuda():
    if torch.cuda.is_available():
        print("GPU detected. Setting CUDA_VISIBLE_DEVICES=0")
    else:
         print("No GPU detected. Running on CPU.")

check_cuda()

class TestSemanticCache:
    # Only the symbolic (HermiT) reasoner is exercised here; the neural (EBR) path is in TestSemanticCacheNeural.
    def setup_method(self):
        self.path_kg = "KGs/Family/father.owl" #path to the father datasets
        self.path_kge = None
        self.symbolic_reasoner = "HermiT"
        self.num_concepts = 800
        self.cache_size = int(0.8*self.num_concepts)
        self.eviction = "LRU"
        self.cache_type = "cold"

    def run_cache_tests(self, cache_semantic, cache_non_semantic):
        assert cache_semantic["hit_ratio"] >= cache_non_semantic["hit_ratio"], f"Expected semantic caching to have higher hit ratio, but got {cache_semantic['hit_ratio']} vs {cache_non_semantic['hit_ratio']}"
        assert cache_semantic["miss_ratio"] <= cache_non_semantic["miss_ratio"], f"Expected semantic caching to have lower miss ratio, but got {cache_semantic['miss_ratio']} vs {cache_non_semantic['miss_ratio']}"

    def test_jaccard(self):

        cache_symbolic,_ = run_semantic_cache(self.path_kg, self.path_kge, self.cache_size, self.symbolic_reasoner, self.eviction, 0, self.cache_type, True)

        assert float(cache_symbolic["avg_jaccard"]) >= float(cache_symbolic["avg_jaccard_reas"]), "Expected average Jaccard similarity to be at least as good as reasoner-based retrieval."


    def test_cache_methods(self):
        for reasoner in [self.symbolic_reasoner]:
            cache_semantic,_ = run_semantic_cache(self.path_kg, self.path_kge, self.cache_size, reasoner, self.eviction, 0, self.cache_type, True)
            cache_non_semantic,_ = run_non_semantic_cache(self.path_kg, self.path_kge, self.cache_size, reasoner, True)
            self.run_cache_tests(cache_semantic, cache_non_semantic)

    def test_cache_size(self):
        cache_large,_ = run_semantic_cache(self.path_kg, self.path_kge, self.cache_size, self.symbolic_reasoner, self.eviction, 0, self.cache_type, True)

        for k in [0.1, 0.2]:
            cache_small,_ = run_semantic_cache(self.path_kg, self.path_kge, int(k * self.num_concepts), self.symbolic_reasoner, self.eviction, 0, self.cache_type, True)
            assert cache_small["hit_ratio"] <= cache_large["hit_ratio"], f"Expected hit ratio to increase with cache size, but got {cache_small['hit_ratio']} vs {cache_large['hit_ratio']}"
            assert cache_small["miss_ratio"] >= cache_large["miss_ratio"], f"Expected miss ratio to decrease with cache size, but got {cache_small['miss_ratio']} vs {cache_large['miss_ratio']}"

    def test_eviction_strategy(self):
        eviction_strategies = ["LRU", "FIFO", "LIFO", "MRU", "RP"]
        results = {strategy: float(run_semantic_cache(self.path_kg, self.path_kge, self.cache_size, self.symbolic_reasoner, strategy, 10, self.cache_type, True)[0]["hit_ratio"]) for strategy in eviction_strategies}

        for strategy, hit_ratio in results.items():
            assert isinstance(hit_ratio, float), f"Hit ratio for {strategy} should be a float, but got {type(hit_ratio)}"

        best_strategy = max(results, key=results.get)
        assert best_strategy == "LRU", f"Expected LRU to be the best, but got {best_strategy}"

        assert results, "No results were generated, possibly due to a failure in the cache evaluation process."
        for strategy, hit_ratio in results.items():
            assert 0.0 <= hit_ratio <= 1.0, f"Hit ratio for {strategy} is out of bounds: {hit_ratio}"


class TestSemanticCacheNeural:
    """Neural (EBR) retrieval, training a KGE on a copy of father.owl, run in a child process.

    Combining torch/lightning with the JVM segfaulted CI inside the pytest process (#515). A child
    process turns such a crash into a failed assertion on the return code instead of killing the suite.
    """
    SCRIPT = """
import json
from ontolearn.semantic_caching import run_semantic_cache, run_non_semantic_cache
import sys
args = (sys.argv[1], None, 100, "EBR")
semantic, _ = run_semantic_cache(*args, "LRU", 0, "cold", True)
non_semantic, _ = run_non_semantic_cache(*args, True)
print("RESULT", json.dumps({"semantic": semantic["hit_ratio"], "non_semantic": non_semantic["hit_ratio"]}))
"""

    def test_neural_retrieval(self, tmp_path):
        # The KGE is trained next to the KG, so work on a copy to leave KGs/ untouched.
        path_kg = shutil.copy("KGs/Family/father.owl", tmp_path)
        proc = subprocess.run([sys.executable, "-c", self.SCRIPT, str(path_kg)], capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, f"Neural retrieval process failed (code {proc.returncode}):\n{proc.stderr[-2000:]}"
        line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
        hit_ratio = json.loads(line[len("RESULT "):])
        assert float(hit_ratio["semantic"]) >= float(hit_ratio["non_semantic"]), hit_ratio
