import os
import unittest
from examples.retrieval_eval import execute
from examples.retrieval_eval_under_incomplete import execute as execute2
import shutil


def clear():
    if os.path.exists("incomplete_father_0_1"):
        shutil.rmtree("incomplete_father_0_1")
    if os.path.exists("KGs_Family_father_owl"):
        shutil.rmtree("KGs_Family_father_owl")
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")


class RetrievalTests(unittest.TestCase):

    def test_retrieval_eval(self):
        class ARGS:
            def __init__(self):
                self.path_kg = "KGs/Family/father.owl"
                self.path_kge_model = None
                self.endpoint_triple_store = None
                self.gamma = 0.9
                self.seed = 1
                self.min_jaccard_similarity = 0.0
                self.ratio_sample_nc = 0.2
                self.ratio_sample_object_prop = 0.1
                self.num_nominals = 10
                self.path_report = "incomplete_father_0_1/ALCQHI_Retrieval_Results.csv"
        args = ARGS()
        clear()
        os.mkdir("incomplete_father_0_1")
        js, f1 = execute(args)

        self.assertEqual(js, 1.0)
        self.assertEqual(f1, 1.0)

    # def test_retrieval_eval_under_incomplete(self):
    #     class ARGS:
    #         def __init__(self):
    #             self.path_kg = "KGs/Family/father.owl"
    #             self.seed = 1
    #             self.ratio_sample_nc = None
    #             self.ratio_sample_object_prop = None
    #             self.path_report = "ALCQHI_Retrieval_Results.csv"
    #             self.number_of_subgraphs = 1
    #             self.ratio = 0.1
    #             self.operation = "incomplete"
    #             self.sample = "No"
    #
    #     args = ARGS()
    #     results = execute2(args)
    #     for r, v in results.items():
    #         self.assertGreaterEqual(v, 0.9)
    #     clear()

