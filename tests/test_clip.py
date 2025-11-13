from ontolearn.learners import CLIP
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
import os, pathlib
import warnings
warnings.filterwarnings("ignore")

class TestCLIP:

    def test_prediction_quality(self):
        base_path = pathlib.Path(__file__).parent.resolve()._str
        knowledge_base_path = base_path[:base_path.rfind("/")+1] + "KGs/Family/family-benchmark_rich_background.owl"
        KB = KnowledgeBase(path=knowledge_base_path)
        op = ExpressRefinement(knowledge_base=KB, use_inverse=False,
                          use_numeric_datatypes=False)
        clip = CLIP(knowledge_base=KB,
             refinement_operator=op, load_pretrained=True, max_runtime=60)
        kb_namespace = list(KB.ontology.classes_in_signature())[0].iri.get_namespace()
        dl_parser = DLSyntaxParser(kb_namespace)
        brother = dl_parser.parse('Brother')
        daughter = dl_parser.parse('Daughter')
        pos = set(KB.individuals(brother)).union(set(KB.individuals(daughter)))
        neg = set(KB.individuals())-set(pos)
        node = list(clip.fit(pos, neg).best_descriptions)[0]
        assert node.quality > 0.25

if __name__ == "__main__":
    test = TestCLIP()
    test.test_prediction_quality()
