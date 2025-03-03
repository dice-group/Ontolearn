from ontolearn.concept_learner import CLIP
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
import os
import warnings
warnings.filterwarnings("ignore")

class TestCLIP:

    def test_prediction_quality_family(self):
        knowledge_base_path="./CLIPData/family/family.owl"
        path_of_embeddings="./CLIPData/family/embeddings/ConEx_entity_embeddings.csv"
        if os.path.exists(knowledge_base_path) and os.path.exists(path_of_embeddings):
            KB = KnowledgeBase(path="./CLIPData/family/family.owl")
            op = ExpressRefinement(knowledge_base=KB, use_inverse=False,
                              use_numeric_datatypes=False)
            clip = CLIP(knowledge_base=KB, path_of_embeddings=path_of_embeddings,
                 refinement_operator=op, load_pretrained=True, max_runtime=60)
            kb_namespace = list(KB.ontology.classes_in_signature())[0].iri.get_namespace()
            dl_parser = DLSyntaxParser(kb_namespace)
            brother = dl_parser.parse('Brother')
            daughter = dl_parser.parse('Daughter')
            pos = set(KB.individuals(brother)).union(set(KB.individuals(daughter)))
            neg = set(KB.individuals())-set(pos)
            node = list(clip.fit(pos, neg).best_descriptions)[0]
            assert node.quality > 0.85

    def test_prediction_quality_mutagenesis(self):
        knowledge_base_path="./CLIPData/mutagenesis/mutagenesis.owl"
        path_of_embeddings="./CLIPData/mutagenesis/embeddings/ConEx_entity_embeddings.csv"
        if os.path.exists(knowledge_base_path) and os.path.exists(path_of_embeddings):
            KB = KnowledgeBase(path=knowledge_base_path)
            op = ExpressRefinement(knowledge_base=KB, use_inverse=False,
                              use_numeric_datatypes=False)
            clip = CLIP(knowledge_base=KB, path_of_embeddings=path_of_embeddings,
                 refinement_operator=op, load_pretrained=True, max_runtime=60)
            kb_namespace = list(KB.ontology.classes_in_signature())[0].iri.get_namespace()
            dl_parser = DLSyntaxParser(kb_namespace)
            exists_inbond = dl_parser.parse('∃ inBond.Carbon-10')
            not_bond7 = dl_parser.parse('¬Bond-7')
            pos = set(KB.individuals(exists_inbond)).intersection(set(KB.individuals(not_bond7)))
            neg = set(KB.individuals())-set(pos)
            node = list(clip.fit(pos, neg).best_descriptions)[0]
            assert node.quality > 0.25
