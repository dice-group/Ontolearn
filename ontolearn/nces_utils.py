from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import random, numpy as np
from rdflib import graph
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.refinement_operators import ExpressRefinement
import os, json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SimpleSolution:
    
    def __init__(self, vocab, atomic_concept_names):
        self.name = 'SimpleSolution'
        self.atomic_concept_names = atomic_concept_names
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.train_from_iterator(vocab, trainer)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.tokenizer.pad_token = "[PAD]"
            
    def predict(self, expression: str):
        atomic_classes = [atm for atm in self.tokenizer.tokenize(expression) if atm in self.atomic_concept_names]
        if atomic_classes == []:
            atomic_classes =['⊤']
        return " ⊔ ".join(atomic_classes)
    
    
class ConceptDescriptionGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator, depth=2, max_length=10, num_rand_samples=150):
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.depth = depth
        self.num_rand_samples = num_rand_samples
        self.max_length = max_length

    def apply_rho(self, concept):
        #refinements = {ref for ref in self.rho.refine(concept, max_length=self.max_length)}
        return {ref for ref in self.rho.refine(concept, max_length=self.max_length)}

    def generate(self):
        roots = self.apply_rho(self.kb.thing)
        Refinements = set()
        Refinements.update(roots)
        print ("|Thing refinements|: ", len(roots))
        roots_sample = random.sample(list(roots), k=self.num_rand_samples)
        print("Size of sample: ", len(roots_sample))
        for root in tqdm(roots_sample, desc="Refining roots..."):
            Refinements.update(self.apply_rho(root))
        return Refinements
    
    
class RDFTriples:
    """The knowledge graph/base is converted into triples of the form: individual_i ---role_j---> concept_k or individual_i ---role_j---> individual_k
    and stored in a txt file for the computation of embeddings."""
    
    def __init__(self, source_kg_path):
        self.Graph = graph.Graph()
        self.Graph.parse(source_kg_path)
        self.source_kg_path = source_kg_path
              
    def export_triples(self, export_folder_name='triples'):
        if not os.path.exists(os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name)):
            os.mkdir(os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name))
        if os.path.isfile(os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name, "train.txt")):
            print("\n*** Embedding triples exist ***\n")
            return
        train_file = open("%s/train.txt" % os.path.join(self.source_kg_path[:self.source_kg_path.rfind("/")], export_folder_name), mode="w")
        for s,p,o in self.Graph:
            s = s.expandtabs()[s.expandtabs().rfind("/")+1:]
            p = p.expandtabs()[p.expandtabs().rfind("/")+1:]
            o = o.expandtabs()[o.expandtabs().rfind("/")+1:]
            if s and p and o:
                train_file.write(s+"\t"+p+"\t"+o+"\n")
        train_file.close()
        print("*********************Finished exporting triples*********************\n")

class KB2Data:
    """
    This class takes an owl file, loads it into a knowledge base using ontolearn.knowledge_base.KnowledgeBase.
    A refinement operator is used to generate a large number of concepts, from which we filter and retain the shortest non-redundant concepts.
   Finally, we aim at training a deep neural network to predict the syntax of concepts from their instances. Hence, we export each concept and its instances (eventually positive and negative examples) into json files.  
    """

    def __init__(self, path, rho_name="ExpressRefinement", depth=5, max_child_length=25, refinement_expressivity=0.6, downsample_refinements=True, k=10, num_rand_samples=150, min_num_pos_examples=1, max_num_pos_examples=2000):
        self.path = path
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.kb = KnowledgeBase(path=path)
        self.num_examples = min(self.kb.individuals_count()//2, 1000)
        self.min_num_pos_examples = min_num_pos_examples
        self.max_num_pos_examples = max_num_pos_examples
        atomic_concepts = frozenset(self.kb.ontology().classes_in_signature())
        self.atomic_concept_names = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        rho = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length, sample_fillers_count=k, downsample=downsample_refinements,\
                                use_inverse=False, use_card_restrictions=False, use_numeric_datatypes=False, use_time_datatypes=False,\
                                use_boolean_datatype=False, expressivity=refinement_expressivity)
        self.lp_gen = ConceptDescriptionGenerator(knowledge_base=self.kb, refinement_operator=rho, depth=depth, num_rand_samples=num_rand_samples)

    
    def generate_descriptions(self):
        print()
        print("#"*60)
        print("Started generating data on the "+self.path.split("/")[-1].split(".")[0]+" knowledge base")
        print("#"*60)
        print()
        All_individuals = set(self.kb.individuals())
        print("Number of individuals in the knowledge base: {} \n".format(len(All_individuals)))
        Concepts = self.lp_gen.generate()
        non_redundancy_hash_map = dict()
        show_some_length = True
        for concept in tqdm(sorted(Concepts, key=lambda c: self.kb.concept_len(c)), desc="Filtering process..."):
            if not self.kb.individuals_set(concept) in non_redundancy_hash_map and self.min_num_pos_examples <= self.kb.individuals_count(concept):
                non_redundancy_hash_map[self.kb.individuals_set(concept)] = concept
            else: continue
            if self.kb.concept_len(concept) >= 15 and show_some_length:
                print("A long concept found: ", self.kb.concept_len(concept))
                show_some_length = False
        print("Concepts generation done!\n")
        print("Number of atomic concepts: ", len(self.atomic_concept_names))
        print("Longest concept length: ", max({l for l in [self.kb.concept_len(c) for c in non_redundancy_hash_map.values()]}), "\n")
        print("Total number of concepts: ", len(non_redundancy_hash_map), "\n")
        self.train_concepts = list(non_redundancy_hash_map.values())
        print("Data generation completed")
        return self
    

    def save_data(self):
        data = dict()
        for concept in tqdm(self.train_concepts, desc="Sample examples and save data..."):
            pos = set(self.kb.individuals(concept))
            neg = set(self.kb.individuals())-pos
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
            neg = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
            if min(len(neg),len(pos)) >= self.num_examples//2:
                if len(pos) > len(neg):
                    num_neg_ex = self.num_examples//2
                    num_pos_ex = self.num_examples-num_neg_ex
                else:
                    num_pos_ex = self.num_examples//2
                    num_neg_ex = self.num_examples-num_pos_ex
            elif len(pos) > len(neg):
                num_neg_ex = len(neg)
                num_pos_ex = self.num_examples-num_neg_ex
            elif len(pos) < len(neg):
                num_pos_ex = len(pos)
                num_neg_ex = self.num_examples-num_pos_ex
            else:
                print("Invalid number of instances")
                continue
            positive = random.sample(pos, num_pos_ex)
            negative = random.sample(neg, num_neg_ex)
            
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            data[concept_name] = {'positive examples': positive, 'negative examples': negative}
            
        data = list(data.items())
        os.makedirs(f'../NCESData/{self.path.split("/")[-2]}/training_data/', exist_ok=True)
        if not os.path.isfile(f'../NCESData/{self.path.split("/")[-2]}/training_data/Data.json'):
            with open(f'../NCESData/{self.path.split("/")[-2]}/training_data/Data.json', 'w') as file_train:
                json.dump(dict(data), file_train, indent=3, ensure_ascii=False)
            print(f'Data saved at ../datasets/{self.path.split("/")[-2]}')
        else:
            print("Training data already exists!")
              
            

