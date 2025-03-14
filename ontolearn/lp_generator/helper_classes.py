from tqdm import tqdm
import random
from rdflib import graph
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.refinement_operators import ExpressRefinement
import os
import json


class ConceptDescriptionGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator, depth=2, max_length=10, num_sub_roots=150):
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.depth = depth
        self.num_sub_roots = num_sub_roots
        self.max_length = max_length

    def apply_rho(self, concept):
        return {ref for ref in self.rho.refine(concept, max_length=self.max_length)}

    def generate(self):
        roots = self.apply_rho(self.kb.generator.thing)
        Refinements = set()
        Refinements.update(roots)
        print("|Thing refinements|: ", len(roots))
        roots_sample = random.sample(list(roots), k=self.num_sub_roots)
        print("Size of sample: ", len(roots_sample))
        for root in tqdm(roots_sample, desc="Refining roots..."):
            Refinements.update(self.apply_rho(root))
        return Refinements


class RDFTriples:
    """The knowledge graph/base is converted into triples of the form: individual_i ---role_j---> concept_k or
    individual_i ---role_j---> individual_k and stored in a txt file for the computation of embeddings."""

    def __init__(self, kb_path, storage_dir=None):
        """
        Args
        - kb_path: path to the owl file representing the knowledge base/ontology
        - storage_dir: directory in which to store the data to be generated. Not the directory needs not to exists, it would be created automatically
        """
        self.Graph = graph.Graph()
        self.Graph.parse(kb_path)
        self.kb_path = kb_path
        if storage_dir is None:
            self.storage_dir = self.kb_path[:self.kb_path.rfind("/")]
        else:
            self.storage_dir = storage_dir

    def export_triples(self, export_folder_name='triples'):
        os.makedirs(os.path.join(self.storage_dir, export_folder_name), exist_ok=True)
        if os.path.isfile(os.path.join(self.storage_dir, export_folder_name, "train.txt")):
            print("\n*** Embedding triples exist ***\n")
            return None
        train_file = open("%s/train.txt" % os.path.join(self.storage_dir, export_folder_name), mode="w")
        for s, p, o in self.Graph:
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
    A refinement operator is used to generate a large number of concepts, from which we filter and retain the shortest
    non-redundant concepts. We export each concept and its instances (eventually positive and negative examples) into
    a json file.
    """

    def __init__(self, path, storage_dir=None, max_num_lps=1000, depth=3, max_child_length=20, refinement_expressivity=0.2,
                 downsample_refinements=True, sample_fillers_count=10, num_sub_roots=50, min_num_pos_examples=1):
        """
        Args
        - kb_path: path to the owl file representing the knowledge base/ontology
        - storage_dir: directory in which to store the data to be generated. Not the directory needs not to exists, it would be created automatically
        - max_num_lps: the maximum number of learning problems to store
        - depth, max_child_length, refinement_expressivity, sample_fillers_count, num_sub_roots all refer to the size of the data (learning problems) to be generated
        - downsample_refinements: whether to downsample refinements in ExpressRefinement. If refinement_expressivity<1, this must be set to True
        """
        self.path = path
        if storage_dir is None:
            self.storage_dir = f'{self.path[:self.path.rfind("/")]}/LPs/'
        else:
            self.storage_dir = storage_dir
        self.max_num_lps = max_num_lps
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.kb = KnowledgeBase(path=path)
        self.num_examples = self.find_optimal_number_of_examples()
        self.min_num_pos_examples = min_num_pos_examples
        atomic_concepts = frozenset(self.kb.ontology.classes_in_signature())
        self.atomic_concept_names = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        rho = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length, sample_fillers_count=sample_fillers_count,
                                downsample=downsample_refinements, use_inverse=False, use_card_restrictions=False,
                                use_numeric_datatypes=False, use_time_datatypes=False, use_boolean_datatype=False,
                                expressivity=refinement_expressivity)
        self.lp_gen = ConceptDescriptionGenerator(knowledge_base=self.kb, refinement_operator=rho, depth=depth,
                                                  num_sub_roots=num_sub_roots)

    def find_optimal_number_of_examples(self):
        if self.kb.individuals_count() >= 600:
            return min(self.kb.individuals_count()//2, 1000)
        return self.kb.individuals_count()

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
            if not self.kb.individuals_set(concept) in non_redundancy_hash_map and \
                    self.min_num_pos_examples <= self.kb.individuals_count(concept):
                non_redundancy_hash_map[self.kb.individuals_set(concept)] = concept
            else:
                continue
            if len(non_redundancy_hash_map) >= self.max_num_lps:
                break
        print("Concepts generation done!\n")
        print("Number of atomic concepts: ", len(self.atomic_concept_names))
        print("Longest concept length: ",
              max({l for l in [self.kb.concept_len(c) for c in non_redundancy_hash_map.values()]}), "\n")
        print("Total number of concepts: ", len(non_redundancy_hash_map), "\n")
        self.train_concepts = list(non_redundancy_hash_map.values())
        print("Data generation completed")
        return self

    def sample_examples(self, pos, neg):
        if min(len(pos), len(neg)) >= self.num_examples//2:
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
        positive = random.sample(pos, min(num_pos_ex, len(pos)))
        negative = random.sample(neg, min(num_neg_ex, len(neg)))
        return positive, negative

    def save_data(self):
        data = dict()
        for concept in tqdm(self.train_concepts, desc="Sample examples and save data..."):
            pos = set(self.kb.individuals(concept))
            neg = set(self.kb.individuals())-pos
            if len(neg) == 0:
                continue
            pos = [ind.str.split("/")[-1] for ind in pos]
            neg = [ind.str.split("/")[-1] for ind in neg]
            positive, negative = self.sample_examples(pos, neg)
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            data[concept_name] = {'positive examples': positive, 'negative examples': negative}
        data = list(data.items())
        os.makedirs(self.storage_dir, exist_ok=True)
        with open(f'{self.storage_dir}/LPs.json', 'w') as file_train:
            json.dump(dict(data), file_train, indent=3, ensure_ascii=False)
        print(f'Data saved at {self.storage_dir}')
