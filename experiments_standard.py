"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Reproducing our experiments Experiments

This script performs the following computations
1. Parse KG.
2. Load learning problems LP= {(E^+,E^-)...]

3. Initialize models .
    3.1. Initialize DL-learnerBinder objects to communicate with DL-learner binaries.
    3.2. Initialize DRILL.
4. Provide models + LP to Experiments object.
    4.1. Each learning problem provided into models
    4.2. Best hypothesis/predictions of models given E^+ and E^- are obtained.
    4.3. F1-score, Accuracy, Runtimes and Number description tested information stored and serialized.
"""
import json
import os
import time
from argparse import ArgumentParser

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import Drill
from ontolearn.experiments import Experiments
from ontolearn.metrics import F1
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.utils import setup_logging
from owlapy.model import OWLOntology, OWLReasoner

setup_logging()
full_computation_time = time.time()


def sanity_checking_args(args):
    try:
        assert os.path.isfile(args.path_knowledge_base)
    except AssertionError:
        print(f'--path_knowledge_base ***{args.path_knowledge_base}*** does not lead to a file.')
        exit(1)
    assert os.path.isfile(args.path_knowledge_base_embeddings)
    assert os.path.isfile(args.path_knowledge_base)


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    from owlapy.owlready2 import OWLOntology_Owlready2
    from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
    from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def start(args):
    sanity_checking_args(args)
    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
    with open(args.path_lp) as json_file:
        settings = json.load(json_file)
    problems = [(k, set(v['positive_examples']), set(v['negative_examples'])) for k, v in
                settings['problems'].items()]

    print(f'Number of problems {len(problems)} on {kb}')
    # @ TODO write curl for getting DL-learner binaries
    # Initialize models
    # celoe = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='celoe')
    # ocel = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='ocel')
    # eltl = DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='eltl')
    drill = Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(),
                  num_workers=args.num_workers, pretrained_model_path=args.pretrained_drill_avg_path,
                  verbose=args.verbose)

    Experiments(max_test_time_per_concept=args.max_test_time_per_concept).start(dataset=problems,
                                                                                models=[drill,
                                                                                        # celoe,ocel,eltl
                                                                                        ])


if __name__ == '__main__':
    parser = ArgumentParser()
    # LP dependent
    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='embeddings/ConEx_Family/ConEx_entity_embeddings.csv')
    parser.add_argument("--path_lp", type=str, default='LPs/Family/lp.json')
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='pre_trained_agents/Family/DrillHeuristic_averaging/DrillHeuristic_averaging.pth',
                        help='Provide a path of .pth file')
    # Binaries for DL-learner
    parser.add_argument("--path_dl_learner", type=str, default='/home/demir/Desktop/Softwares/DRILL/dllearner-1.4.0')
    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')
    # General
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')

    start(parser.parse_args())
