from argparse import ArgumentParser
from typing import List

from flask import Flask, request

from ontolearn import KnowledgeBase, DrillAverage, Node


def create_flask_app():
    # TODO: is it necessary to add the kb and drill objects here or is adding them globally sufficient
    app = Flask(__name__, instance_relative_config=True)

    @app.route('concept_learning', methods=['POST'])
    def concept_learning_endpoint():
        learning_problem = request.get_json()
        no_of_hypotheses = request.form.get("no_of_hypotheses", 1, type=int)
        drill_average.fit(learning_problem["positives"], learning_problem["negatives"])
        hypotheses: List[Node] = drill_average.best_hypotheses(1)
        rdf_xml = ""
        # todo: convert hypotheses to RDF/XML
        return rdf_xml

    return app


kb = None

drill_average = None

if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str,
                        default='/home/demir/Desktop/Onto-learn_dev/KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=2)
    parser.add_argument("--min_length", type=int, default=3, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=6, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_per_concept", type=int, default=1)
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=2)

    # Evaluation related
    parser.add_argument('--num_fold_for_k_fold_cv', type=int, default=3, help='Number of cpus used during batching')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3,
                        help='Maximum allowed runtime during testing')
    # DQL related
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='../embeddings/Shallom_Family/Shallom_entity_embeddings.csv')
    parser.add_argument("--num_episode", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--num_of_sequential_actions', type=int, default=2)
    parser.add_argument('--pretrained_drill_sample_path', type=str, default='', help='Provide a path of .pth file')
    parser.add_argument('--pretrained_drill_avg_path', type=str, default='', help='Provide a path of .pth file')
    args = parser.parse_args()
    kb = KnowledgeBase(args.path_knowledge_base)
    # TODO: use DrillAverage or DrillSample? What is the difference?
    drill_average = DrillAverage(pretrained_model_path=args.pretrained_drill_avg_path,
                                 num_of_sequential_actions=args.num_of_sequential_actions,
                                 knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                                 num_episode=args.num_episode, verbose=args.verbose,
                                 num_workers=args.num_workers)

    app = create_flask_app()
    app.run(host="0.0.0.0", port=8090, processes=1)  # processes=1 is important to avoid copying the kb
