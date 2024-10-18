# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

from ontolearn.executor import execute
from argparse import ArgumentParser


def get_default_arguments(description=None):
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="celoe", choices=["celoe", "ocel", "evolearner", "nces"],
                        help="Available concept learning models.")

    # Knowledge graph related arguments
    parser.add_argument("--knowledge_base_path", type=str, default="KGs/Family/family-benchmark_rich_background.owl",
                        help="Path to the knowledge base/ontology. This file contains '.owl' extension,"
                             "e.g. 'some/path/kb.owl'")
    parser.add_argument("--sparql_endpoint", type=str, default=None,
                        help="An endpoint of a triple store, e.g. 'http://localhost:3030/family/sparql'. ")
    parser.add_argument("--path_of_embeddings", type=str,
                        default='NCESData/family/embeddings/ConEx_entity_embeddings.csv',
                        help="Path to knowledge base embeddings. Some models like NCES require this,"
                             "e.g. 'some/path/kb_embeddings.csv'")
    parser.add_argument("--save", action="store_true", help="save the hypothesis?")
    # Common model arguments
    parser.add_argument("--path_learning_problem", type=str, default='examples/uncle_lp2.json',
                        help="Path to a .json file that contains 2 properties 'positive_examples' and "
                             "'negative_examples'. Each of this properties should contain the IRIs of the respective"
                             "instances. e.g. 'some/path/lp.json'")
    parser.add_argument("--quality_metric", type=str, default='f1',
                        choices=["f1", "accuracy", "recall", "precision", "weighted_accuracy"],
                        help="Quality metric.")
    parser.add_argument("--max_runtime", type=int, default=5, help="Maximum runtime.")

    # CELOE, OCEL and Evolearner only

    parser.add_argument('--terminate_on_goal', type=bool, default=True, help="Terminate when finding concept of quality"
                                                                             "1.0?")
    parser.add_argument("--use_card_restrictions", type=bool, default=True,
                        help="Use cardinality restrictions for object properties?")
    parser.add_argument("--use_inverse", type=bool, default=True, help="Use inverse.")
    parser.add_argument("--card_limit", type=int, default=10, help="Cardinality limit for object properties.")
    parser.add_argument("--max_nr_splits", type=int, default=12, help="Maximum number of splits.")

    # CELOE and OCEL only
    parser.add_argument("--max_results", type=int, default=10, help="Maximum results to find (not to show)")
    parser.add_argument("--iter_bound", type=int, default=10_000, help="Iterations bound.")
    parser.add_argument("--max_num_of_concepts_tested", type=int, default=10_000,
                        help="Maximum number of concepts tested.")
    parser.add_argument("--best_only", type=bool, default=True, help="Best results only?")
    parser.add_argument("--calculate_min_max", type=bool, default=True, help="Only for statistical purpose.")
    parser.add_argument("--gain_bonus_factor", type=float, default=0.3,
                        help="Factor that weighs the increase in quality compared to the parent node.")
    parser.add_argument("--expansion_penalty_factor", type=float, default=0.1,
                        help="The value that is subtracted from the heuristic for each horizontal expansion of this")
    parser.add_argument("--max_child_length", type=int, default=10, help="Maximum child length")
    parser.add_argument("--use_negation", type=bool, default=True, help="Use negation?")
    parser.add_argument("--use_all_constructor", type=bool, default=True, help="Use all constructors?")
    parser.add_argument("--use_numeric_datatypes", type=bool, default=True, help="Use numeric data types?")
    parser.add_argument("--use_time_datatypes", type=bool, default=True, help="Use time datatypes?")
    parser.add_argument("--use_boolean_datatype", type=bool, default=True, help="Use boolean datatypes?")

    # CELOE only
    parser.add_argument("--start_node_bonus", type=float, default=0.1, help="Special value added to the root node.")
    parser.add_argument("--node_refinement_penalty", type=float, default=0.001, help="Node refinement penalty.")

    # EvoLearner Only
    parser.add_argument("--use_data_properties", type=bool, default=True, help="Use data properties?")
    parser.add_argument("--tournament_size", type=int, default=7, help="Tournament size.")
    parser.add_argument("--population_size", type=int, default=800, help="Population size.")
    parser.add_argument("--num_generations", type=int, default=200, help="Number of generations.")
    parser.add_argument("--height_limit", type=int, default=17, help="Height limit.")
    parser.add_argument("--gain", type=int, default=2048, help="Gain.")
    parser.add_argument("--penalty", type=int, default=1, help="Penalty.")
    parser.add_argument("--max_t", type=int, default=2, help="Number of paths.")
    parser.add_argument("--jump_pr", type=float, default=0.5, help="Probability to explore paths of length 2.")
    parser.add_argument("--crossover_pr", type=float, default=0.9, help="Crossover probability.")
    parser.add_argument("--mutation_pr", type=float, default=0.1, help="Mutation probability")
    parser.add_argument("--elitism", type=bool, default=False, help="Elitism.")
    parser.add_argument("--elite_size", type=float, default=0.1, help="Elite size")
    parser.add_argument("--min_height", type=int, default=1, help="Minimum height of trees")
    parser.add_argument("--max_height", type=int, default=3, help="Maximum height of trees")
    parser.add_argument("--init_method_type", type=str, default="RAMPED_HALF_HALF",
                        help="Random initialization method.", choices=["GROW", "FULL", "RAMPED_HALF_HALF"])

    # NCES only
    parser.add_argument("--learner_names", type=str, nargs="+", default=["SetTransformer"], help="Learner name.",
                        choices=["SetTransformer", "GRU", "LSTM"])
    parser.add_argument("--proj_dim", type=int, default=128, help="Number of projection dimensions.")
    parser.add_argument("--rnn_n_layers", type=int, default=2, help="Number of RNN layers (only for LSTM and GRU).")
    parser.add_argument("--drop_prob", type=float, default=0.1, help="Drop probability.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds (only for SetTransformer).")
    parser.add_argument("--num_inds", type=int, default=32, help="Number of inducing points (only for SetTransformer).")
    parser.add_argument("--ln", type=bool, default=False, help="Layer normalization (only for SetTransformer).")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--decay_rate", type=int, default=0, help="Decay rate.")
    parser.add_argument("--clip_value", type=int, default=5, help="Clip value.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--max_length", type=int, default=48, help="Maximum length")
    parser.add_argument("--load_pretrained", type=bool, default=True, help="Load pretrained.")
    parser.add_argument("--sorted_examples", type=bool, default=True, help="Sorted examples.")
#    parser.add_argument("--pretrained_model_name", type=str, default="SetTransformer", help="Pretrained model name",
#                        choices=["SetTransformer", "GRU", "LSTM"])

    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    execute(get_default_arguments())
