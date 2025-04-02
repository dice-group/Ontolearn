from cnir.executer import ExecuteTest
from cnir.utils import str2bool
import argparse
import warnings
import os

"""
How to run? E.g.,

cnir --dataset_dir datasets/lymphography --output_dir CNIR_Composite_lymph --model composite --pma_model_path PMA_lymph/model.pt --batch_size 256 --num_workers 0
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def get_default_arguments(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="The path of a folder containing training_data.json, which contains a list of class expressions or list of tuples where the first elements are class expressions."
                             ",e.g., datasets/carcinogenesis/data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The location where to store the trained model and training results.")
    parser.add_argument("--th", type=float, default=0.5,
                        help="Threshold on probabilities to decide which individual is an instance of a class expression")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        help="An optimizer",
                        choices=["adam", "adamw", "sdg"])
    parser.add_argument("--model", type=str,
                        default="transformer",
                        choices=["transformer", "composite", "lstm", "gru", "Transformer",
                                 "Composite", "Lstm", "Gru", "TRANSFORMER", "COMPOSITE", "LSTM",
                                 "GRU"],
                        help="Available models graph embedding models.")
    parser.add_argument("--use_pma", type=str2bool, default=True,
                        help="Whether to use PMA as encoder for atomic concepts via their sets of instances. This applies only to the `composite` model.")
    parser.add_argument("--pma_model_path", type=str, default=None,
                        help="Path to a pretrained PMA model.")
    parser.add_argument("--pe_dropout", type=float, default=None,
                        help="Dropout probability in positional encoding.")
    parser.add_argument("--pretrained_tokenizer_path", type=str, default=None,
                        help="Path to a pretrained tokenizer. The directory must contain files like `special_tokens_map.json, tokenizer_config.json, tokenizer.json`")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to a pretrained model, which must be of type `transformers.PretrainedModel`")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Maximum sequence length (number of tokens in a class expression)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini batch size.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of CPUs to use for batch data loading.")
    parser.add_argument("--lr", type=float, default=3.5e-4)
    return parser.parse_args()


def main():
    args = get_default_arguments()
    """
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    print("Parent dir: ", parent_dir)
    dataset_dir = os.path.join(parent_dir, "datasets\datasets\mutagenesis")
    print("Dataset dir: ", dataset_dir)
    args.dataset_dir = dataset_dir
    output_dir = os.path.join(parent_dir, "output\composite")
    args.output_dir = output_dir
    args.model = 'composite'
    args.use_pma = False
    args.pretrained_model_path =  'D:\PycharmProjects\CoNeuralReasoner\output\composite'
    #pma_model_path = os.path.join(parent_dir, "output\model.pt")
    #args.pma_model_path = pma_model_path
    """
    ExecuteTest(args).start()


if __name__ == "__main__":
    main()