"""How to run? E.g.,

python cnir/scripts/run_pma.py --dataset_dir datasets/animals --output_dir PMA_animals --num_example 20 --epochs 500 --batch_size 8 --num_workers 0
"""

from cnir.executer import ExecutePMA
from cnir.utils import str2bool
import argparse
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def get_default_arguments(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="The path of a folder containing data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The location where to store the trained model and training results.")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--th", type=float, default=0.5,
                        help="Threshold on probabilities to decide which individual is an instance of a class expression")
    parser.add_argument("--num_examples", type=int, default=200,
                        help="Number of examples to sample per learning problem during training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of CPUs to use for batch data loading.")
    return parser.parse_args()
def main():
    args = get_default_arguments()
    ExecutePMA(args).start()

if __name__ =="__main__":
    main()