from cnir.executer import Execute
from cnir.utils import str2bool
import argparse
import warnings
import os

"""How to run? E.g.,

cnir --dataset_dir datasets/lymphography --output_dir CNIR_Composite_lymph --num_example 100 --epochs 200 --model composite --pma_model_path PMA_lymph/model.pt --batch_size 256 --num_workers 0
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
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of examples to sample per learning problem during training")
    parser.add_argument("--th", type=float, default=0.5,
                        help="Threshold on probabilities to decide which individual is an instance of a class expression")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        help="An optimizer",
                        choices=["adam", "adamw", "sdg"])
    parser.add_argument("--model", type=str,
                        default="transformer",
                        choices=["transformer", "composite", "lstm", "gru", "Transformer", "Composite", "Lstm", "Gru", "TRANSFORMER", "COMPOSITE", "LSTM", "GRU"],
                        help="Available models graph embedding models.")
    parser.add_argument("--use_pma", type=str2bool, default=True, help="Whether to use PMA as encoder for atomic concepts via their sets of instances. This applies only to the `composite` model.")
    parser.add_argument("--pma_model_path", type=str, default=None, help="Path to a pretrained PMA model.")
    parser.add_argument("--pe_dropout", type=float, default=None, help="Dropout probability in positional encoding.")
    parser.add_argument("--num_encoder_layers", type=int, default=None, help="Number of encoder layers in transformer.")
    parser.add_argument("--num_rnn_layers", type=int, default=None, help="Number of encoder layers in RNNs.")
    parser.add_argument("--num_attention_heads", type=int, default=None, help="Number of attention heads in transformer.")
    parser.add_argument("--vocab_size", type=int, default=None, help="Embedding lookup table size.")
    parser.add_argument("--train_tokenizer", type=str2bool, default=True, help="Whether to train a tokenizer on atomic concept names, role names, and description logic-specific tokens")
    parser.add_argument("--pretrained_tokenizer_path", type=str, default=None, help="Path to a pretrained tokenizer. The directory must contain files like `special_tokens_map.json, tokenizer_config.json, tokenizer.json`")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to a pretrained model, which must be of type `transformers.PretrainedModel`")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length (number of tokens in a class expression)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini batch size.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of CPUs to use for batch data loading.")
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--train_test_split", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()
    
    
def main():
    args = get_default_arguments()
    args.dataset_dir = "D:\PycharmProjects\CoNeuralReasoner\datasets\datasets\semantic_bible"
    args.model = "composite"
    args.output_dir = "D:\PycharmProjects\CoNeuralReasoner\output\comp_seman"
    args.use_pma = True
    args.pma_model_path = "D:\PycharmProjects\CoNeuralReasoner\output\model.pt"
    args.num_workers = 0
    args.epochs = 2
    Execute(args).start()

if __name__ =="__main__":
    main()