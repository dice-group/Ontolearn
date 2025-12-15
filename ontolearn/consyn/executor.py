import logging
from pathlib import Path
from typing import Optional
from torch.optim import AdamW

from ontolearn.consyn.configs import CONFIG
from ontolearn.consyn.initializer import Initializer
from ontolearn.consyn.trainer import ConSynTrainer
from ontolearn.consyn.model.model import ConSynGeneratorModel

logger = logging.getLogger(__name__)


class ConSynExecutor:
    def __init__(self, kb_path, lps_path, pretrained_path: str, num_k_predictions: int = 30, device: Optional[str] = None, verbose: bool = False):
        CONFIG['KNOWLEDGE_BASE_PATH'] = kb_path
        CONFIG['LEARNING_PROBLEM_PATH'] = lps_path

        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            raise FileNotFoundError(f"path_of_consyn_trained_models does not exist: {pretrained_path}")
        if not pretrained_path.is_dir():
            raise NotADirectoryError(f"path_of_consyn_trained_models is not a directory: {pretrained_path}")

        experiment_name = pretrained_path.name
        CONFIG['EXPERIMENT_DIR'] = str(pretrained_path)

        data_dir = pretrained_path / "data"

        required_paths = {
            "GENERATED_DATA_PATH": data_dir / "generated_raw_data.json",
            "TASK_LABEL_MAPPING_PATH": data_dir / "task_label_mappings.json",
            "FIT_GENERATED_DATA_PATH": data_dir / "fit" / "generated_raw_data.json",
            "FIT_TASK_LABEL_MAPPING_PATH": data_dir / "fit" / "task_label_mappings.json",
            "MODEL_DIR": pretrained_path / "model",
            "TOKENIZER_JSON": pretrained_path / "tokenizer.json"
        }

        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required files/directories for experiment '{experiment_name}': {missing}"
            )

        CONFIG['GENERATED_DATA_PATH'] = str(required_paths["GENERATED_DATA_PATH"])
        CONFIG['TASK_LABEL_MAPPING_PATH'] = str(required_paths["TASK_LABEL_MAPPING_PATH"])
        CONFIG['FIT_PATH'] = {
            'GENERATED_DATA_PATH': str(required_paths["FIT_GENERATED_DATA_PATH"]),
            'TASK_LABEL_MAPPING_PATH': str(required_paths["FIT_TASK_LABEL_MAPPING_PATH"]),
        }

        if device is not None:
            CONFIG['device'] = device

        self.num_k_predictions = num_k_predictions
        self.verbose = verbose

        self.config = CONFIG
        self.device = CONFIG['device']

        if self.verbose:
            print(f"Using device: {self.device}\n")

        # Initialize core components
        initializer = Initializer(config=self.config, mode="fit", verbose=self.verbose)
        components = initializer.get_components()

        self.tokenizer = components['tokenizer']
        self.grammar_parser = components['grammar_parser']
        self.reward = components['reward']
        self.heuristic = components['heuristic']

        vocab_size = self.tokenizer.vocab_size

        # Initialize model
        self.model = ConSynGeneratorModel(
            tokenizer=self.tokenizer,
            input_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            embed_dim=self.config['d_model'],
            num_encoder_layers=self.config['num_layers'],
            num_decoder_layers=self.config['num_layers'],
            num_heads=self.config['nhead'],
            ff_dim=self.config['dim_feedforward'],
            dropout_prob=self.config['dropout'],
            num_segments=4,
            use_checkpointing=self.config['use_checkpointing'],
            pre_norm=self.config['pre_norm']
        ).to(self.device)

        # Optimizer and trainer
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        self.trainer = ConSynTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            grammar_parser=self.grammar_parser,
            reward_function=self.reward,
            heuristic_function=self.heuristic,
            optimizer=optimizer,
            device=self.device,
            num_k_predictions=self.num_k_predictions,
            max_gen_length=self.config['max_output_seq_len'],
            expr_save_path=self.config['EXPERIMENT_DIR'],
            verbose=self.verbose
        )

        self.trainer.prepare_for_fit(verbose=self.verbose)
