from functools import partial
import json
import os
import random
import time

from owlapy import owl_expression_to_dl
from torch.optim import AdamW 
from torch.utils.data import DataLoader

from ontolearn.consyn.configs import CONFIG
from ontolearn.consyn.intializer import Initializer
from ontolearn.consyn.model.model import ConSynGeneratorModel
from ontolearn.consyn.trainer import ConSynTrainer
from ontolearn.consyn.utils import ConceptLearningDataset, custom_collate_fn_for_dataloader

from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils.static_funcs import compute_f1_score


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = CONFIG
    exp_save_dir = config['EXPERIMENT_DIR']
    print(f"Using device: {config['device']}\n")
    # print(torch.cuda.device_count())
    # exit()

    initializer = Initializer(config)
    components = initializer.get_components()

    dl_to_owl_converter = components['converter']
    tokenizer = components['tokenizer']
    reasoner = components['reasoner']
    grammar_parser = components['grammar_parser']
    reward = components['reward']
    heuristic = components['heuristic']
    train_data, val_data, test_data = components['datasets']

    # if CONFIG['apply_task_label_logical_aug']:
    #     train_data = create_augmented_raw_data(train_data, tokenizer, CONFIG)

    train_dataset, val_dataset, test_dataset = None, None, None
    
    train_dataset = ConceptLearningDataset(train_data, tokenizer)
    if val_data:
        val_dataset = ConceptLearningDataset(val_data, tokenizer)

    if test_data:
        test_dataset = ConceptLearningDataset(test_data, tokenizer)

    # import json
    # with open("output.json", "w", encoding="utf-8") as f:
    #     json.dump(val_data, f, indent=4)
    
    collate_fn_train = partial(custom_collate_fn_for_dataloader, tokenizer=tokenizer)
    collate_fn_val_test = partial(custom_collate_fn_for_dataloader, tokenizer=tokenizer)

    train_loader, val_loader, test_loader = None, None, None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_dataloader_workers'], 
        collate_fn=collate_fn_train,
        pin_memory=True
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_dataloader_workers'],
            collate_fn=collate_fn_val_test,
            pin_memory=True
        )
        
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_dataloader_workers'],
            collate_fn=collate_fn_val_test,
            pin_memory=True
        )

    # print(len(test_dataset), test_dataset[0])
    # exit()

    vocab_size = tokenizer.vocab_size

    input_vocab_size = vocab_size
    target_vocab_size = vocab_size
    embed_dim = config['d_model']
    num_encoder_layers = config['num_layers']
    num_decoder_layers = config['num_layers']
    num_heads = config['nhead']
    ff_dim = config['dim_feedforward']
    dropout_prob = config['dropout']
    use_checkpointing = config['use_checkpointing']
    pre_norm = config['pre_norm']
    num_segments = 4
    
    model = ConSynGeneratorModel(tokenizer=tokenizer,
        input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, num_heads=num_heads,
        ff_dim=ff_dim, dropout_prob=dropout_prob, num_segments=num_segments, use_checkpointing=use_checkpointing, pre_norm=pre_norm
    )#.to(config['device'])

    model = model.to(config['device'])

    num_epochs = config['num_epochs']
    num_k_predictions = 30
    verbose = True

    optimizer = AdamW(model.parameters(), lr=1e-5)

    trainer = ConSynTrainer(model=model, tokenizer=tokenizer, grammar_parser=grammar_parser, reward_function=reward, heuristic_function = heuristic,
        optimizer=optimizer, device=config['device'], num_k_predictions=num_k_predictions, max_gen_length=config['max_output_seq_len'], expr_save_path = exp_save_dir, verbose=verbose
    )

    # trainer.train(num_epochs=num_epochs, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader)

    # use in test mode only
    # trainer.train(train_dataloader=None, val_dataloader=None, test_dataloader=test_loader, num_epochs=num_epochs)

    ### Testing fit method
    # np.random.seed(None)
    # torch_seed = int.from_bytes(os.urandom(4), 'little')
    # torch.manual_seed(torch_seed)
    # random.seed(None)
    
    # mocking cv lp splitting
    def load_traget_lp(json_file_path, problem_key, split=True, train_ratio=0.8):
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if 'problems' not in data or problem_key not in data['problems']:
            return None, None  if split else None

        problem_data = data['problems'][problem_key]
        pos_list = problem_data.get('positive_examples', [])
        neg_list = problem_data.get('negative_examples', [])

        def to_named_individuals(lst):
            return {OWLNamedIndividual(i) for i in lst}

        if not split:
            return PosNegLPStandard(
                pos=to_named_individuals(pos_list),
                neg=to_named_individuals(neg_list)
            )

        def split_list(lst):
            lst = lst[:]
            random.shuffle(lst)
            split_idx = int(len(lst) * train_ratio)
            return lst[:split_idx], lst[split_idx:]

        train_pos_list, test_pos_list = split_list(pos_list)
        train_neg_list, test_neg_list = split_list(neg_list)

        train_lp = PosNegLPStandard(
            pos=to_named_individuals(train_pos_list),
            neg=to_named_individuals(train_neg_list)
        )

        test_lp = PosNegLPStandard(
            pos=to_named_individuals(test_pos_list),
            neg=to_named_individuals(test_neg_list)
        )

        return train_lp, test_lp

    trainer.prepare_for_fit(verbose=verbose)
    for str_target_concept in ["Aunt", "Brother", "Mother", "Sister", "Cousin", "Grandgrandfather", "Grandgrandmother", "Grandmother", "Grandfather", "Uncle", "Grandson", "Son"]:
    # for str_target_concept in ["NotKnown" if 'mutagenesis' in str(config['LEARNING_PROBLEM_PATH']).lower() else "NOTKNOWN"]: #"NOTKNOWN"
        for ratio in [0.65, 0.80, 0.95]:
            str_target_concept_lp = load_traget_lp(config['LEARNING_PROBLEM_PATH'], str_target_concept, split=ratio)

            test_lp = None
            if isinstance(str_target_concept_lp, tuple):
                train_lp, test_lp = str_target_concept_lp
            else:
                train_lp = str_target_concept_lp

            kb = KnowledgeBase(path=config['KNOWLEDGE_BASE_PATH'])
            num_predictions = trainer.num_k_predictions
            print("ConSyn starts..", end="\t")
            start_time = time.time()
            pred_consyn = trainer.fit(knowledge_base=kb, target_concept=str_target_concept, target_concept_lp=train_lp, path = CONFIG['FIT_PATH'], num_predictions=num_predictions).best_hypotheses(n=1)
            print("ConSyn ends..")

            avg_rt_consyn = (time.time() - start_time)/num_predictions

            train_f1_consyn = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_consyn)}),
                                                pos=train_lp.pos, neg=train_lp.neg)
            
            if test_lp:                                    
                test_f1_consyn = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_consyn)}),
                                                pos=test_lp.pos, neg=test_lp.neg)
            
            print(f"Target concept: {str_target_concept}", end="\t")
            print(f"Predicted concept: {owl_expression_to_dl(pred_consyn)}", end="\t")
            print(f"Train Quality: {train_f1_consyn:.4f}", end="\t")
            if test_lp:
                print(f"Test Quality: {test_f1_consyn:.4f}", end="\t")
            print(f"Runtime: {avg_rt_consyn:.4f}")

            # trainer.cshs.display(paradigm="fit")
            trainer.cshs.clear(paradigm='fit')
            print()
            print()