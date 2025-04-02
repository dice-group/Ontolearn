import os
import time
import random
import itertools
import argparse
import ast
from itertools import chain

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoConfig

from owlapy import owl_expression_to_dl
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectOneOf,
)
from owlapy.parser import DLSyntaxParser

from ontolearn.utils import (
    jaccard_similarity,
    f1_set_similarity,
    concept_reducer,
    concept_reducer_properties,
)

from cnir import InferenceDataset
from cnir.config import CNIRConfig
from cnir.models import CNIRComposite
from cnir.models.pmanet import PMAnet
from cnir.utils import str2bool, read_embs_and_apply_agg

"""
How to run? E.g.,

cnir --dataset_dir "your dataset dir" --output_dir "your dataset dir" --model "composite" 
     --use_pma True --pretrained_model_path "" \
     --ratio_sample_object_prob 0.9 --ratio_sample_nc 0.9 --pma_model_path "" \
     --path_kge_model ""  
"""

def execute(args):
    dataset_dir = args.dataset_dir
    # Fix the random seed.
    seed = args.seed
    random.seed(seed)
    if args.use_pma:
        pma_net = PMAnet(CNIRConfig().embedding_dim, CNIRConfig().num_attention_heads, 1)
        pma_net.load_state_dict(
            torch.load(args.pma_model_path, map_location="cpu", weights_only=True))
    kb, all_individuals, embeddings = read_embs_and_apply_agg(args.dataset_dir, nn_agg=pma_net,
                                                              merge=True)
    all_individuals_set = set(all_individuals)
    all_individuals_arr = np.array(sorted(all_individuals), dtype=object)
    all_ind_embs = torch.FloatTensor(
        embeddings.loc[all_individuals_arr].values)  # .to(device)
    kb_namespace = list(kb.ontology.classes_in_signature())[0].str
    if "#" in kb_namespace:
        kb_namespace = kb_namespace.split("#")[0] + "#"
    elif "/" in kb_namespace:
        kb_namespace = kb_namespace[:kb_namespace.rfind("/")] + "/"
    elif ":" in kb_namespace:
        kb_namespace = kb_namespace[:kb_namespace.rfind(":")] + ":"
    expression_parser = DLSyntaxParser(kb_namespace)

    # R: Extract object properties.
    object_properties = {i for i in kb.get_object_properties()}
    ratio_sample_object_prob = args.ratio_sample_object_prob
    ratio_sample_nc = args.ratio_sample_nc
    # Subsample if required.
    if ratio_sample_object_prob:
        object_properties = {i for i in random.sample(population=list(object_properties),
                                                      k=max(1, int(len(
                                                          object_properties) * ratio_sample_nc)))}
    # R⁻: Inverse of object properties.
    object_properties_inverse = {i.get_inverse_property() for i in object_properties}
    # (5) R*: R UNION R⁻.
    object_properties_and_inverse = object_properties.union(object_properties_inverse)
    # NC: Named owl concepts.
    nc = {i for i in kb.get_concepts()}

    if ratio_sample_nc:
        # Subsample if required.
        nc = {i for i in
              random.sample(population=list(nc), k=max(1, int(len(nc) * ratio_sample_nc)))}

    # NC⁻: Complement of NC.
    nnc = {i.get_object_complement_of() for i in nc}
    # UNNC: NC UNION NC⁻.
    unnc = nc.union(nnc)

    # Retrieve 10 random Nominals.
    nominals = set(random.sample(kb.individuals(), 3))
    # All Combinations of 3 for Nominals.
    nominal_combinations = set(
        OWLObjectOneOf(combination)
        for combination in itertools.combinations(nominals, 3)
    )
    # NC UNION NC.
    unions = concept_reducer(nc, opt=OWLObjectUnionOf)
    # NC INTERSECTION NC.
    intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
    # UNNC UNION UNNC.
    unions_unnc = concept_reducer(unnc, opt=OWLObjectUnionOf)
    # UNNC INTERACTION UNNC.
    intersections_unnc = concept_reducer(unnc, opt=OWLObjectIntersectionOf)
    # \exist r. C s.t. C \in UNNC and r \in R* .
    exist_unnc = concept_reducer_properties(
        concepts=unnc,
        properties=object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )
    # \forall r. C s.t. C \in UNNC and r \in R* .
    for_all_unnc = concept_reducer_properties(
        concepts=unnc,
        properties=object_properties_and_inverse,
        cls=OWLObjectAllValuesFrom,
    )
    # >= n r. C  and =< n r. C, s.t. C \in UNNC and r \in R* .
    min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3 = (
        concept_reducer_properties(
            concepts=unnc,
            properties=object_properties_and_inverse,
            cls=OWLObjectMinCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3 = (
        concept_reducer_properties(
            concepts=unnc,
            properties=object_properties_and_inverse,
            cls=OWLObjectMaxCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    # \exist r. Nominal s.t. Nominal \in Nominals and r \in R* .
    exist_nominals = concept_reducer_properties(
        concepts=nominal_combinations,
        properties=object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )

    # Collect the result.
    results = []
    # Converted to list so that the progress bar works.
    concepts = list(
        chain(
            nc, unions, intersections, nnc, unions_unnc, intersections_unnc,
            exist_unnc, for_all_unnc,
            min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3,
            max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3,
            exist_nominals,
        )
    )
    random.shuffle(concepts)


    if args.model.lower() == "composite":
        CNIRConfig.batch_training = False
        AutoConfig.register("cnir", CNIRConfig)
        AutoModel.register(CNIRConfig, CNIRComposite)
        model = AutoModel.from_pretrained(args.pretrained_model_path)
        print("\n\x1b[6;30;42mSuccessfully loaded a pretrained model!\x1b[0m\n")

    def predict_cnir(model, expr, all_individuals_set, all_individuals_arr, all_ind_embs,
                     embeddings, th):
        start_time = time.time()
        model.eval()
        if isinstance(expr, str):
            expr = [expr]
        data = InferenceDataset(data=expr, all_individuals=all_individuals_set,
                                concept_to_instance_set=None,
                                embeddings=embeddings)
        for i in range(len(data)):
            expr, component_embeddings_dict = data[i]
            output = model(expr, all_ind_embs, component_embeddings_dict)
            retrieved = set(all_individuals_arr[np.where(output > th)])
        time_taken = time.time() - start_time
        return retrieved, time_taken

    # Retrieval Results
    def concept_retrieval(expr):
        start_time = time.time()
        actual = set([ind.str.split("/")[-1] for ind in kb.individuals(expression_parser.parse(expr))])
        #print(f"Actual: {actual}")
        return actual, time.time() - start_time
    counter = 0
    for expression in (tqdm_bar := tqdm(concepts, position=0, leave=True)):
        expr = owl_expression_to_dl(expression)
        try:
            retrieval_y, runtime_y = predict_cnir(model, expr, all_individuals_set,
                                                  all_individuals_arr,
                                                  all_ind_embs, embeddings, args.th)
        except Exception as e: # Catch any exception.
            error_log_path = os.path.join(args.output_dir, "error_log.txt")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"Time: {time.ctime()}\n")
                f.write(f"Error: {e} for expression: {expr}\n")
            continue
        retrieval_kb_y, runtime_kb_y = concept_retrieval(expr)
        jaccard_sim = jaccard_similarity(retrieval_y, retrieval_kb_y)
        # Compute the F1-score.
        f1_sim = f1_set_similarity(retrieval_y, retrieval_kb_y)
        # Store the data.
        results.append(
            {
                "Expression": owl_expression_to_dl(expression),
                "Type": type(expression).__name__,
                "Jaccard Similarity": jaccard_sim,
                "F1": f1_sim,
                "Runtime Benefits": runtime_y - runtime_kb_y,
                "Runtime kb": runtime_kb_y,
                "cnir_Retrieval": retrieval_y,
                "kb_Retrieval": retrieval_kb_y,
            }
        )
        # Update the progress bar.
        tqdm_bar.set_description_str(
            f"Expression {counter}: {owl_expression_to_dl(expression)} | Jaccard Similarity:{jaccard_sim:.4f} | F1 :{f1_sim:.4f} | Runtime Benefits:{runtime_y - runtime_kb_y:.3f}"
        )
        counter += 1
    # Read the data into pandas dataframe
    df = pd.DataFrame(results)
    # Save the experimental results into csv file.
    output_path_name = os.path.join(args.output_dir, f"{args.model}_results.csv")
    df.to_csv(output_path_name)
    print("\n\x1b[6;30;42mSuccessfully saved the results!\x1b[0m\n")
    del df
    # Load the saved CSV file.
    df = pd.read_csv(output_path_name, index_col=0,
                     converters={'cnir_Retrieval': lambda x: ast.literal_eval(x),
                                 'kb_Retrieval': lambda x: ast.literal_eval(
                                     x)})
    # A retrieval result can be parsed into  set of instances to python object.
    x = df["kb_Retrieval"].iloc[0]
    assert isinstance(x, set)
    # Extract the numerical features.
    numerical_df = df.select_dtypes(include=["number"])
    # Extract the type of owl concepts
    df_g = df.groupby(by="Type")
    print(df_g["Type"].count())
    mean_df = df_g[numerical_df.columns].mean()
    print(mean_df)

def get_default_arguments(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="The path of a folder containing training_data.json, which contains a list of class expressions or list of tuples where the first elements are class expressions."
                             ",e.g., datasets/carcinogenesis/data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The location where to store the trained model and training results.")
    parser.add_argument("--th", type=float, default=0.5,
                        help="Threshold on probabilities to decide which individual is an instance of a class expression")
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
    parser.add_argument("--path_kge_model", type=str, default=None)
    parser.add_argument("--path_kg", type=str, default="KGs/Family/father.owl")
    parser.add_argument("--ratio_sample_object_prob", type=float, default=None,
                        help="To sample OWL Object Properties.")
    parser.add_argument("--ratio_sample_nc", type=float, default=None,
                        help="To sample OWL Classes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = get_default_arguments()
    dataset_dir = "D:\PycharmProjects\CoNeuralReasoner\datasets\datasets\mutagenesis"
    args.dataset_dir = dataset_dir
    output_dir = "D:\PycharmProjects\CoNeuralReasoner\output\composite"
    args.output_dir = output_dir
    args.model = 'composite'
    args.use_pma = True
    args.pretrained_model_path = 'D:\PycharmProjects\CoNeuralReasoner\output\composite'
    args.ratio_sample_object_prob = 0.01
    args.ratio_sample_nc = 0.01
    pma_model_path = "D:\PycharmProjects\CoNeuralReasoner\output\model.pt"
    args.pma_model_path = pma_model_path
    args.path_kge_model = "D:\PycharmProjects\CoNeuralReasoner\datasets\datasets\mutagenesis_kb_ontology_owl"
    execute(args)


if __name__ == "__main__":
    main()
