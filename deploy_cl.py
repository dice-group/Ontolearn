import pandas as pd
import torch
import gradio as gr
from argparse import ArgumentParser
import random
import os

from ontolearn.ea_algorithms import EASimple
from ontolearn.ea_initialization import EARandomWalkInitialization, RandomInitMethod, EARandomInitialization
from ontolearn.fitness_functions import LinearPressureFitness
from ontolearn.heuristics import CELOEHeuristic, OCELHeuristic
from ontolearn.metrics import Accuracy, F1, Recall, Precision, WeightedAccuracy
from ontolearn.concept_learner import EvoLearner, CELOE, NCES, OCEL
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.value_splitter import EntropyValueSplitter, BinningValueSplitter
from ontolearn.owlapy.model import OWLNamedIndividual, IRI
from ontolearn.owlapy.render import DLSyntaxObjectRenderer

metrics = {'F1': F1,
           'Accuracy': Accuracy,
           'Recall': Recall,
           'Precision': Precision,
           'WeightedAccuracy': WeightedAccuracy
           }
renderer = DLSyntaxObjectRenderer()


def compute_quality(KB, solution, pos, neg, qulaity_func="F1"):
    func = metrics[qulaity_func]().score2
    instances = set(KB.individuals(solution))
    if isinstance(list(pos)[0], str):
        instances = {ind.get_iri().as_str().split("/")[-1] for ind in instances}
    tp = len(pos.intersection(instances))
    fn = len(pos.difference(instances))
    fp = len(neg.intersection(instances))
    tn = len(neg.difference(instances))
    return func(tp=tp, fn=fn, fp=fp, tn=tn)[-1]


def setup_prerequisites(individuals, pos_ex, neg_ex, random_ex: bool, size_of_ex):

    # start_time = time.time()
    if (pos_ex == "" or neg_ex == "") and not random_ex:
        return None, "No examples to run the algorithm! Please enter the pos and neg examples."

    if random_ex:
        typed_pos = set(random.sample(individuals, int(size_of_ex)))
        remaining = list(set(individuals)-typed_pos)
        typed_neg = set(random.sample(remaining, min(len(remaining), int(size_of_ex))))
        pos_str = [pos_ind.get_iri().as_str() for pos_ind in typed_pos]
        neg_str = [neg_ind.get_iri().as_str() for neg_ind in typed_neg]
    else:
        pos_str = pos_ex.replace(" ", "").replace("\n", "").replace("\"", "").split(",")
        neg_str = neg_ex.replace(" ", "").replace("\n", "").replace("\"", "").split(",")
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, pos_str)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, neg_str)))

    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

    if len(typed_pos) < 20:
        s = f'E\u207A: {", ".join(pos_str)}\nE\u207B: {", ".join(neg_str)}\n \n'
    else:
        s = f'|E\u207A|: {len(pos_str)}\n|E\u207B|: {len(neg_str)}\n'

    return lp, s


# kb: ../KGs/father.owl
# pos: http://example.com/father#markus,http://example.com/father#martin,http://example.com/father#stefan
# neg: http://example.com/father#anna,http://example.com/father#heinz,http://example.com/father#michelle


# ------------------------------------------------EvoLearner-----------------------------------------------------------
def launch_evolearner(args):

    kb = KnowledgeBase(path=args.path_knowledge_base)
    individuals = list(kb.individuals())

    def predict(positive_examples, negative_examples, random_examples, size_of_examples,
                quality_func, terminate_on_goal, max_runtime, use_data_properties, use_card_restrictions,
                use_inverse, tournament_size, card_limit, population_size, num_generations, height_limit, gain,
                penalty, max_t, jump_pr, crossover_pr, mutation_pr, elitism, elite_size, min_height,
                max_height, method, max_nr_splits):

        top3reports = list()
        lp, s = setup_prerequisites(individuals, positive_examples, negative_examples, random_examples,
                                    size_of_examples)
        if lp is None:
            return s, None

        fit_func = LinearPressureFitness(gain, penalty)
        init_method = EARandomWalkInitialization(int(max_t), jump_pr)
        algorithm = EASimple(crossover_pr, mutation_pr, elitism, elite_size)
        mut_uniform_gen = EARandomInitialization(int(min_height), int(max_height), getattr(RandomInitMethod, method))
        value_splitter = EntropyValueSplitter(int(max_nr_splits))

        model = EvoLearner(knowledge_base=kb, quality_func=metrics[quality_func](), fitness_func=fit_func,
                           init_method=init_method, algorithm=algorithm, mut_uniform_gen=mut_uniform_gen,
                           value_splitter=value_splitter, terminate_on_goal=terminate_on_goal, max_runtime=max_runtime,
                           use_data_properties=use_data_properties, use_card_restrictions=use_card_restrictions,
                           use_inverse=use_inverse, tournament_size=int(tournament_size), card_limit=int(card_limit),
                           population_size=int(population_size), num_generations=int(num_generations),
                           height_limit=int(height_limit))
        with torch.no_grad():
            model.fit(lp, verbose=False)
            hypotheses = list(model.best_hypotheses(n=3))
            for h in hypotheses:
                report = {'Prediction': renderer.render(h.concept),
                          f'Quality({quality_func})': h.quality,
                          'Tree Length': h.tree_length,
                          'Tree Depth': h.tree_depth,
                          'Individuals': h.individuals_count,
                          }
                top3reports.append(report)

        return s, pd.DataFrame([report for report in top3reports])

    with gr.Blocks(title="EvoLearner") as evolearner_interface:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Set Examples"):
                    gr.Markdown("Set examples (separated by comma)")
                    i1 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#pos_id1, "
                                                         "http://example.com/placeholder#pos_id2",
                                    label='Positive Examples')
                    i2 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#neg_id1, "
                                                         "http://example.com/placeholder#neg_id2",
                                    label='Negative Examples')
                    with gr.Row():
                        i3 = gr.Checkbox(label="Random Examples", info="If you select this option, you can set the "
                                                                       "number of the random set by using the slider.")
                        i4 = gr.Slider(minimum=1, label="Size of the random set", maximum=len(individuals) - 1,
                                       info="The slider has no effect if the 'Random Examples' option is not selected.")
                with gr.Tab("Algorithm Settings"):
                    gr.Markdown("General arguments")
                    with gr.Box():
                        i5 = gr.Dropdown(label="Quality function", choices=["F1", "Accuracy", "Recall", "Precision",
                                                                            "WeightedAccuracy"], value="F1")
                        with gr.Row():
                            i6 = gr.Checkbox(label="Terminate on goal", value=True)
                            i8 = gr.Checkbox(label="Use data properties", value=True)
                            i9 = gr.Checkbox(label="Use card restrictions", value=True)
                            i10 = gr.Checkbox(label="Use inverse", value=False)
                        with gr.Row():
                            i7 = gr.Number(label="Maximum runtime", value=600)
                            i11 = gr.Number(label="Tournament size", value=7)
                            i13 = gr.Number(label="Population size", value=800)
                        with gr.Row():
                            i14 = gr.Number(label="Num generations", value=200)
                            i12 = gr.Number(label="Card limit", value=10)
                            i15 = gr.Number(label="Height limit", value=17)
                    gr.Markdown("Set arguments for the fitness function (LinearPressureFitness)")
                    with gr.Box():
                        with gr.Row():
                            i16 = gr.Number(label="Gain", value=2048)
                            i17 = gr.Number(label="Penalty", value=1)
                    gr.Markdown("Set arguments for the initialization function (EARandomWalkInitialization)")
                    with gr.Box():
                        with gr.Row():
                            i18 = gr.Number(label="Number of paths (max_t)", value=2)
                            i19 = gr.Number(label="Jump probability", value=0.5, info="Probability to explore paths of "
                                                                                      "length 2")
                    gr.Markdown("Set arguments for the evolutionary algorithm (EASimple)")
                    with gr.Box():
                        with gr.Row():
                            i20 = gr.Number(label="Crossover probability", value=0.9)
                            i21 = gr.Number(label="Mutation probability", value=0.1)
                        with gr.Row():
                            i22 = gr.Checkbox(label="Elitism", value=False)
                            i23 = gr.Number(label="Elite size", value=0.1)
                    gr.Markdown("Set arguments for the evolutionary initialization method (EARandomInitialization)")
                    with gr.Box():
                        with gr.Row():
                            i24 = gr.Number(label="Minimum height", value=1, info="Minimum height of trees")
                            i25 = gr.Number(label="Maximum height", value=3, info="Maximum height of trees")
                        i26 = gr.Dropdown(label="Random initialization method", choices=["GROW", "FULL",
                                                                                         "RAMPED_HALF_HALF"],
                                          value="RAMPED_HALF_HALF")
                    gr.Markdown("Set arguments for the data properties value splitter (EntropyValueSplitter)")
                    with gr.Box():
                        i27 = gr.Number(label="Maximum number of splits", value=2)
                submit_btn = gr.Button("Run")
            with gr.Column():
                o1 = gr.Textbox(label='Learning Problem')
                o2 = gr.Dataframe(label='Predictions', type='pandas')
        submit_btn.click(predict, inputs=[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16,
                                          i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27], outputs=[o1, o2])
    evolearner_interface.launch(share=True)


# -----------------------------------------------CELOE------------------------------------------------------------------

def launch_celoe(args):

    kb = KnowledgeBase(path=args.path_knowledge_base)
    individuals = list(kb.individuals())

    def predict(positive_examples, negative_examples, random_examples: bool, size_of_examples,
                quality_func, terminate_on_goal: bool, max_runtime, iter_bound, max_num_of_concepts_tested,
                max_results, best_only, calculate_min_max, gain_bonus_factor, start_node_bonus,
                node_refinement_penalty, expansion_penalty_factor, max_nr_splits, max_child_length,
                use_negation, use_all_constructor, use_inverse, use_card_restrictions, use_numeric_datatypes,
                use_time_datatypes, use_boolean_datatype, card_limit):

        top3reports = list()
        lp, s = setup_prerequisites(individuals, positive_examples, negative_examples, random_examples,
                                    size_of_examples)
        if lp is None:
            return s, None

        heur_func = CELOEHeuristic(gainBonusFactor=gain_bonus_factor, startNodeBonus=start_node_bonus,
                                   nodeRefinementPenalty=node_refinement_penalty,
                                   expansionPenaltyFactor=expansion_penalty_factor)

        refinement_op = ModifiedCELOERefinement(kb, BinningValueSplitter(int(max_nr_splits)), int(max_child_length),
                                                use_negation, use_all_constructor, use_inverse, use_card_restrictions,
                                                use_numeric_datatypes, use_time_datatypes, use_boolean_datatype,
                                                int(card_limit))

        model = CELOE(knowledge_base=kb, quality_func=metrics[quality_func](), heuristic_func=heur_func,
                      refinement_operator=refinement_op, terminate_on_goal=terminate_on_goal,
                      max_runtime=int(max_runtime), iter_bound=int(iter_bound),
                      max_num_of_concepts_tested=int(max_num_of_concepts_tested), max_results=int(max_results),
                      best_only=best_only, calculate_min_max=calculate_min_max)
        with torch.no_grad():
            model.fit(lp)
            hypotheses = list(model.best_hypotheses(n=3))
            for h in hypotheses:
                report = {'Prediction': renderer.render(h.concept),
                          f'Quality({quality_func})': h.quality,
                          'Heuristic': h.heuristic,
                          'Depth': h.depth(),
                          }
                top3reports.append(report)

        return s, pd.DataFrame([report for report in top3reports])

    with gr.Blocks(title="CELOE") as celoe_interface:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Set Examples"):
                    gr.Markdown("Set examples (separated by comma)")
                    i1 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#pos_id1, "
                                                         "http://example.com/placeholder#pos_id2",
                                    label='Positive Examples')
                    i2 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#neg_id1, "
                                                         "http://example.com/placeholder#neg_id2",
                                    label='Negative Examples')
                    with gr.Row():
                        i3 = gr.Checkbox(label="Random Examples", info="If you select this option, you can set the "
                                                                       "number of the random set by using the slider.")
                        i4 = gr.Slider(minimum=1, label="Size of the random set", maximum=len(individuals) - 1,
                                       info="The slider has no effect if the 'Random Examples' option is not selected.")
                with gr.Tab("Algorithm Settings"):
                    gr.Markdown("General arguments")
                    with gr.Box():
                        i5 = gr.Dropdown(label="Quality function", choices=["F1", "Accuracy", "Recall", "Precision",
                                                                            "WeightedAccuracy"], value="F1")
                        with gr.Row():
                            i7 = gr.Number(label="Maximum runtime", value=5)
                            i8 = gr.Number(label="Iterations bound", value=10_000)
                        with gr.Row():
                            i9 = gr.Number(label="Maximum number of concepts tested", value=10_000)
                            i10 = gr.Number(label="Maximum results to find (not to show)", value=10)
                        with gr.Row():
                            i6 = gr.Checkbox(label="Terminate on goal", value=True)
                            i11 = gr.Checkbox(label="Best results only", value=False)
                            i12 = gr.Checkbox(label="Calculate min max", info="Only for statistical purpose, "
                                                                              "it does not influence CELOE", value=True)
                    gr.Markdown("Set arguments for the heuristic function (CELOEHeuristic)")
                    with gr.Box():
                        with gr.Row():
                            i13 = gr.Number(label="Gain bonus factor", value=0.3,
                                            info="Factor that weighs the increase in quality compared to the parent "
                                                 "node")
                            i14 = gr.Number(label="Start node bonus", value=0.1, info="Special value added to the root "
                                                                                      "node")
                        with gr.Row():
                            i15 = gr.Number(label="Node refinement penalty", value=0.001,
                                            info="The value that is subtracted from the heuristic for each refinement"
                                                 " attempt of this node")
                            i16 = gr.Number(label="Expansion penalty factor", value=0.1,
                                            info="The value that is subtracted from the heuristic for each horizontal "
                                                 "expansion of this")
                    gr.Markdown("Set arguments for the refinement operator (ModifiedCELOERefinement)")
                    with gr.Box():
                        with gr.Row():
                            i17 = gr.Number(label="Maximum number of splits", value=12,
                                            info="For the value splitter: BinningValueSplitter")
                            i18 = gr.Number(label="Maximum child length", value=10, info="\n")
                        with gr.Row():
                            i22 = gr.Checkbox(label="Use card restrictions", value=True)
                            i26 = gr.Number(label="Card limit", value=10)
                        with gr.Row():
                            i19 = gr.Checkbox(label="Use negation", value=True)
                            i20 = gr.Checkbox(label="Use all constructors", value=True)
                            i21 = gr.Checkbox(label="Use inverse", value=True)
                        with gr.Row():
                            i23 = gr.Checkbox(label="Use numeric data types", value=True)
                            i24 = gr.Checkbox(label="Use time datatypes", value=True)
                            i25 = gr.Checkbox(label="Use boolean datatypes", value=True)

                submit_btn = gr.Button("Run")
            with gr.Column():
                o1 = gr.Textbox(label='Learning Problem')
                o2 = gr.Dataframe(label='Predictions', type='pandas')
        submit_btn.click(predict, inputs=[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16,
                                          i17, i18, i19, i20, i21, i22, i23, i24, i25, i26], outputs=[o1, o2])
    celoe_interface.launch(share=True)

# -----------------------------------------------OCEL------------------------------------------------------------------


def launch_ocel(args):

    kb = KnowledgeBase(path=args.path_knowledge_base)
    individuals = list(kb.individuals())

    def predict(positive_examples, negative_examples, random_examples: bool, size_of_examples,
                quality_func, terminate_on_goal: bool, max_runtime, iter_bound, max_num_of_concepts_tested,
                max_results, best_only, calculate_min_max, gain_bonus_factor, expansion_penalty_factor,
                max_nr_splits, max_child_length,
                use_negation, use_all_constructor, use_inverse, use_card_restrictions, use_numeric_datatypes,
                use_time_datatypes, use_boolean_datatype, card_limit):

        top3reports = list()
        lp, s = setup_prerequisites(individuals, positive_examples, negative_examples, random_examples,
                                    size_of_examples)
        if lp is None:
            return s, None

        heur_func = OCELHeuristic(gainBonusFactor=gain_bonus_factor, expansionPenaltyFactor=expansion_penalty_factor)

        refinement_op = ModifiedCELOERefinement(kb, BinningValueSplitter(int(max_nr_splits)), int(max_child_length),
                                                use_negation, use_all_constructor, use_inverse, use_card_restrictions,
                                                use_numeric_datatypes, use_time_datatypes, use_boolean_datatype,
                                                int(card_limit))

        model = OCEL(knowledge_base=kb, quality_func=metrics[quality_func](), heuristic_func=heur_func,
                     refinement_operator=refinement_op, terminate_on_goal=terminate_on_goal,
                     max_runtime=int(max_runtime), iter_bound=int(iter_bound),
                     max_num_of_concepts_tested=int(max_num_of_concepts_tested), max_results=int(max_results),
                     best_only=best_only, calculate_min_max=calculate_min_max)

        with torch.no_grad():
            model.fit(lp)
            hypotheses = list(model.best_hypotheses(n=3))
            for h in hypotheses:
                report = {'Prediction': renderer.render(h.concept),
                          f'Quality({quality_func})': h.quality,
                          'Heuristic': h.heuristic,
                          'Depth': h.depth(),
                          }
                top3reports.append(report)

        return s, pd.DataFrame([report for report in top3reports])

    with gr.Blocks(title="OCEL") as ocel_interface:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Set Examples"):
                    gr.Markdown("Set examples (separated by comma)")
                    i1 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#pos_id1, "
                                                         "http://example.com/placeholder#pos_id2",
                                    label='Positive Examples')
                    i2 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#neg_id1, "
                                                         "http://example.com/placeholder#neg_id2",
                                    label='Negative Examples')
                    with gr.Row():
                        i3 = gr.Checkbox(label="Random Examples", info="If you select this option, you can set the "
                                                                       "number of the random set by using the slider.")
                        i4 = gr.Slider(minimum=1, label="Size of the random set", maximum=len(individuals) - 1,
                                       info="The slider has no effect if the 'Random Examples' option is not selected.")
                with gr.Tab("Algorithm Settings"):
                    gr.Markdown("General arguments")
                    with gr.Box():
                        i5 = gr.Dropdown(label="Quality function", choices=["F1", "Accuracy", "Recall", "Precision",
                                                                            "WeightedAccuracy"], value="F1")
                        with gr.Row():
                            i7 = gr.Number(label="Maximum runtime", value=5)
                            i8 = gr.Number(label="Iterations bound", value=10_000)
                        with gr.Row():
                            i9 = gr.Number(label="Maximum number of concepts tested", value=10_000)
                            i10 = gr.Number(label="Maximum results to find (not to show)", value=10)
                        with gr.Row():
                            i6 = gr.Checkbox(label="Terminate on goal", value=True)
                            i11 = gr.Checkbox(label="Best results only", value=False)
                            i12 = gr.Checkbox(label="Calculate min max", info="Only for statistical purpose, "
                                                                              "it does not influence OCEL", value=True)
                    gr.Markdown("Set arguments for the heuristic function (OCELHeuristic)")
                    with gr.Box():
                        with gr.Row():
                            i13 = gr.Number(label="Gain bonus factor", value=0.3,
                                            info="Factor that weighs the increase in quality compared to the parent "
                                                 "node")
                            i14 = gr.Number(label="Expansion penalty factor", value=0.1,
                                            info="The value that is subtracted from the heuristic for each horizontal "
                                                 "expansion of this")
                    gr.Markdown("Set arguments for the refinement operator (ModifiedCELOERefinement)")
                    with gr.Box():
                        with gr.Row():
                            i15 = gr.Number(label="Maximum number of splits", value=12,
                                            info="For the value splitter: BinningValueSplitter")
                            i16 = gr.Number(label="Maximum child length", value=10, info="\n")
                        with gr.Row():
                            i20 = gr.Checkbox(label="Use card restrictions", value=True)
                            i24 = gr.Number(label="Card limit", value=10)
                        with gr.Row():
                            i17 = gr.Checkbox(label="Use negation", value=True)
                            i18 = gr.Checkbox(label="Use all constructors", value=True)
                            i19 = gr.Checkbox(label="Use inverse", value=True)
                        with gr.Row():
                            i21 = gr.Checkbox(label="Use numeric data types", value=True)
                            i22 = gr.Checkbox(label="Use time datatypes", value=True)
                            i23 = gr.Checkbox(label="Use boolean datatypes", value=True)

                submit_btn = gr.Button("Run")
            with gr.Column():
                o1 = gr.Textbox(label='Learning Problem')
                o2 = gr.Dataframe(label='Predictions', type='pandas')
        submit_btn.click(predict, inputs=[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16,
                                          i17, i18, i19, i20, i21, i22, i23, i24], outputs=[o1, o2])
    ocel_interface.launch(share=True)


# -----------------------------------------------NCES------------------------------------------------------------------


def launch_nces(args):

    kb = KnowledgeBase(path=args.path_knowledge_base)
    individuals = list(kb.individuals())
    # kb_display_value = args.path_knowledge_base.split("/")[-1]

    def predict(positive_examples, negative_examples, random_examples: bool, size_of_examples, quality_func,
                learner_name, proj_dim, rnn_n_layers, drop_prob, num_heads, num_seeds, num_inds, ln, learning_rate,
                decay_rate, clip_value, batch_size, num_workers, max_length, load_pretrained, sorted_examples,
                pretrained_model_name):

        lp, s = setup_prerequisites(individuals, positive_examples, negative_examples, random_examples,
                                    size_of_examples)
        if lp is None:
            return s, None

        model = NCES(knowledge_base_path=args.path_knowledge_base, learner_name=learner_name,
                     path_of_embeddings=args.path_knowledge_base_embeddings, num_seeds=int(num_seeds),
                     load_pretrained=load_pretrained, max_length=int(max_length), proj_dim=int(proj_dim),
                     rnn_n_layers=int(rnn_n_layers), drop_prob=drop_prob, num_heads=int(num_heads),
                     pretrained_model_name=pretrained_model_name, ln=ln,
                     num_inds=int(num_inds), learning_rate=learning_rate, decay_rate=decay_rate, clip_value=clip_value,
                     batch_size=int(batch_size), num_workers=int(num_workers), sorted_examples=sorted_examples)

        with torch.no_grad():
            hypotheses = model.fit(lp.pos, lp.neg)
            report = {'Prediction': renderer.render(hypotheses),
                      f'Quality({quality_func})': compute_quality(kb, hypotheses, lp.pos, lp.neg, quality_func),
                      'Individuals': kb.individuals_count(hypotheses)}

        return s, pd.DataFrame([report])

    with gr.Blocks(title="NCES") as nces_interface:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Set Examples"):
                    gr.Markdown("Set examples (separated by comma)")
                    i1 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#pos_id1, "
                                                         "http://example.com/placeholder#pos_id2",
                                    label='Positive Examples')
                    i2 = gr.Textbox(lines=5, placeholder="http://example.com/placeholder#neg_id1, "
                                                         "http://example.com/placeholder#neg_id2",
                                    label='Negative Examples')
                    with gr.Row():
                        i3 = gr.Checkbox(label="Random Examples", info="If you select this option, you can set the "
                                                                       "number of the random set by using the slider.")
                        i4 = gr.Slider(minimum=1, label="Size of the random set", maximum=len(individuals) - 1,
                                       info="The slider has no effect if the 'Random Examples' option is not selected.")
                with gr.Tab("Algorithm Settings"):
                    with gr.Box():
                        i5 = gr.Dropdown(label="Quality function", choices=["F1", "Accuracy", "Recall", "Precision",
                                                                            "WeightedAccuracy"], value="F1")
                    gr.Markdown("General arguments")
                    with gr.Box():
                        i6 = gr.Dropdown(label="Learner name", choices=["SetTransformer", "GRU", "LSTM"],
                                         value="SetTransformer")
                        with gr.Row():
                            i7 = gr.Number(label="Number of projection dimensions", value=128)
                            i8 = gr.Number(label="Number of RNN layers", value=2, info="only for LSTM and GRU")
                            i9 = gr.Number(label="Drop probability", value=0.1)
                        with gr.Row():
                            i10 = gr.Number(label="Number of heads", value=4)
                            i11 = gr.Number(label="Number of seeds", value=1, info="only for SetTransformer")
                            i12 = gr.Number(label="Number of inducing points", value=32, info="only for SetTransformer")
                        with gr.Row():
                            i14 = gr.Number(label="Learning rate", value=1e-4)
                            i15 = gr.Number(label="Decay rate", value=0)
                            i16 = gr.Number(label="Clip value", value=5)
                        with gr.Row():
                            i17 = gr.Number(label="Batch size", value=256)
                            i18 = gr.Number(label="Number of workers", value=8)
                            i19 = gr.Number(label="Maximum length", value=48)
                        with gr.Row():
                            i13 = gr.Checkbox(label="Layer normalization", value=False, info="only for SetTransformer")
                            i20 = gr.Checkbox(label="Load pretrained", value=True)
                            i21 = gr.Checkbox(label="Sorted examples", value=True)
                        i22 = gr.CheckboxGroup(label="Pretrained model name", choices=["SetTransformer", "GRU", "LSTM"],
                                               value="SetTransformer")
                submit_btn = gr.Button("Run")
            with gr.Column():
                o1 = gr.Textbox(label='Learning Problem')
                o2 = gr.Dataframe(label='Predictions', type='pandas')
        submit_btn.click(predict, inputs=[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16,
                                          i17, i18, i19, i20, i21, i22], outputs=[o1, o2])
    nces_interface.launch(share=True)


# -----------------------------------------------DRILL------------------------------------------------------------------


# def launch_drill(args):
#
#     kb = KnowledgeBase(path=args.path_knowledge_base)
#     individuals = list(kb.individuals())
#     # kb_display_value = args.path_knowledge_base.split("/")[-1]
#
#     def predict(md1, positive_examples, negative_examples, random_examples: bool, size_of_examples, md2, min_length,
#                 max_length, md3, n, min_num_problems,  num_diff_runs, min_num_instances, search_algo, md4,
#                 drill_first_out_channels, quality_func,
#                 iter_bound, max_num_of_concepts_tested, terminate_on_goal,
#                 max_len_replay_memory, batch_size, epsilon_decay, num_epochs_per_replay,
#                 num_episodes_per_replay, learning_rate, use_target_net,
#                 max_runtime, num_of_sequential_actions, num_episode, num_workers, max_child_length, md5,
#                 max_test_time_per_concept, k
#                 ):
#
#         # lp, s = setup_prerequisites(individuals, positive_examples, negative_examples, random_examples,
#         #                             size_of_examples)
#
#         lp = LearningProblemGenerator(knowledge_base=kb, min_length=min_length, max_length=max_length)
#         balanced_examples = lp.get_balanced_n_samples_per_examples(n=n,
#                                                                    min_num_problems=min_num_problems,
#                                                                    max_length=max_length, min_length=min_length,
#                                                                    num_diff_runs=num_diff_runs,
#                                                                    min_num_instances=min_num_instances,
#                                                                    search_algo=search_algo)
#
#         drill_average = DrillAverage(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
#                                      drill_first_out_channels=drill_first_out_channels,
#                                      quality_func=metrics[quality_func],
#                                      pretrained_model_path=args.pretrained_drill_sample_path, iter_bound=iter_bound,
#                                      max_num_of_concepts_tested=max_num_of_concepts_tested,
#                                      terminate_on_goal=terminate_on_goal,
#                                      max_len_replay_memory=max_len_replay_memory, batch_size=batch_size,
#                                      epsilon_decay=epsilon_decay, num_epochs_per_replay=num_epochs_per_replay,
#                                      num_episodes_per_replay=num_episodes_per_replay, learning_rate=learning_rate,
#                                      use_target_net=use_target_net, max_runtime=max_runtime, num_workers=num_workers,
#                                      num_of_sequential_actions=num_of_sequential_actions, num_episode=num_episode)
#
#         drill_sample = DrillSample(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
#                                    quality_func=metrics[quality_func],
#                                    pretrained_model_path=args.pretrained_drill_sample_path,
#                                    iter_bound=iter_bound,
#                                    max_num_of_concepts_tested=max_num_of_concepts_tested,
#                                    terminate_on_goal=terminate_on_goal,
#                                    max_len_replay_memory=max_len_replay_memory, batch_size=batch_size,
#                                    epsilon_decay=epsilon_decay, num_epochs_per_replay=num_epochs_per_replay,
#                                    learning_rate=learning_rate, max_runtime=max_runtime, num_workers=num_workers,
#                                    num_of_sequential_actions=num_of_sequential_actions, num_episode=num_episode,
#                                    max_child_length=max_child_length)
#
#         report = Experiments(max_test_time_per_concept=max_test_time_per_concept).start_KFold(k=k,
#                              dataset=balanced_examples, models=[drill_average, drill_sample])
#
#         return "tmp", pd.DataFrame([report])
#
#     gr.Interface(
#         fn=predict,
#         inputs=[set_examples_md, pos_textbox, neg_textbox, random_ex_checkbox,
#                 gr.Slider(minimum=1, label="Size of the random set", maximum=len(individuals)),
#                 gr.Markdown("Set arguments for the Learning problem generator"),
#                 gr.Number(label="Minimum length", value=3),
#                 gr.Number(label="Maximum length", value=6),
#                 gr.Markdown("Set arguments for the Balanced example withdrawal method"),
#                 gr.Number(label="Number of randomly created problems per concept", value=5),
#                 gr.Number(label="Minimum number problems", value=2),
#                 gr.Number(label="Number of different runs", value=1),
#                 gr.Number(label="Minimum number instances", value=1),
#                 gr.Dropdown(label="Search algorithm", choices=["strict-dfs", "dfs"], value="strict-dfs"),
#                 gr.Markdown("Set arguments for Drill models"),
#                 gr.Number(label="Drill first out channels(for DrillAverage only)", value=32),
#                 quality_func_dropdown,
#                 gr.Number(label="Iterations bound", value=10_000),
#                 gr.Number(label="Maximum number of concepts tested", value=10_000),
#                 terminate_on_goal_checkbox,
#                 gr.Number(label="Maximum length of replay memory", value=256),
#                 gr.Number(label="Batch size", value=1024),
#                 gr.Number(label="Epsilon decay", value=0.01),
#                 gr.Number(label="Number of epochs per replay", value=1),
#                 gr.Number(label="Number of episodes per replay(for DrillAverage only)", value=2),
#                 gr.Number(label="Learning rate", value=0.001),
#                 gr.Checkbox(label="Use target net", value=False),
#                 gr.Number(label="Maximum runtime", value=5),
#                 gr.Number(label="Number of workers", value=32),
#                 gr.Number(label="Number of sequential actions", value=3),
#                 gr.Number(label="Number of episodes", value=2),
#                 gr.Number(label="Maximum child length (for DrillSample only)", value=10),
#                 gr.Markdown("Set experiment properties"),
#                 gr.Number(label="Maximum time per concept", value=3),
#                 gr.Number(label="Number of CPUs used during batching", value=3),
#                 ],
#         outputs=[lp_textbox, results_dataframe],
#         title="Drill: Deep Reinforcement Learning for Refinement Operators in ALC",
#         description=description).launch()


def run(args):
    if args.model == "evolearner":
        launch_evolearner(args)
    elif args.model == "celoe":
        launch_celoe(args)
    elif args.model == "ocel":
        launch_ocel(args)
    elif args.model == "nces":
        launch_nces(args)
    # elif args.model == "drill":
    #     launch_drill(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    # ---- General ----
    parser.add_argument("--model", type=str, default='nces', choices=['evolearner', 'celoe', 'ocel', 'nces'],
                        help='The concept learning model. (default: nces) ')

    parser.add_argument("--path_knowledge_base", type=str, default='NCESData/family/family.owl',
                        help='Location of the knowledge base that you wish to use')

    # ---- DRILL, NCES only ----
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='NCESData/family/embeddings/ConEx_entity_embeddings.csv',
                        help='*Only for NCES*')

    # ---- DRILL only ----

    # parser.add_argument('--pretrained_drill_sample_path',
    #                     type=str, default='pre_trained_agents/DrillHeuristic_sampling/DrillHeuristic_sampling.pth',
    #                     help='*Only for DRILL* Provide a path of .pth file')
    # parser.add_argument('--pretrained_drill_avg_path',
    #                     type=str,
    #                     default='pre_trained_agents/DrillHeuristic_averaging/DrillHeuristic_averaging.pth',
    #                     help='*Only for DRILL* Provide a path of .pth file')

    args = parser.parse_args()

    if not os.path.exists("NCESData/") and args.model == "nces":
        print("\nWarning! You are trying to deploy NCES without the NCES data!")
        print(f"Please download the necessary files first: see ./download_external_resources.sh\n")
    elif not os.path.exists("KGs") and "KGs/" in args.path_knowledge_base:
        print("\nWarning! There is no 'KGs' folder!")
        print(f"Please download the datasets first: see ./download_external_resources.sh\n")
    else:
        run(args)
