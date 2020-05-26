import torch
import random
import matplotlib.pyplot as plt
from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data
from learners.base import DCL
import umap

path = '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=path)

data = Data(knowledge_base=kb)
rho = Refinement(kb)
refined_concepts = set()
concepts_to_be_refined = set()

params = {'num_of_concepts_to_be_refined': 10,
          'num_of_inputs_for_model': 10,  # data.num_individuals // 10
          'num_of_datapoints_per_concept': 10,
          'num_of_times_sampling_at_testing': 5,
          'show_topN': 3,
          }

concepts_to_be_refined.add(kb.thing)
while len(refined_concepts) < params['num_of_concepts_to_be_refined']:
    try:
        c = concepts_to_be_refined.pop()
    except KeyError as ind:
        print('Break')
        break
    if c in refined_concepts:
        continue
    #print(len(refined_concepts), '.th Concept ', c.str, ' is refined.')
    for i in rho.refine(c):
        concepts_to_be_refined.add(i)
    refined_concepts.add(c)

concepts = []
concepts.extend(refined_concepts)
concepts.extend(concepts_to_be_refined)

data.concepts_for_training(concepts)

params['num_instances'] = data.num_individuals
params['num_of_outputs'] = len(concepts)

print('Number of outputs',len(concepts))

model = DCL(params)
model.init()
opt = torch.optim.Adam(model.parameters())

for all_positives, all_negatives, TARGET_CONCEPT in data.train(params['num_of_inputs_for_model']):
    model.train()
    losses = []
    for it in range(params['num_of_datapoints_per_concept']):
        opt.zero_grad()

        X_pos = random.sample(all_positives, params['num_of_inputs_for_model'] // 2)
        X_neg = random.sample(all_negatives, params['num_of_inputs_for_model'] // 2)

        y = data.score(X_pos, X_neg)
        X = X_pos + X_neg

        X = torch.tensor(X)
        y = torch.tensor(y)

        y = torch.softmax(y, dim=0)
        predictions = model.forward(X)

        loss = model.loss(predictions.log(), y)

        loss.backward()
        opt.step()
        losses.append(loss.item())

    model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
    with torch.no_grad():  # Important:
        averaged_best_predictions = dict()
        for _ in range(params['num_of_times_sampling_at_testing']):
            X_pos = random.sample(all_positives, params['num_of_inputs_for_model'] // 2)
            X_neg = random.sample(all_negatives, params['num_of_inputs_for_model'] // 2)

            y = data.score(X_pos, X_neg)
            X = X_pos + X_neg

            X, y = torch.tensor(X), torch.tensor(y)
            y = torch.softmax(y, dim=0)  # True positive rates per classes converted to probability distribution

            predictions = model.forward(X)
            kl_div = (y * (y.log() - predictions.log())).sum()
            # print(
            #    '\nKL loss for positive and negative examples sampled from concept: {0}:{1}'.format(TARGET_CONCEPT.str,
            #                                                                                       kl_div))

            values, indices = torch.topk(predictions, k=len(concepts))  # Consider TOP10 next time

            for predicted_concept, prediction_score in zip(data.labels[indices], values):

                tp = len(TARGET_CONCEPT.instances & predicted_concept.instances)
                fn = len(TARGET_CONCEPT.instances - predicted_concept.instances)

                negatives = kb.thing.instances - TARGET_CONCEPT.instances
                fp = len(negatives & predicted_concept.instances)

                try:
                    recall = tp / (tp + fn)
                    precision = tp / (tp + fp)
                    f_1 = 2 * ((precision * recall) / (precision + recall))
                except:
                    f_1 = 0

                averaged_best_predictions.setdefault(predicted_concept, []).append(f_1)

        ranked = []
        for k, v in averaged_best_predictions.items():
            ranked.append((k, sum(v) / len(v)))

        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

        print('\nTarget:{0}\tnum of instances:{1}'.format(TARGET_CONCEPT.str, len(all_positives)))

        print('Top {0} Predictions'.format(params['show_topN']))
        for ith, pred in enumerate(ranked):
            ith_best_prediction, f_score = pred
            print('{0}.th {1}\tF1-score: {2}'.format(ith + 1, ith_best_prediction.str, f_score))
            if ith + 1 == params['show_topN']:
                break
reducer = umap.UMAP()

torch.save(model.state_dict(), kb.name + '_model.pt')
embeddings = model.state_dict()['embedding.weight']

low_embd = reducer.fit_transform(embeddings)

fig, ax = plt.subplots()
ax.scatter(low_embd[:, 0], low_embd[:, 1])

for i, txt in enumerate(data.individuals):
    ax.annotate(txt, (low_embd[i, 0], low_embd[i, 1]))







plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of {0}'.format(kb.name))

plt.show()
