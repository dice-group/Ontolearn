"""
differentiable_concept_learning.py illustrates one of our research idea.
"""
import random
import torch
from torch.utils.data import DataLoader
from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data
from learners.base import DeepT
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from core.util import create_experiment_folder, create_logger,TorchData
import pandas as pd
import numpy as np
import umap

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

storage_path, _ = create_experiment_folder(folder_name='../Log')

logger = create_logger(name='DeepT', p=storage_path)

path = '../data/family-benchmark_rich_background.owl'
# path = '../data/biopax.owl'
kb = KnowledgeBase(path=path)
logger.info('Deep tunnelling for Refinement Operator'.format())
logger.info('Knowledgebase:{0}'.format(kb.name))

data = Data(knowledge_base=kb, logger=logger)
rho = Refinement(kb)

params = {
    'learning_problem': 'concept_learning',
    'num_instances': len(kb.thing.instances),
    'num_dim': 2,
    'num_of_epochs': 10,
    'batch_size': 64,
    'num_of_concepts_refined': 1,
    'num_of_inputs_for_model': 3,  # data.num_individuals // 10,
    'root_concept': kb.thing,
    'num_of_times_sample_per_concept': 1,
    'refinement_operator': rho,
    'flag_for_plotting': False
}
# Generate concepts and prune those ones that do not satisfy the provided constraint.
concepts = [concept for concept in data.generate_concepts(**params) if
            len(kb.thing.instances) > (params['num_of_inputs_for_model'] + len(concept.instances)) > params[
                'num_of_inputs_for_model']]

concepts_train, concepts_test = train_test_split(concepts, test_size=0.3, random_state=RANDOM_SEED)

# Important decision: Apply Jaccard, PPMI, etc
labels = np.array(random.sample(concepts_train, 10))
# labels = np.array(concepts_train)

params['num_of_outputs'] = len(labels)

# Generate Training Data
X, y = data.convert_data(concepts_train, labels, params)
X = torch.tensor(X)
y = torch.softmax(torch.tensor(y), dim=1)  # F-scores are turned into f-score distributions.

dataloader = DataLoader(TorchData(X, y), batch_size=params['batch_size'], shuffle=True, num_workers=1)

logger.info('Number of unique concepts in training split:{0}\tNumber of data-points {1}'.format(len(labels), len(X)))

model = DeepT(params)
model.init()
opt = torch.optim.Adam(model.parameters())

logger.info(model)
####################################################################################################################
loss_per_epoch = []
model.train()
for it in range(1, params['num_of_epochs'] + 1):
    running_loss = 0.0

    if it % 100 == 0:
        print(it)

    for (x_batch, y_batch) in dataloader:
        predictions = model.forward(x_batch)
        loss = model.loss(predictions.log(), y_batch)  # for KL
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()

    loss_per_epoch.append(running_loss / len(dataloader))

    """
    for j in range(0, len(X), params['batch_size']):
        opt.zero_grad()

        x_batch = X[j:j + params['batch_size']]
        y_batch = y[j:j + params['batch_size']]

        print(x_batch.shape)
        predictions = model.forward(x_batch)

        print(predictions.shape)

        print(y_batch.shape)

        exit(1)

        # loss = model.loss(predictions.log(), y_batch)  # for KL
        loss = model.loss(predictions, y_batch)  # for CrossEnentro

        exit(1)
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()
    """
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
torch.save(model.state_dict(), storage_path + '/kb.name' + '_model.pt')

if params['flag_for_plotting']:
    plt.plot(loss_per_epoch)
    plt.grid(True)
    plt.savefig(storage_path + '/loss_history.pdf')
    plt.show()

    reducer = umap.UMAP()
    embeddings = model.state_dict()['embedding.weight']
    low_embd = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter(low_embd[:, 0], low_embd[:, 1])
    # for i, txt in enumerate(data.individuals):
    #    ax.annotate(txt, (low_embd[i, 0], low_embd[i, 1]))
    plt.show()

logger.info('Testing starts on:{0}'.format(len(concepts_test)))
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

params['num_of_times_sample_per_concept'] = 1

true_f1_scores = []
predicted_f_dist = []
f1_score_to_report = []
with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    for true_concept in concepts_test:
        k_inputs = []
        k_f_measures_per_label = []
        k_pred = []
        for k in range(10):
            try:
                x_pos, x_neg = data.pos_neg_sampling_from_concept(true_concept, params['num_of_inputs_for_model'])
                k_f_measures_per_label.append(data.score_with_labels(pos=x_pos, neg=x_neg, labels=labels))
                k_inputs.append([data.indx[i] for i in x_pos + x_neg])
                # input = torch.tensor(input).reshape(1, len(input))
            except ValueError:
                continue
        k_inputs = torch.tensor(np.array(k_inputs), dtype=torch.int64)

        if len(k_inputs) == 0:
            print('Can not test {0} num_instance:{1}'.format(true_concept, len(true_concept.instances)))
            continue

        predictions = model.forward(k_inputs)

        averaged_predicted_f1_dist = torch.mean(predictions, dim=0)

        # We use averaging as we can not make use of all individuals.
        # Save average predicted F-1 score distribution and average TRUE F1-scores
        predicted_f_dist.append(averaged_predicted_f1_dist.numpy())
        true_f1_scores.append(np.array(k_f_measures_per_label).mean(axis=0))

        # Save average predicted F-1 score distribution.
        values, indexes = torch.topk(averaged_predicted_f1_dist, 3)

        best_pred = labels[indexes]

        logger.info('Top {0} Predictions for true concept:{1} num_instance:{2}'.format(len(best_pred), true_concept.str,
                                                                                       len(true_concept.instances)))
        for ith, pred in enumerate(best_pred):
            f_1 = data.score_with_instances(pos=true_concept.instances, neg=kb.thing.instances - true_concept.instances,
                                            instances=pred.instances)

            if ith == 0:
                f1_score_to_report.append(f_1)

            logger.info(
                '{0}.th {1} with num_instance:{2}\tF-1 score:{3}'.format(ith + 1, pred.str, len(pred.instances), f_1))

f1_score_to_report = np.array(f1_score_to_report)
logger.info(
    'Mean and STD of F-1 score of 1.th predictions at testing:{0:.3f} +- {1:.3f}'.format(f1_score_to_report.mean(),
                                                                                         f1_score_to_report.std()))

if params['flag_for_plotting']:
    true_f1_scores = pd.DataFrame(np.array(true_f1_scores), columns=[x.str for x in labels])

    plt.matshow(true_f1_scores.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(true_f1_scores.columns))], true_f1_scores.columns, rotation=90)
    plt.yticks([i for i in range(len(true_f1_scores.columns))], true_f1_scores.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    plt.show()

    predicted_f_dist = pd.DataFrame(np.array(predicted_f_dist), columns=[x.str for x in labels])
    plt.matshow(predicted_f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(predicted_f_dist.columns))], predicted_f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(predicted_f_dist.columns))], predicted_f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    plt.show()

"""
logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)

    logger.info('Loss at testing: {0}'.format(loss.item()))

    f_dist = pd.DataFrame(y_test.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    # plt.show()

    f_dist = pd.DataFrame(predictions.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    # plt.show()
exit(1)

X, y, kw = data.generate_data(**params)

params.update(kw)
logger.info('Hyperparameters:{0}'.format(params))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logger.info('Number of concepts in training split:{0}'.format(len(X_train)))

model = DeepT(params)
model.init()
opt = torch.optim.Adam(model.parameters())

assert len(X_train) == len(y_train)  # for sanity check.

logger.info(model)
####################################################################################################################
loss_per_epoch = []
model.train()
for it in range(1, params['num_of_epochs'] + 1):
    running_loss = 0.0

    if it % 100 == 0:
        print(it)
    for j in range(0, len(X_train), params['batch_size']):
        opt.zero_grad()

        x_batch = X_train[j:j + params['batch_size']]
        y_batch = y_train[j:j + params['batch_size']]

        predictions = model.forward(x_batch)

        loss = model.loss(predictions.log(), y_batch)
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()
    loss_per_epoch.append(running_loss / len(X_train))
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
plt.plot(loss_per_epoch)
plt.grid(True)
plt.savefig(storage_path + '/loss_history.pdf')
# plt.show()

targets = []
inputs_ = []

logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)

    logger.info('Loss at testing: {0}'.format(loss.item()))

    f_dist = pd.DataFrame(y_test.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    # plt.show()

    f_dist = pd.DataFrame(predictions.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    # plt.show()
"""
