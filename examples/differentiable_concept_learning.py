"""
differentiable_concept_learning.py illustrates one of our research idea.
"""
import random
import torch
from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data
from learners.base import ConceptLearner
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from core.util import create_experiment_folder, create_logger
import pandas as pd
import numpy as np

storage_path, _ = create_experiment_folder(folder_name='../Log')

logger = create_logger(name='DCL', p=storage_path)

path = '../data/family-benchmark_rich_background.owl'
# path = '../data/biopax.owl'
kb = KnowledgeBase(path=path)
logger.info('ConceptLearning'.format())
logger.info('Knowledgebase:{0}'.format(kb.name))

data = Data(knowledge_base=kb, logger=logger)
rho = Refinement(kb)

params = {
    'learning_problem': 'concept_learning',
    'num_instances': len(kb.thing.instances),
    'num_dim': 50,
    'num_of_epochs': 50,
    'batch_size': 32,
    'num_of_concepts_refined': 1,
    'num_of_inputs_for_model': data.num_individuals // 20,
    'root_concept': kb.thing,
    'num_of_times_sample_per_concept': 2,
    'refinement_operator': rho}

# Generate concepts and prune those ones that do not satisfy the provided constraint.
concepts = [concept for concept in data.generate_concepts(**params) if
            len(kb.thing.instances) > len(concept.instances) > params['num_of_inputs_for_model']]

concepts_train, concepts_test = train_test_split(concepts, test_size=0.3, random_state=1)

# Important decision:
labels = np.array(random.sample(concepts_train, 50))
# labels = np.array(concepts_train)
params['num_of_outputs'] = len(labels)

# Generate Training Data
X, y = data.convert_data(concepts_train, labels, params)

X = torch.tensor(X)
y = torch.softmax(torch.tensor(y), dim=1)  # F-scores are turned into f-score distributions.

logger.info('Number of concepts in training split:{0}'.format(len(X)))

model = ConceptLearner(params)
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
    for j in range(0, len(X), params['batch_size']):
        opt.zero_grad()

        x_batch = X[j:j + params['batch_size']]
        y_batch = y[j:j + params['batch_size']]

        predictions = model.forward(x_batch)

        loss = model.loss(predictions.log(), y_batch)
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()
    loss_per_epoch.append(running_loss / len(X))
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
plt.plot(loss_per_epoch)
plt.grid(True)
plt.savefig(storage_path + '/loss_history.pdf')
plt.show()

logger.info('Testing starts on:{0}'.format(len(concepts_test)))
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

params['num_of_times_sample_per_concept'] = 1

scores = []
true_f_dist = []
predicted_f_dist = []

with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    for true_concept in concepts_test:

        try:
            x_pos, x_neg = data.pos_neg_sampling_from_concept(true_concept, params['num_of_inputs_for_model'])
        except ValueError:
            continue

        true_f_dist.append(data.score_with_labels(pos=x_pos, neg=x_neg, labels=labels))

        input = [data.indx[i] for i in x_pos + x_neg]
        input = torch.tensor(input).reshape(1, len(input))

        predictions = model.forward(input)

        predicted_f_dist.append(predictions.flatten().numpy().tolist())

        best_pred = labels[predictions.numpy().argmax()]
        f_1 = data.score_with_instances(pos=true_concept.instances, neg=kb.thing.instances - true_concept.instances,
                                        instances=best_pred.instances)

        scores.append(f_1)
        print('True concept:{0}\tBest prediction:{1}\tF1-score:{2}'.format(true_concept.str, best_pred.str, f_1))

scores = np.array(scores)
print(scores.mean())

f_dist = pd.DataFrame(np.array(true_f_dist), columns=[x.str for x in labels])
plt.matshow(f_dist.corr())
# plt.title('Correlation of true quality of concepts at testing')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
plt.gcf().set_size_inches(25, 25)
plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
plt.show()

f_dist = pd.DataFrame(np.array(predicted_f_dist), columns=[x.str for x in labels])
plt.matshow(f_dist.corr())
# plt.title('Correlation of predicted quality of concepts at testing')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
plt.gcf().set_size_inches(25, 25)
plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
plt.show()

exit(1)

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

model = ConceptLearner(params)
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
#### Analysis
