import torch
from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data
from learners.base import ConceptLearner
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from core.util import create_experiment_folder, create_logger
import random
import random

storage_path, _ = create_experiment_folder(folder_name='../Log')

logger = create_logger(name='ConceptLearning', p=storage_path)

path = '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=path)

data = Data(knowledge_base=kb)
rho = Refinement(kb)

logger.info('ConceptLearning'.format())
logger.info('Knowledgebase:{0}'.format(kb.name))

params = {
    'learning_problem': 'concept_learning',
    'num_dim': 5,
    'num_of_epochs': 10,
    'batch_size': 1,
    'num_of_concepts_refined': 1,
    'num_of_inputs_for_model': 4,  # data.num_individuals // 10,
    'root_concept': kb.thing,
    'refinement_operator': rho, }

all_concepts, kw = data.generate_concepts(**params)

params.update(kw)
logger.info('Hyperparameters:{0}'.format(params))
logger.info('Number of concepts generated:{0}'.format(len(all_concepts)))

############################ Define the learning problem ##############################################################

X = []
y = []

labels = all_concepts  # random.sample(all_concepts, params['num_of_outputs'])

params['num_of_outputs'] = len(labels)
indx = dict(zip(kb.thing.instances, list(range(len(kb.thing.instances)))))

for _ in range(1):
    for c in all_concepts:
        try:
            x_pos = random.sample(c.instances, params['num_of_inputs_for_model'] // 2)
            x_neg = random.sample(kb.thing.instances - c.instances, params['num_of_inputs_for_model'] // 2)
        except:
            continue
        f_dist = data.score_with_labels(pos=x_pos, neg=x_neg, labels=labels)
        y.append(f_dist)
        X.append([indx[i] for i in x_pos + x_neg])

X = torch.tensor(X)

y = torch.tensor(y)

assert len(X) == len(y)

y = torch.softmax(y, dim=1)  # F-scores are turned into dist.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logger.info('Number of concepts in training split:{0}'.format(len(X_train)))

model = ConceptLearner(params)
model.init()
opt = torch.optim.Adam(model.parameters())

assert len(X_train) == len(y_train)

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
plt.show()

targets = []
inputs_ = []

logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)
    print(loss.item())