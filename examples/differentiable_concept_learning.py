import torch
from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data
from learners.base import ConceptLearner
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from core.util import create_experiment_folder, create_logger
import pandas as pd

storage_path, _ = create_experiment_folder(folder_name='../Log')

logger = create_logger(name='DCL', p=storage_path)

path = '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=path)
logger.info('ConceptLearning'.format())
logger.info('Knowledgebase:{0}'.format(kb.name))

data = Data(knowledge_base=kb, logger=logger)
rho = Refinement(kb)

params = {
    'learning_problem': 'concept_learning',
    'num_dim': 10,
    'num_of_epochs': 30,
    'batch_size': 32,
    'num_of_concepts_refined': 1,
    'num_of_inputs_for_model': data.num_individuals // 20,
    'root_concept': kb.thing,
    'num_of_times_sample_per_concept':2,
    'refinement_operator': rho}

X, y, kw = data.generate_training_data(**params)

X = torch.tensor(X)
y = torch.tensor(y)
y = torch.softmax(y, dim=1)  # F-scores are turned into f-score distributions.

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
plt.show()

targets = []
inputs_ = []

logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)

    logger.info('Loss at testing: {0}'.format(loss.item()))

    f_dist = pd.DataFrame(y_test.numpy(),columns=[ x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    plt.show()

    f_dist = pd.DataFrame(predictions.numpy(),columns=[ x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    plt.show()




