import torch
from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data
from learners.base import DCL, LengthClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from core.util import create_experiment_folder,create_logger

storage_path, _ = create_experiment_folder(folder_name='../Log')

logger = create_logger(name='LengthPrediction', p=storage_path)

path = '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=path)

data = Data(knowledge_base=kb)
rho = Refinement(kb)

logger.info('Length Prediction starts'.format())
logger.info('Knowledgebase:{0}'.format(kb.name))

params = {
    'learning_problem': 'length_prediction',
    'num_dim': 50,
    'num_of_epochs': 1000,
    'batch_size': 256,
    'num_of_concepts_refined': 200,
    'num_of_inputs_for_model': data.num_individuals // 10,
    'root_concept': kb.thing,
    'refinement_operator': rho,
}

X, y, kw = data.generate(**params)

params.update(kw)

logger.info('Hyperparameters:{0}'.format(params))

logger.info('Number of concepts generated:{0}'.format(len(X)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logger.info('Number of concepts in training split:{0}'.format(len(X_train)))

y_train = torch.tensor(y_train, dtype=torch.float32)

model = LengthClassifier(params)
model.init()
opt = torch.optim.Adam(model.parameters())

assert len(X_train) == len(y_train)


plt.hist([len(i) for i in X_train], 20, density=False,label='Training')
plt.hist([len(i) for i in X_test], 20, density=False,label='Testing')
plt.xlabel('Lengths')
plt.ylabel('Number of concepts')
plt.title('Histogram of concept lengths')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(storage_path+'/historgram_of_lengths.pdf')
# TODO: We need sampling.


emp=np.array([len(i.instances) for i in X_train])
plt.hist([len(i.instances) for i in X_train], 50, density=False,label='Training')
plt.xlabel('Number of instances belonging to a class')
plt.ylabel('Number of concepts')
plt.title('Histogram of concept lengths')
plt.grid(True)
plt.show()
plt.savefig(storage_path+'/historgram_of_instances.pdf')


logger.info('Average length of concepts in training: {0}'.format(sum([len(i) for i in X_train]) / len(X_train)))
logger.info(model)
####################################################################################################################
loss_per_epoch = []
model.train()
for it in range(1, params['num_of_epochs'] + 1):
    running_loss = 0.0

    if it%100==0:
        print(it)
    for j in range(0, len(X_train), params['batch_size']):
        opt.zero_grad()

        concepts, x_batch, y_batch = data.get_mini_batch(X_train, y_train, j, params)
        predictions = model.forward(x_batch)
        loss = model.loss(predictions, y_batch)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    loss_per_epoch.append(running_loss / len(X_train))
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
plt.plot(loss_per_epoch)
plt.show()
plt.savefig(storage_path+'/loss_history.pdf')

targets = []
inputs_ = []
y_test = torch.tensor(y_test, dtype=torch.float32)

logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
with torch.no_grad():  # Important:
    for j in range(0, len(X_test), params['batch_size']):
        concepts, x_batch, y_batch = data.get_mini_batch(X_test, y_test, j, params)
        predictions = model.forward(x_batch)

        loss = model.loss(predictions, y_batch)

        for c, t, p in zip(concepts, y_batch, predictions):
            true_length = torch.argmax(t) + 1
            predicted_length = torch.argmax(p) + 1

            targets.append(true_length.numpy())
            inputs_.append(predicted_length.numpy())

            logger.info('{0}\t{1}\t{2}'.format(c.str, true_length, predicted_length))

targets = np.array(targets)
inputs_ = np.array(inputs_)

logger.info('Mean squared error on test data:{0}'.format(np.mean((targets - inputs_) ** 2)))
