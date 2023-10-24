import json

import torch
from .helper_funcs import *
from .helper_classes import *
from .complex_models import *
from .real_models import *
from torch.optim.lr_scheduler import ExponentialLR
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd


# Fixing the random seeds.
# seed = 1
# np.random.seed(seed)
# torch.manual_seed(seed)


class Experiment:
    """
    Experiment class for training and evaluation
    """

    def __init__(self, *, dataset, model, parameters, ith_logger, store_emb_dataframe=False, storage_path=""):

        self.dataset = dataset
        self.model = model
        self.store_emb_dataframe = store_emb_dataframe

        self.embedding_dim = parameters['embedding_dim']
        self.num_of_epochs = parameters['num_of_epochs']
        self.learning_rate = parameters['learning_rate']
        self.batch_size = parameters['batch_size']
        self.decay_rate = parameters['decay_rate']
        self.label_smoothing = parameters['label_smoothing']
        self.optim = parameters['optim']
        self.cuda = torch.cuda.is_available()
        self.num_of_workers = parameters['num_workers']
        self.optimizer = None
        self.entity_idxs, self.relation_idxs, self.scheduler = None, None, None

        self.negative_label = 0.0
        self.positive_label = 1.0

        # Algorithm dependent hyper-parameters
        self.kwargs = parameters
        self.kwargs['model'] = self.model

        if self.kwargs['scoring_technique'] != 'KvsAll':
            self.neg_sample_ratio = int(self.kwargs['scoring_technique'])
        else:
            self.neg_sample_ratio = None

        self.storage_path = storage_path
        # self.logger = create_logger(name=self.model + ith_logger, p=self.storage_path)

        print('Cuda available:{0}'.format(self.cuda))
        if 'norm_flag' not in self.kwargs:
            self.kwargs['norm_flag'] = False

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    @staticmethod
    def get_er_vocab(data):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    @staticmethod
    def get_re_vocab(data):
        # relation and tail entity
        re_vocab = defaultdict(list)
        for triple in data:
            re_vocab[(triple[1], triple[2])].append(triple[0])
        return re_vocab

    def get_batch_1_to_N(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.ones((len(batch), len(self.dataset.entities))) * self.negative_label
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = self.positive_label
        return np.array(batch), torch.FloatTensor(targets)

    def describe(self):
        print("Info pertaining to dataset:{0}".format(self.dataset.info))
        print("Number of triples in training data:{0}".format(len(self.dataset.train_data)))
        print("Number of triples in validation data:{0}".format(len(self.dataset.valid_data)))
        print("Number of triples in testing data:{0}".format(len(self.dataset.test_data)))
        print("Number of entities:{0}".format(len(self.entity_idxs)))
        print("Number of relations:{0}".format(len(self.relation_idxs)))
        # print("HyperParameter Settings:{0}".format(self.kwargs))

    def evaluate_one_to_n(self, model, data, log_info='Evaluate one to N.'):
        """
         Evaluate model
        """
        print(log_info)
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.dataset.data))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch_1_to_N(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward_head_batch(e1_idx=e1_idx, rel_idx=r_idx)
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)

        hit_1 = sum(hits[0]) / (float(len(data)))
        hit_3 = sum(hits[2]) / (float(len(data)))
        hit_10 = sum(hits[9]) / (float(len(data)))
        mean_rank = np.mean(ranks)
        mean_reciprocal_rank = np.mean(1. / np.array(ranks))

        print(f'Hits @10: {hit_10}')
        print(f'Hits @3: {hit_3}')
        print(f'Hits @1: {hit_1}')
        print(f'Mean rank: {mean_rank}')
        print(f'Mean reciprocal rank: {mean_reciprocal_rank}')

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
                   'MR': mean_rank, 'MRR': mean_reciprocal_rank}

        return results

    def evaluate_standard(self, model, data, log_info='Evaluate one to N.'):
        print(log_info)
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.dataset.data))

        for i in range(0, len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            rel_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                rel_idx = rel_idx.cuda()
                e2_idx = e2_idx.cuda()

            all_entities = torch.arange(0, len(self.entity_idxs)).long()
            all_entities = all_entities.reshape(len(all_entities), )
            if self.cuda:
                all_entities = all_entities.cuda()
            predictions = model.forward_triples(e1_idx=e1_idx.repeat(len(self.entity_idxs), ),
                                                rel_idx=rel_idx.repeat(len(self.entity_idxs), ),
                                                e2_idx=all_entities)

            filt = er_vocab[(data_point[0], data_point[1])]
            target_value = predictions[e2_idx].item()
            predictions[filt] = -np.Inf
            predictions[e1_idx] = -np.Inf
            predictions[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        hit_1 = sum(hits[0]) / (float(len(data)))
        hit_3 = sum(hits[2]) / (float(len(data)))
        hit_10 = sum(hits[9]) / (float(len(data)))
        mean_rank = np.mean(ranks)
        mean_reciprocal_rank = np.mean(1. / np.array(ranks))

        print(f'Hits @10: {hit_10}')
        print(f'Hits @3: {hit_3}')
        print(f'Hits @1: {hit_1}')
        print(f'Mean rank: {mean_rank}')
        print(f'Mean reciprocal rank: {mean_reciprocal_rank}')

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
                   'MR': mean_rank, 'MRR': mean_reciprocal_rank}

        return results

    def eval(self, model):
        """
        trained model
        """
        if self.dataset.train_data:
            if self.kwargs['scoring_technique'] == 'KvsAll':
                results = self.evaluate_one_to_n(model, self.dataset.train_data,
                                                 'Standard Link Prediction evaluation on Train Data')
            elif self.neg_sample_ratio > 0:

                results = self.evaluate_standard(model, self.dataset.train_data,
                                                 'Standard Link Prediction evaluation on Train Data')
            else:
                raise ValueError

            with open(self.storage_path + '/results.json', 'w') as file_descriptor:
                num_param = sum([p.numel() for p in model.parameters()])
                results['Number_param'] = num_param
                results.update(self.kwargs)
                json.dump(results, file_descriptor)

    def val(self, model):
        """
        Validation
        """
        model.eval()
        if self.dataset.valid_data:
            if self.kwargs['scoring_technique'] == 'KvsAll':
                self.evaluate_one_to_n(model, self.dataset.valid_data,
                                       'KvsAll Link Prediction validation on Validation')
            elif self.neg_sample_ratio > 0:
                self.evaluate_standard(model, self.dataset.valid_data,
                                       'Standard Link Prediction validation on Validation Data')
            else:
                raise ValueError
        model.train()

    def train(self, model):
        """ Training."""
        model.init()
        if self.cuda:
            model.cuda()
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optim == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)
        else:
            print(f'Please provide valid name for optimizer. Currently => {self.optim}')
            raise ValueError
        if self.decay_rate:
            self.scheduler = ExponentialLR(self.optimizer, self.decay_rate)

        print("{0} starts training".format(model.name))
        num_param = sum([p.numel() for p in model.parameters()])
        print("'Number of free parameters: {0}".format(num_param))
        # Store the setting.
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)
        with open(self.storage_path + '/settings.json', 'w') as file_descriptor:
            json.dump(self.kwargs, file_descriptor)

        self.describe()
        if self.kwargs['scoring_technique'] == 'KvsAll':
            model = self.k_vs_all_training_schema(model)
        elif self.neg_sample_ratio > 0:
            model = self.negative_sampling_training_schema(model)
        else:
            s = self.kwargs["scoring_technique"]
            raise ValueError(f'scoring_technique is not valid ***{s}**')
        # Save the trained model.
        # torch.save(model.state_dict(), self.storage_path + '/model.pt')
        # Save embeddings of entities and relations in csv file.
        if self.store_emb_dataframe:
            entity_emb, emb_rel = model.get_embeddings()
            # pd.DataFrame(index=self.dataset.entities, data=entity_emb.numpy()).to_csv(TypeError:
            # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            print("Storing embeddings at ", self.storage_path)
            pd.DataFrame(index=self.dataset.entities, data=entity_emb.cpu().numpy()).to_csv(
                '{0}/{1}_entity_embeddings.csv'.format(self.storage_path, model.name))
            pd.DataFrame(index=self.dataset.relations, data=emb_rel.cpu().numpy()).to_csv(
                '{0}/{1}_relation_embeddings.csv'.format(self.storage_path, model.name))

    def train_and_eval(self):
        """
        Train and evaluate phases.
        """

        self.entity_idxs = {self.dataset.entities[i]: i for i in range(len(self.dataset.entities))}
        self.relation_idxs = {self.dataset.relations[i]: i for i in range(len(self.dataset.relations))}

        self.kwargs.update({'num_entities': len(self.entity_idxs),
                            'num_relations': len(self.relation_idxs)})
        self.kwargs.update(self.dataset.info)
        model = None
        if self.model == 'ConEx':
            model = ConEx(self.kwargs)
        elif self.model == 'Distmult':
            model = Distmult(self.kwargs)
        elif self.model == 'Tucker':
            model = Tucker(self.kwargs)
        elif self.model == 'Complex':
            model = Complex(self.kwargs)
        elif self.model == 'TransE':
            model = TransE(self.kwargs)
        else:
            print(self.model, ' is not valid name')
            raise ValueError

        self.train(model)
        if 'vicodi'not in self.dataset.info['dataset'] and 'carcinogenesis' not in self.dataset.info['dataset']:
            self.eval(model)
        else:
            print('\n## No evaluation on large datasets, skipping ##\n')

    def k_vs_all_training_schema(self, model):
        print('k_vs_all_training_schema starts')
        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        losses = []

        head_to_relation_batch = DataLoader(
            HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(train_data_idxs), num_e=len(self.dataset.entities)),
            batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)

        # To indicate that model is not trained if for if self.num_of_epochs=0
        loss_of_epoch, it = -1, -1

        for it in range(1, self.num_of_epochs + 1):
            loss_of_epoch = 0.0
            # given a triple (e_i,r_k,e_j), we generate two sets of corrupted triples
            # 1) (e_i,r_k,x) where x \in Entities AND (e_i,r_k,x) \not \in KG
            for head_batch in head_to_relation_batch:  # mini batches
                e1_idx, r_idx, targets = head_batch
                if self.cuda:
                    targets = targets.cuda()
                    r_idx = r_idx.cuda()
                    e1_idx = e1_idx.cuda()

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

                self.optimizer.zero_grad()
                loss = model.forward_head_and_loss(e1_idx, r_idx, targets)
                loss_of_epoch += loss.item()
                loss.backward()
                self.optimizer.step()
            if self.decay_rate:
                self.scheduler.step()
            losses.append(loss_of_epoch)
            print('Loss at {0}.th epoch:{1}'.format(it, loss_of_epoch))
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
        model.eval()
        return model

    def negative_sampling_training_schema(self, model):
        model.train()
        print('negative_sampling_training_schema starts')
        train_data_idxs = np.array(self.get_data_idxs(self.dataset.train_data))
        losses = []

        batch_loader = DataLoader(
            DatasetTriple(data=train_data_idxs),
            batch_size=self.batch_size, num_workers=self.num_of_workers,
            shuffle=True, drop_last=True)

        # To indicate that model is not trained if for if self.num_of_epochs=0
        loss_of_epoch, it = -1, -1

        printout_it = self.num_of_epochs // 10
        for it in range(1, self.num_of_epochs + 1):
            loss_of_epoch = 0.0

            for (h, r, t) in batch_loader:
                label = torch.ones((len(h),))*self.positive_label
                # Generate Negative Triples
                corr = torch.randint(0, len(self.entity_idxs), (self.batch_size * self.neg_sample_ratio, 2))

                # 2.1 Head Corrupt:
                h_head_corr = corr[:, 0]
                r_head_corr = r.repeat(self.neg_sample_ratio, )
                t_head_corr = t.repeat(self.neg_sample_ratio, )
                label_head_corr = torch.ones(len(t_head_corr), )*self.negative_label

                # 2.2. Tail Corrupt
                h_tail_corr = h.repeat(self.neg_sample_ratio, )
                r_tail_corr = r.repeat(self.neg_sample_ratio, )
                t_tail_corr = corr[:, 1]
                label_tail_corr = torch.ones(len(t_tail_corr), )*self.negative_label

                # 3. Stack True and Corrupted Triples
                h = torch.cat((h, h_head_corr, h_tail_corr), 0)
                r = torch.cat((r, r_head_corr, r_tail_corr), 0)
                t = torch.cat((t, t_head_corr, t_tail_corr), 0)
                label = torch.cat((label, label_head_corr, label_tail_corr), 0)
                if self.cuda:
                    h, r, t, label = h.cuda(), r.cuda(), t.cuda(), label.cuda()
                self.optimizer.zero_grad()
                batch_loss = model.forward_triples_and_loss(h, r, t, label)
                loss_of_epoch += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()
            if it % printout_it == 0:
                self.val(model)

            print('Loss at {0}.th epoch:{1}'.format(it, loss_of_epoch))
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
        model.eval()
        return model
