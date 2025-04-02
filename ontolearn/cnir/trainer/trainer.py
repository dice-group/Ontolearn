import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import trange

from cnir.base import BaseTrainer
from cnir.dataset import Dataset, CompositeDataset, InferenceDataset
from cnir.utils import score_all_inds, f1_score, jaccard_index, score_all_inds_composite


class Trainer(BaseTrainer):
    def __init__(self, model, tokenizer, data, embeddings, all_individuals, concept_to_instance_set,
                 num_examples, th, optimizer, pretrained_model_path=None, output_dir=None,
                 epochs=300, batch_size=256,
                 num_workers=4, lr=3.5e-4, train_test_split=True):
        super().__init__(model, data, embeddings, all_individuals, concept_to_instance_set,
                         num_examples, th, optimizer, output_dir, epochs, batch_size, num_workers,
                         lr)
        self.tokenizer = tokenizer
        self.train_test_split = train_test_split
        self.all_individuals_set = set(all_individuals)
        self.all_individuals_arr = np.array(sorted(all_individuals), dtype=object)
        self.pretrained_model_path = pretrained_model_path
        self.all_ind_embs = torch.FloatTensor(
            self.embeddings.loc[self.all_individuals_arr].values).to(self.device)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=40, eta_min=1e-4)
        try:
            self.load_data()
        except Exception as e:
            print("Error loading data: ", e)

    def collate_fn(self, batch):
        exprs, ind_embs, labels = zip(*batch)
        batch_dict = {
            'exprs': exprs,
            'ind_embs': ind_embs,
            'labels': labels
        }
        return batch_dict

    def collate_fn_composite(self, batch):
        exprs, ind_embs, labels, component_embeddings_dicts = zip(*batch)
        return {
            'exprs': exprs,
            'ind_embs': ind_embs,
            'labels': labels,
            'component_embeddings_dict': component_embeddings_dicts
        }

    def collate_fn_inference(self, batch):
        exprs, component_embeddings_dicts = zip(*batch)

        return {'exprs': exprs, 'component_embeddings_dict': component_embeddings_dicts}

    def load_data(self):
        if self.train_test_split:
            if isinstance(self.data, list):
                data = np.array(self.data, dtype=object)
            else:
                data = self.data
            mean_length = np.mean(list(map(lambda x: len(x[0].split()), data)))
            stratify_by = np.array([int(len(x[0].split()) >= mean_length) for x in data])
            train_data, val_data = train_test_split(data, test_size=0.1, stratify=stratify_by)
            if self.tokenizer is not None:
                self.train_dataloader = DataLoader(
                    Dataset(train_data, self.all_individuals_set, self.concept_to_instance_set,
                            self.embeddings, False, self.num_examples),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn,
                    shuffle=True)
                self.val_dataloader = DataLoader(
                    Dataset(val_data, self.all_individuals_set, self.concept_to_instance_set,
                            self.embeddings, False, self.num_examples),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn,
                    shuffle=False)
            else:
                self.train_dataloader = DataLoader(
                    CompositeDataset(train_data, self.all_individuals_set,
                                     self.concept_to_instance_set,
                                     self.embeddings, self.num_examples),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn_composite,
                    shuffle=True)
                self.val_dataloader = DataLoader(
                    CompositeDataset(val_data, self.all_individuals_set,
                                     self.concept_to_instance_set,
                                     self.embeddings, self.num_examples),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn_composite,
                    shuffle=False)

        else:
            if self.tokenizer is not None:
                self.train_dataloader = DataLoader(
                    Dataset(self.data, self.all_individuals_set, self.concept_to_instance_set,
                            self.embeddings, self.num_examples),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn,
                    shuffle=True)
            else:
                self.train_dataloader = DataLoader(
                    CompositeDataset(self.data, self.all_individuals_set,
                                     self.concept_to_instance_set,
                                     self.embeddings, self.num_examples),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn_composite,
                    shuffle=True)

            self.val_dataloader = None

    def train_step(self, th=None):
        if th is None:
            th = self.th
        self.model.train()
        total_loss = 0
        f1 = 0
        jaccard = 0
        f1_local, jaccard_local = 0, 0
        batch_count = 0
        batch_data = trange(len(self.train_dataloader),
                            desc=f'Train: <Batch: {batch_count}/{len(self.train_dataloader)}, Loss: {np.nan}, F1: {f1_local}, Jaccard: {jaccard_local}>',
                            leave=True)

        for _, batch in zip(batch_data, self.train_dataloader):
            exprs = [e for e in batch['exprs']]
            no_of_expr = len(exprs)
            if self.tokenizer is not None:
                inputs = self.tokenizer(exprs, padding="max_length", truncation=True,
                                        max_length=self.model.max_length, return_tensors='pt')
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                ind_embs = torch.cat(batch['ind_embs'], dim=0).to(self.device)
                labels = torch.cat(batch['labels'], dim=0).to(self.device)
            else:
                component_embeddings_dict = batch['component_embeddings_dict']
                component_embeddings_dict = [
                    {key: val.to(self.device) for key, val in d.items()}
                    for d in component_embeddings_dict
                ]
                ind_embs = [e.to(self.device) for e in batch['ind_embs']]
                labels = [l.to(self.device) for l in batch['labels']]

            self.optimizer.zero_grad()
            if self.tokenizer is not None:
                _, loss = self.model(input_ids, attention_mask, label_features=ind_embs,
                                     labels=labels)
            else:
                _, loss = self.model(exprs, ind_embs, component_embeddings_dict, label=labels)
            loss.backward()
            self.optimizer.step()

            actual = [self.concept_to_instance_set[e] for e in exprs]
            if self.tokenizer is not None:
                outputs = score_all_inds(self.model, self.tokenizer, self.all_ind_embs, exprs,
                                         hidden_size=self.all_ind_embs.shape[1], chunk_size=1024)
            else:
                outputs = score_all_inds_composite(self.model, exprs, self.all_ind_embs,
                                                   component_embeddings_dict)
            retrieved = [
                set(self.all_individuals_arr[np.where(outputs[i] > th)[0]]) for i in
                range(no_of_expr)
            ]
            f1_local = sum(list(map(f1_score, actual, retrieved))) / no_of_expr
            jaccard_local = sum(list(map(jaccard_index, actual, retrieved))) / no_of_expr
            batch_count += 1
            batch_data.set_description(
                'Train: <Batch: {}/{}, Loss: {:.4f}, F1: {:.2f}, Jaccard: {:.2f}>'.format(
                    batch_count,
                    len(self.train_dataloader),
                    loss.item(),
                    f1_local,
                    jaccard_local))
            batch_data.refresh()
            total_loss += loss.item()
            f1 += f1_local
            jaccard += jaccard_local
        self.scheduler.step()
        last_lr = self.scheduler.get_last_lr()[0]
        loss = total_loss / len(self.train_dataloader)
        f1 = f1 / len(self.train_dataloader)
        jaccard = jaccard / len(self.train_dataloader)

        return loss, f1, jaccard, last_lr

    def val_step(self, th=None):
        if th is None:
            th = self.th
        self.model.eval()
        total_loss = 0
        f1 = 0
        jaccard = 0
        f1_local, jaccard_local = 0, 0
        batch_count = 0
        batch_data = trange(len(self.val_dataloader),
                            desc=f'Validation: <Batch: {batch_count}/{len(self.val_dataloader)}, Loss: {np.nan}, F1: {f1_local}, Jaccard: {jaccard_local}>',
                            leave=True)

        with torch.no_grad():
            for _, batch in zip(batch_data, self.val_dataloader):
                exprs = [e for e in batch['exprs']]
                no_of_expr = len(exprs)
                if self.tokenizer is not None:
                    inputs = self.tokenizer(exprs, padding="max_length", truncation=True,
                                            max_length=self.model.max_length, return_tensors='pt')
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    ind_embs = torch.cat(batch['ind_embs'], dim=0).to(self.device)
                    labels = torch.cat(batch['labels'], dim=0).to(self.device)
                else:
                    component_embeddings_dict = batch['component_embeddings_dict']
                    component_embeddings_dict = [
                        {key: val.to(self.device) for key, val in d.items()}
                        for d in component_embeddings_dict
                    ]
                    ind_embs = [e.to(self.device) for e in batch['ind_embs']]
                    labels = [l.to(self.device) for l in batch['labels']]
                if self.tokenizer is not None:
                    _, loss = self.model(input_ids, attention_mask, label_features=ind_embs,
                                         labels=labels)
                else:
                    _, loss = self.model(exprs, ind_embs, component_embeddings_dict, label=labels)

                actual = [self.concept_to_instance_set[expr] for expr in exprs]
                if self.tokenizer is not None:
                    outputs = score_all_inds(self.model, self.tokenizer, self.all_ind_embs, exprs,
                                             hidden_size=self.all_ind_embs.shape[1],
                                             chunk_size=1024)
                else:
                    outputs = score_all_inds_composite(self.model, exprs, self.all_ind_embs,
                                                       component_embeddings_dict)
                retrieved = [
                    set(self.all_individuals_arr[np.where(outputs[i] > th)[0]]) for i in
                    range(no_of_expr)
                ]
                f1_local = sum(list(map(f1_score, actual, retrieved))) / no_of_expr
                jaccard_local = sum(list(map(jaccard_index, actual, retrieved))) / no_of_expr
                batch_count += 1
                batch_data.set_description(
                    'Validation: <Batch: {}/{}, Loss: {:.4f}, F1: {:.2f}, Jaccard: {:.2f}>'.format(
                        batch_count,
                        len(self.val_dataloader),
                        loss.item(),
                        f1_local,
                        jaccard_local))
                batch_data.refresh()
                total_loss += loss.item()
                f1 += f1_local
                jaccard += jaccard_local

        loss = total_loss / len(self.val_dataloader)
        f1 = f1 / len(self.val_dataloader)
        jaccard = jaccard / len(self.val_dataloader)

        return loss, f1, jaccard

    def train(self, th=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        time_spent = 0
        losses = []
        f1_scores = []
        jaccard_scores = []

        losses_val = []
        f1_scores_val = []
        jaccard_scores_val = []

        best_score = 0
        for e in range(self.epochs):
            start_time = time.time()
            loss, f1, jaccard, last_lr = self.train_step(th)
            # Append metrics to lists
            losses.append(loss)
            f1_scores.append(f1)
            jaccard_scores.append(jaccard)

            if self.train_test_split:
                loss_val, f1_val, jaccard_val = self.val_step(th)
                losses_val.append(loss_val)
                f1_scores_val.append(f1_val)
                jaccard_scores_val.append(jaccard_val)

            # Keep best model weights    
            if self.train_test_split:
                if f1_val > best_score:
                    best_score = f1_val
                    self.model.save_pretrained(self.output_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(self.output_dir)
            else:
                if f1 > best_score:
                    best_score = f1
                    self.model.save_pretrained(self.output_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(self.output_dir)

            # Print metrics and time taken for each epoch
            print("#" * 100)
            print(
                f"\n===>Train: <Epoch {e + 1}/{self.epochs} - Loss: {loss}, F1: {f1}, Jaccard: {jaccard} - Lr: {last_lr}>\n")
            if self.train_test_split:
                print(
                    f"===>Validation: <Epoch {e + 1}/{self.epochs} - Loss: {loss_val}, F1: {f1_val}, Jaccard: {jaccard_val}>\n")
            end_time = time.time()
            print("Time taken: ", end_time - start_time)
            print("#" * 100, "\n")
        results = {"train": {"loss": losses, "f1": f1_scores, "jaccard": jaccard_scores}}
        if self.train_test_split:
            results.update(
                {"val": {"loss": losses_val, "f1": f1_scores_val, "jaccard": jaccard_scores_val}})
        if self.pretrained_model_path is not None and os.path.exists(
                self.output_dir + "/results.json"):
            with open(self.output_dir + "/results.json") as f:
                prev_results = json.load(f)
                for key1 in results:
                    for key2 in results[key1]:
                        results[key1][key2] = prev_results[key1][key2] + results[key1][key2]
        print("\nBest F1 score: ", best_score)
        with open(self.output_dir + "/results.json", "w") as f:
            json.dump(results, f, ensure_ascii=False)

    def predict(self, expr, th=None):
        if th is None:
            th = self.th
        self.model.eval()
        if isinstance(expr, str):
            expr = [expr]
        else:
            assert isinstance(expr, list) or isinstance(expr, tuple) or isinstance(expr,
                                                                                   np.ndarray), f"You must provide a class expression or iterable of class expressions. Got {type(expr)}"
        test_dataloader = DataLoader(
            InferenceDataset(data=expr, all_individuals=self.all_individuals_set,
                             concept_to_instance_set=self.concept_to_instance_set,
                             embeddings=self.embeddings),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn_inference,
            shuffle=False)
        batch_count = 0
        batch_data = trange(len(test_dataloader),
                            desc=f'Inference: <Batch: {batch_count}/{len(test_dataloader)}>',
                            leave=False)
        predictions = []
        f1 = 0
        jaccard = 0
        results = {}
        with torch.no_grad():
            for _, batch in zip(batch_data, test_dataloader):
                exprs = [e for e in batch['exprs']]
                component_embeddings_dict = batch['component_embeddings_dict']
                component_embeddings_dict = [
                    {key: val.to(self.device) for key, val in d.items()}
                    for d in component_embeddings_dict
                ]
                no_of_expr = len(exprs)
                if self.tokenizer is not None:
                    outputs = score_all_inds(self.model, self.tokenizer, self.all_ind_embs, exprs,
                                             hidden_size=self.all_ind_embs.shape[1], chunk_size=1024)
                else:
                    outputs = score_all_inds_composite(self.model, exprs, self.all_ind_embs,
                                                       component_embeddings_dict)
                actual = [self.concept_to_instance_set[e] for e in exprs]
                retrieved = [
                    set(self.all_individuals_arr[np.where(outputs[i] > th)[0]]) for i in
                    range(no_of_expr)
                ]
                f1_per_expr = list(map(f1_score, actual, retrieved))
                f1_local = sum(f1_per_expr) / no_of_expr
                jaccard_per_expr = list(map(jaccard_index, actual, retrieved))
                jaccard_local = sum(jaccard_per_expr) / no_of_expr
                for i in range(len(exprs)):
                    results[exprs[i]] = {"f1":  f1_per_expr[i], "jaccard": jaccard_per_expr[i]}
                f1 += f1_local
                jaccard += jaccard_local
                predictions.extend(retrieved)
                batch_count += 1
                batch_data.refresh()
        f1 = f1 / len(test_dataloader)
        jaccard = jaccard / len(test_dataloader)
        results["f1"] = f1
        results["jaccard"] = jaccard
        with open(self.output_dir + "/results.json", "w") as f:
            json.dump(results, f)


        return f1, jaccard


class PMATrainer:
    def __init__(self, model, optimizer, data, valid_inds, instances, emb, output_dir, th,
                 batch_size=8,
                 epochs=25, num_examples=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.data = data
        self.valid_inds = valid_inds
        self.instances = instances
        self.emb = emb
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_examples = num_examples
        self.valid_inds_arr = np.array(sorted(valid_inds), dtype=object)
        self.valid_inds_embs = torch.FloatTensor(
            self.emb.loc[self.valid_inds_arr].values).to(self.device)
        self.th = th
        self.load_data()

        if output_dir is None or not os.path.exists(output_dir):
            if output_dir is None:
                output_dir = "PMA_out"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def collate_fn(self, batch):
        try:
            named_concept, class_emb, ind_emb, label = zip(*batch)
            return {
                'named_concept': named_concept,
                'class_emb': class_emb,
                'ind_emb': ind_emb,
                'label': label
            }
        except:
            named_concept, ind_emb, label = zip(*batch)
            return {
                'named_concept': named_concept,
                'ind_emb': ind_emb,
                'label': label
            }

    def load_data(self):
        train_dataset = Dataset(self.data, self.valid_inds, self.instances, self.emb,
                                num_examples=self.num_examples, pma_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=self.collate_fn)
        self.train_loader = train_loader

    def pad(self, sequences, padding_value=0):
        lengths = [len(seq) for seq in sequences]
        max_length = max(lengths)
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, padding_value=padding_value,
                                                           batch_first=True)

        return padded_sequences, lengths

    def train(self):
        self.model.train()
        losses = []
        f1_scores = []
        jaccards = []
        best_score = 0
        best_weights = None
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0
            total_f1 = 0
            total_jaccard = 0
            for batch in self.train_loader:
                if "class_emb" in batch:
                    named_concept, class_emb, ind_emb, label = batch['named_concept'], batch[
                        'class_emb'], batch['ind_emb'], batch['label']
                else:
                    named_concept, ind_emb, label = batch['named_concept'], batch['ind_emb'], batch[
                        'label']
                no_of_classes = len(named_concept)
                actual = [set(self.instances[e]) for e in named_concept]
                ind_emb, lengths = self.pad(ind_emb)
                if "class_emb" in batch:
                    class_emb, _ = self.pad(class_emb)
                else:
                    class_emb = ind_emb
                label = torch.stack(label, dim=0)
                self.optimizer.zero_grad()
                _, loss = self.model(class_emb.to(self.device), ind_emb.to(self.device),
                                     label.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                valid_inds_embs_expanded = self.valid_inds_embs.repeat(no_of_classes, 1, 1)
                out = self.model(class_emb.to(self.device),
                                 valid_inds_embs_expanded).detach().cpu().numpy()
                retrieved = [set(self.valid_inds_arr[np.where(out[i] > self.th)[0]]) for i in
                             range(no_of_classes)]
                f1_local = sum(list(map(f1_score, actual, retrieved))) / no_of_classes
                jaccard_local = sum(list(map(jaccard_index, actual, retrieved))) / no_of_classes
                total_f1 += f1_local
                total_jaccard += jaccard_local
            loss = total_loss / len(self.train_loader)
            f1 = total_f1 / len(self.train_loader)
            jaccard = total_jaccard / len(self.train_loader)
            losses.append(loss)
            f1_scores.append(f1)
            jaccards.append(jaccard)
            # Print metrics and time taken for each epoch
            print("#" * 100)
            print(
                f"Train: <Epoch {epoch + 1}/{self.epochs} - Loss: {loss} F1: {f1}, Jaccard: {jaccard}>")
            end_time = time.time()
            print("Time taken: ", end_time - start_time)
            print("#" * 100, "\n")

            if f1 > best_score:
                best_score = f1
                best_weights = self.model.state_dict()

        results = {"train": {"loss": losses, "f1": f1_scores, "jaccard": jaccards}}
        print("Best F1 score: ", best_score)
        with open(self.output_dir + "/results.json", "w") as f:
            json.dump(results, f, ensure_ascii=False)
        # save model
        torch.save(best_weights, f'{self.output_dir}/model.pt')
