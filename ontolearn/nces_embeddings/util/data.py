class Data:
    def __init__(self, data_dir=None, train_plus_valid=False, reverse=True, tail_pred_constraint=False,
                 out_of_vocab_flag=False):
        """
        ****** reverse=True
        Double the size of datasets by including reciprocal/inverse relations.
        We refer Canonical Tensor Decomposition for Knowledge Base Completion for details.

        ****** tail_pred_constraint=True
        Do not include reciprocal relations into testing. Consequently, MRR is computed by only tail entity rankings.

        ****** train_plus_valid=True
        Use the union of training and validation split during training phase.

        ****** out_of_vocab_flag=True
        Remove all triples from validation and test that contain at least one entity that did not occurred during training.

        """
        self.info = {'dataset': data_dir,
                     'dataset_augmentation': reverse,
                     'train_plus_valid': train_plus_valid,
                     'tail_pred_constraint': tail_pred_constraint}

        self.train_data = self.load_data(data_dir, data_type="train", add_reciprical=reverse)
        self.valid_data = self.load_data(data_dir, data_type="valid", add_reciprical=reverse)
        if tail_pred_constraint:
            self.test_data = self.load_data(data_dir, data_type="test", add_reciprical=False)
        else:
            self.test_data = self.load_data(data_dir, data_type="test", add_reciprical=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        # The order of entities is important
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        # The order of entities is important
        self.relations = self.train_relations + [i for i in self.valid_relations \
                                                 if i not in self.train_relations] + [i for i in self.test_relations \
                                                                                      if i not in self.train_relations]
        # Sanity checking on the framework.
        assert set(self.relations) == set(self.train_relations).union(
            set(self.valid_relations).union(set(self.test_relations)))

        if train_plus_valid:
            self.train_data.extend(self.valid_data)
            self.valid_data = []

        if out_of_vocab_flag:
            print('Triples containing out-of-vocabulary entities will be removed from validation and training splits.')
            ent = set(self.get_entities(self.train_data))
            print('|G^valid|={0}\t|G^test|={1}'.format(len(self.valid_data), len(self.test_data)))
            self.valid_data = [i for i in self.valid_data if i[0] in ent and i[2] in ent]
            self.test_data = [i for i in self.test_data if i[0] in ent and i[2] in ent]
            print('After removal, |G^valid|={0}\t|G^test|={1}'.format(len(self.valid_data), len(self.test_data)))

    @staticmethod
    def load_data(data_dir, data_type, add_reciprical=True):
        try:
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data if len(i.split("\t"))==3]
                if add_reciprical:
                    data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
