import ast
import random

import numpy as np
import pandas as pd
import torch as tc

import json

from typing import Dict, Any

row_dtype = np.dtype([
    ('user_id', np.int32),
    ('item_id', np.int32),
    ('score', np.float32),
    ('tokens', np.float32, (1536,))
])

class DatasetWrapper(tc.utils.data.Dataset):
    def __init__(self, shared_dataset, p_random_negative, mode):
        self.dataset : NEIDataset = shared_dataset
        self.p_random_negative = p_random_negative
        self.mode = mode

    def __getitem__(self, idx):
        row: Dict[str, Any] = self.dataset.x[idx]

        user_id = row['user_id']
        item_id = row['item_id']
        score = row['score']

        if self.mode == 'context':
            tokens = self.dataset.mm[row['index']][3]
        else:
            tokens = self.dataset.null_token

        if score < self.dataset.threshold:
            neg_item_id, neg_tokens = item_id, tokens

            # Se lo user ha una recensione con item positivo > Prendilo per generare la tripla
            if user_id in self.dataset.positive_items:
                item_new = random.choice(self.dataset.positive_items[user_id])
                pos_item_id = item_new['item_id']

                if self.mode == 'no_context':
                    pos_tokens = self.dataset.null_token
                if self.mode == 'context':
                    pos_tokens = self.dataset.mm[item_new['index']][3]

            else:
                # Se lo user NON HA recensioni positive > Genero una tripla virtuale con item positivo popolare non negativo esplicito
                if user_id in self.dataset.negative_items:
                    tmp = self.dataset.positive_global - {item['item_id'] for item in self.dataset.negative_items[user_id]}
                else:
                    tmp = self.dataset.positive_global

                pos_item_id = random.choice(list(tmp))
                pos_tokens = self.dataset.null_token
        else:
            pos_item_id, pos_tokens = item_id, tokens

            # Se lo user ha recensioni negative e randomicamente passa il controllo > Prendi item negativo per la tripla
            if user_id in self.dataset.negative_items and random.random() < (1 - self.p_random_negative):
                # Item negativo che effettivamente lo user ha recensito (esplicito)
                item_new = random.choice(self.dataset.negative_items[user_id])
                neg_item_id = item_new['item_id']

                if self.mode == 'no_context':
                    neg_tokens = self.dataset.null_token
                if self.mode == 'context':
                    neg_tokens = self.dataset.mm[item_new['index']][3]
            else:
                # Random negative item = Every possible item that is not a positive
                if user_id in self.dataset.negative_items:
                    tmp = self.dataset.items - {item['item_id'] for item in self.dataset.negative_items[user_id]}
                else:
                    tmp = self.dataset.items

                neg_item_id = random.choice(list(tmp))
                neg_tokens = self.dataset.null_token

        return user_id, pos_item_id, neg_item_id, pos_tokens, neg_tokens

    def __len__(self):
        return len(self.dataset.x)

class NEIDataset(tc.utils.data.Dataset):
    def __init__(self, folder, threshold = 4, firstPositives = 300):
        self.folder = folder
        self.threshold = threshold
        self.firstPositives = firstPositives
        self.null_token = [0] * 1536

        # Reviews (user, item, score)
        self.x = []

        # All items
        self.items = []
        self.item2idx = {}

        # Positive and negative items per user
        self.positive_items = {}
        self.negative_items = {}

        # Global positive items (First n items with avg_score >= 4 ordered by occurrences)
        self.positive_global = {}

        # Keep Track Variables
        # self.userModel_BPR_hist = {}    # Which triples have been generated to train the model
        # self.userWeight_BPR_hist = {}   # Which triples have been generated to train the TIL module w_uij

        # Review Tokens retrieval directly from file (Lazy access)
        self.mm = np.memmap(self.folder + "/ratings.bin", dtype=row_dtype, mode='r')

        for i in range(len(self.mm)):
            data = self.mm[i]
            user_id, item_id, score = data[0], data[1], data[2]

            self.x.append({
                'user_id': user_id,
                'item_id': item_id,
                'score': score,
                'index': i
            })

            if score < threshold:
                if user_id not in self.negative_items:
                    self.negative_items[user_id] = []
                self.negative_items[user_id].append({
                'item_id': item_id,
                'score': score,
                'index': i
            })
            else:
                if user_id not in self.positive_items:
                    self.positive_items[user_id] = []
                self.positive_items[user_id].append({
                'item_id': item_id,
                'score': score,
                'index': i
            })

        self.items = set([item['item_id'] for item in self.x])
        self.users = set(list(self.positive_items.keys()) + list(self.negative_items.keys()))
        self.positive_users = list(self.positive_items.keys())
        self.negative_users = list(self.negative_items.keys())

        df = pd.read_csv(self.folder + "/item2idx.csv")

        self.item2idx = dict(zip(df['item_id'], df['idx']))
        self.positive_global = set(df[df['global_positive'] == True].sort_values(
            by='occurrences',
            ascending=False
        ).head(self.firstPositives)['idx'].tolist())
        del df

        with open(self.folder + "/meta.json", 'r') as f:
            data = json.loads(f.read())
            self.all_users = data['all_users']
            self.all_items = data['all_items']

    def __getitem__(self, idx):
        row: Dict[str, Any] = self.x[idx]

        user_id = row['user_id']
        item_id = row['item_id']
        score = row['score']
        tokens = self.mm[row['index']][3]

        return user_id, item_id, score, tokens

    def __len__(self):
        return len(self.x)

class SimpleNEIDataset(tc.utils.data.Dataset):
    def __init__(self, folder, mode):
        self.folder = folder
        self.mode = mode

        self.interacted_train = None
        self.interacted_valid = None

        # Carico gli item con cui l'utente ha interagito durante il train
        if mode == "valid" or mode == "test":
            df = pd.read_csv(self.folder + "/interacted_train.csv")
            df['interactions'] = df['interactions'].apply(ast.literal_eval)
            self.interacted_train = df.set_index('user_id')['interactions'].to_dict()

            df = pd.read_csv(self.folder + "/item2idx.csv")
            self.all_items = set(df['idx'])
            del df

            df = pd.read_csv(self.folder + "/item2idx.csv")
            self.all_items = set(df['idx'])
            del df

        # Carico gli item con cui l'utente ha interagito durante il validation
        if mode == "test":
            df = pd.read_csv(self.folder + "/interacted_valid.csv")
            df['interactions'] = df['interactions'].apply(ast.literal_eval)
            self.interacted_valid = df.set_index('user_id')['interactions'].to_dict()
            del df

        # Carico gli item positivi del validation
        if mode == 'valid':
            df = pd.read_csv(self.folder + "/interacted_valid.csv")
            df['interactions'] = df['interactions'].apply(ast.literal_eval)
            self.positive_items = df.set_index('user_id')['interactions'].to_dict()
            self.users = list(set(self.positive_items.keys()))
            del df

        # Carico gli item positivi del test
        if mode == 'test':
            df = pd.read_csv(self.folder + "/interacted_test.csv")
            df['interactions'] = df['interactions'].apply(ast.literal_eval)
            self.positive_items = df.set_index('user_id')['interactions'].to_dict()
            self.users = list(set(self.positive_items.keys()))
            del df