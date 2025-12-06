import json
import csv
import os
from collections import Counter
import numpy as np
import base64
import pandas as pd

row_dtype = np.dtype([
    ('user_id', np.int32),
    ('item_id', np.int32),
    ('score', np.float32),
    ('tokens', np.float32, (1536,))
])

def createIndexes(folder_path, out_name):
    _ = os.path.basename(folder_path)
    subdir, _ = os.path.splitext(_)
    subdir = "./NotEnoughItems/" + out_name
    filepath = folder_path + 'train.json'

    os.makedirs(subdir, exist_ok=True)

    users_id = []
    items_id = []
    ratings = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for js in f:
            js = json.loads(js)
            try:
                users_id.append(str(js['reviewerID']))
                items_id.append(str(js['asin']))
                ratings.append(float(js['overall']))
            except:
                continue

    unique_users = list(set(users_id))
    all_users = len(unique_users)

    users = [(idx, user) for idx, user in enumerate(unique_users)]
    with open(subdir + '/user2idx.csv', 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['idx', 'user_id'])
        writer.writerows(users)
    # idx2user = {idx: user_id for idx, user_id in users}
    user2idx = {user_id: idx for idx, user_id in users}
    del users, unique_users

    unique_items = list(set(items_id))
    all_items = len(unique_items)

    item_counts = Counter(items_id)
    # Calcola la media degli score per ogni item
    item_scores = {}
    for item, rating in zip(items_id, ratings):
        item_scores.setdefault(item, []).append(rating)
    avg_scores = {item: np.mean(scores) for item, scores in item_scores.items()}

    items = [(idx, item, item_counts[item], avg_scores[item], (avg_scores[item] >= 4)) for idx, item in enumerate(unique_items)]
    items = sorted(items, key=lambda x: (not x[4], -x[2]))

    # Scrivi su CSV includendo la nuova colonna
    with open(subdir + '/item2idx.csv', 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['idx', 'item_id', 'occurrences', 'avg_score', 'global_positive'])
        writer.writerows(items)

    # idx2item = {idx: item_id for idx, item_id, _ in items}
    item2idx = {item_id: idx for idx, item_id, _, _, _ in items}

    pos_global = sum(1 for item in items if item[4] == True)

    del items, unique_items, item_counts

    user_hist = {}
    item_hist = {}

    with open(filepath, "r") as f_in:

        n = count_lines(filepath)
        mm = np.memmap(subdir + "/ratings.bin", dtype=row_dtype, mode='w+', shape=(n,))

        for i, js in enumerate(f_in):

            js = json.loads(js)
            mm[i]['user_id'] = int(user2idx[js['reviewerID']])
            mm[i]['item_id'] = int(item2idx[js['asin']])
            mm[i]['score'] = float(js['overall'])

            tokens = np.frombuffer(base64.b64decode(js['tokens']), dtype=np.float32).tolist()
            arr = np.asarray(tokens, dtype=np.float32)
            mm[i]['tokens'] = arr

            user, item = js['reviewerID'], js['asin']

            if user2idx[user] not in user_hist:
                user_hist[user2idx[user]] = set()

            if item2idx[item] not in item_hist:
                item_hist[item2idx[item]] = set()

            user_hist[user2idx[user]].add(item2idx[item])
            item_hist[item2idx[item]].add(user2idx[user])
        mm.flush()


    user_hist = {u: list(items) for u, items in user_hist.items()}
    item_hist = {i: list(users) for i, users in item_hist.items()}

    with open(subdir + "/user_hist.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'items'])
        for user, items in user_hist.items():
            writer.writerow([user, ','.join(map(str, items))])  # separo gli item con una virgola

    # item_hist.csv
    with open(subdir + "/item_hist.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['item', 'users'])
        for item, users in item_hist.items():
            writer.writerow([item, ','.join(map(str, users))])

    with open(subdir + "/meta.json", 'w') as f:
        json.dump({
            'all_users': all_users,
            'all_items': all_items,
            'avg_global': float(np.mean(ratings)),
            'pos_global': pos_global,
            'neg_global': all_items - pos_global,
        }, f)

    with open(subdir + "/struct.json",'w') as m:
        json.dump({'n':n, 'dtype_descr': str(row_dtype)}, m)

def handle_splits(folder_path, out_name, split):
    _ = os.path.basename(folder_path)
    subdir, _ = os.path.splitext(_)
    subdir = "./NotEnoughItems/" + out_name

    tmp = pd.read_csv(subdir + '/user2idx.csv')
    user2idx = dict(zip(tmp['user_id'], tmp['idx']))
    del tmp

    tmp = pd.read_csv(subdir + '/item2idx.csv')
    item2idx = dict(zip(tmp['item_id'], tmp['idx']))
    del tmp

    user_interacts = {}

    with open(folder_path + split + ".json", 'r') as f:
        for js in f:
            js = json.loads(js)
            try:
                if float(js['overall']) < 4:
                    continue

                if user2idx[js['reviewerID']] not in user_interacts:
                    user_interacts[user2idx[js['reviewerID']]] = []

                user_interacts[user2idx[js['reviewerID']]].append(item2idx[js['asin']])
            except:
                continue

    data = []
    for user_id, interactions  in user_interacts.items():
        data.append({
            'user_id': user_id,
            'interactions': interactions
        })
    df = pd.DataFrame(data)
    df.to_csv(subdir + '/interacted_'+split+'.csv', index=False)


def count_lines(tmp_path):
    n=0
    with open(tmp_path,'rb') as f:
        for _ in f:
            n+=1
    return n




def example():
    common_path = "{DATASETS FOLDER}"

    paths = [
        common_path + "reviews_Baby_5_seed242/",
        common_path + "reviews_Clothing_Shoes_and_Jewelry_5_seed242/",
        common_path + "reviews_Toys_and_Games_5_seed242/"
    ]

    names = [
        "Baby_5",
        "Clothing_Shoes_and_Jewelry_5",
        "Toys_and_Games_5"
    ]

    for path, name in zip(paths, names):
        createIndexes(path, name)

        for split in ['train', 'valid', 'test']:
            handle_splits(path, name, split)