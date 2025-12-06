import json
import random
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from torch.func import functional_call
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os

import math
from tqdm import tqdm
import fire
from itertools import product
from datetime import datetime; now = datetime.now
from sklearn.metrics import roc_auc_score

from TIL.models import context_BPR, CrossAttentionReview2User, TIL
from TIL.dataset import NEIDataset
from dataset import NEIDataset, DatasetWrapper, SimpleNEIDataset

seed = 242
grid_seed = 33
extra = False
torch.manual_seed(seed)

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
is_grid = False

TIL_CONFIG = {
    'folder': "NotEnoughItems",
    'dataset': "Clothing_Shoes_and_Jewelry_5",
    'seed': 33,
    'integrate': True,

    'num_epochs': 500,
    'dual_user': False,

    'emb_dim': 128,
    'num_att_layers': 3,
    'num_heads': 16,
    'ffn_mult': 4,
    'null_p': 0.3,
    'dropout': 0.2,
}


# Train Default Parameters
train_params = {
    'num_epochs': 500,
    'batch_size': 8192,
    'num_workers': 3,
    'pin_memory': True,
    'persistent_workers': True,
    'dual_user': False,
    "early_stop": 4,
    "decay_tol": 4,
    "decay_factor": 0.9,

    'emb_dim': 128,
    'num_att_layers': 3,
    'num_heads': 16,
    'ffn_mult': 4,
    'null_p': 0.3,
    'dropout': 0.2,
    'epsilon': 1e-7,

    'lr': 1e-3,
    'lr_step_decay': 3,
    'lr_lambda': 1e-5,


    'dataset': "Clothing_Shoes_and_Jewelry_5",
    'folder': "./NotEnoughItems/",
    #'folder': "/leonardo_work/IscrC_InterLLM/TIL/NotEnoughItems/",

}

def collate_fn_noc(batch):
    u, i1, i2 = zip(*batch)

    u = torch.tensor(u)
    i1 = torch.tensor(i1)
    i2 = torch.tensor(i2)

    return u, i1, i2

def collate_fn_c(batch):
    u, i1, i2, t1, t2 = zip(*batch)

    u = torch.tensor(u)
    i1 = torch.tensor(i1)
    i2 = torch.tensor(i2)

    t1 = torch.from_numpy(np.array(t1, dtype=np.float32))
    t2 = torch.from_numpy(np.array(t2, dtype=np.float32))

    return (u, i1, i2), (t1, t2)

collane_fn = {
    'noc': collate_fn_noc,
    'c': collate_fn_c,
}


def pretrain():

    batch_size = train_params['batch_size']
    num_workers = train_params['num_workers']
    persistent_workers = train_params['persistent_workers']
    pin_memory = train_params['pin_memory']
    emb_dim = train_params['emb_dim']
    num_att_layers = train_params['num_att_layers']
    num_heads = train_params['num_heads']
    ffn_mult = train_params['ffn_mult']
    null_p = train_params['null_p']
    dropout = train_params['dropout']
    lr = train_params['lr']
    num_epochs = train_params['num_epochs']
    dual_user = train_params['dual_user']
    early_stop = train_params['early_stop']
    decay_tol = train_params['decay_tol']
    decay_factor = train_params['decay_factor']

    dataset = train_params['dataset']
    folder = train_params['folder'] + dataset

    print("Start training....")
    names = {'model': "/model_params", }
    path = folder
    if is_grid:
        path += '/grid'
        if grid_seed:
            names['model'] += '_seed-' + str(grid_seed)
        if extra:
            names['model'] += '_extra-' + str(extra)
    names['model'] += '.pth'

    train_data = NEIDataset(folder=folder)

    train_loader = DataLoader(
        DatasetWrapper(shared_dataset=train_data, p_random_negative=0.65, mode='context'),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collane_fn['c'],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    val_data = SimpleNEIDataset(folder, 'valid')

    encoder_user = CrossAttentionReview2User(
        num_obj=train_data.all_users,
        emb_dim=emb_dim,
        device=device,
        num_att_layers=num_att_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        null_p=null_p,
        dropout=dropout
    )

    encoder_item = CrossAttentionReview2User(
        num_obj=train_data.all_items,
        emb_dim=emb_dim,
        device=device,
        num_att_layers=num_att_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        null_p=null_p,
        dropout=dropout
    )

    model = context_BPR(encoder_user, encoder_item, device, dropout=dropout)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_factor)

    model.train()

    current_tol, total_tol = 0, 0
    min_loss = 1e+10


    pbar = tqdm(total=num_epochs * math.ceil(len(train_data) / batch_size), desc='Training')

    for epoch in range(num_epochs):
        total_loss = 0.0

        model.train()
        for i, (x, t) in enumerate(train_loader):
            user_id, pos_item_id, neg_item_id = x
            pos_tokens, neg_tokens = t

            user_embs, pos_item_emb, neg_item_emb = model(user_id, pos_item_id, neg_item_id,
                                                          user_tokens=(pos_tokens, neg_tokens),
                                                          pos_tokens=pos_tokens, neg_tokens=neg_tokens,
                                                          p=True, dual_user=dual_user)
            loss = model.compute_loss(user_embs, user_id, pos_item_emb, pos_item_id, neg_item_emb, neg_item_id,
                                      dual_user)
            loss = loss.mean()
            total_loss += loss.item() * len(user_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

        # Report sulla loss media
        avg_loss = total_loss / len(train_data)
        print(f"\tTraining data: total loss: {total_loss:.4f}, avg loss: {avg_loss:.4f}")

        # Validazione alla fine di ogni epoch
        model.eval()
        neg_recall, _ = predict_bpr(model, data=val_data, k=10)
        model.train()

        if neg_recall < min_loss:
            total_tol = 0
            current_tol = 0
            torch.save(model.state_dict(), path + names['model'])
            min_loss = neg_recall
            print(f"Model saved with [Recall: {min_loss:.4f}] and [Loss: {total_loss:.4f}]")
        else:
            current_tol += 1
            if current_tol == decay_tol:
                print("Reducing learning due to decaying")
                scheduler.step()
                current_tol = 0
                total_tol += 1
            if total_tol == early_stop + 1:
                print("Early stop at epoch %s with best_recall=%.4f" % (epoch, neg_recall))
                break
        print("*" * 30)

    pbar.close()
    model.eval()

    model.load_state_dict(torch.load(path + names['model']))

    print("----" * 20)
    print(f"Inizio calcolo Recall [{now()}]")
    neg_recall, _ = predict_bpr(model, data=val_data, k=10)
    print(f"Fine calcolo Recall [{now()}]")
    print("----" * 20)
    print(f"{dataset} - best_res: {neg_recall}")
    print("----" * 20)

    return neg_recall


def train_weight():

    batch_size = train_params['batch_size']
    num_workers = train_params['num_workers']
    persistent_workers = train_params['persistent_workers']
    pin_memory = train_params['pin_memory']
    emb_dim = train_params['emb_dim']
    num_att_layers = train_params['num_att_layers']
    num_heads = train_params['num_heads']
    ffn_mult = train_params['ffn_mult']
    null_p = train_params['null_p']
    dropout = train_params['dropout']
    num_epochs = train_params['num_epochs']
    dual_user = train_params['dual_user']
    early_stop = train_params['early_stop']
    decay_tol = train_params['decay_tol']
    decay_factor = train_params['decay_factor']
    epsilon = train_params['epsilon']

    lr_lambda = train_params['lr_lambda']

    lr = train_params['lr']
    lr_step_decay = train_params['lr_step_decay']

    # Compensating pretrain decay tol
    lr = lr * (decay_factor ** lr_step_decay)

    print(f"Decay applied {lr}")

    dataset = train_params['dataset']
    folder = train_params['folder'] + dataset

    print("Start training weights....")
    names = {
        'model': "/TIL_model_params",
        'model_grid': "/model_params",
        'weight': "/TIL",
    }

    path = folder
    if is_grid:
        path += '/grid'
        if grid_seed:
            names['model'] += '_seed-' + str(grid_seed)
            names['model_grid'] += '_seed-' + str(grid_seed)
            names['weight'] += '_seed-' + str(grid_seed)
        if extra:
            names['model'] += '_extra-' + str(extra)
            names['model_grid'] += '_extra-seed-' + str(extra)
            names['weight'] += '_extra-' + str(extra)
    names['model_grid'] += '_best.pth'
    names['model'] += '.pth'
    names['weight'] += '.pth'

    train_data = NEIDataset(folder=folder)

    train_loader = DataLoader(
        DatasetWrapper(shared_dataset=train_data, p_random_negative=1, mode='no_context'),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collane_fn['c'],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    weight_loader = DataLoader(
        DatasetWrapper(shared_dataset=train_data, p_random_negative=1, mode='no_context'),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collane_fn['c'],
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    val_data = SimpleNEIDataset(folder, 'valid')

    encoder_user = CrossAttentionReview2User(
        num_obj=train_data.all_users,
        emb_dim=emb_dim,
        device=device,
        num_att_layers=num_att_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        null_p=null_p,
        dropout=dropout
    )

    encoder_item = CrossAttentionReview2User(
        num_obj=train_data.all_items,
        emb_dim=emb_dim,
        device=device,
        num_att_layers=num_att_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        null_p=null_p,
        dropout=dropout
    )

    # Caricamento del modello
    model = context_BPR(encoder_user, encoder_item, device, dropout=dropout)
    model.load_state_dict(torch.load(path + names['model_grid']))
    optimizer_theta = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Caricamento modulo TIL
    weights = TIL(encoder_item, device)
    optimizer_lambda = torch.optim.SGD(weights.parameters(), lr=lr_lambda, weight_decay=1e-4)

    # Scheduler separati
    scheduler_theta = torch.optim.lr_scheduler.StepLR(optimizer_theta, step_size=1, gamma=decay_factor)
    scheduler_lambda = torch.optim.lr_scheduler.StepLR(optimizer_lambda, step_size=1, gamma=decay_factor)

    model.train()

    current_tol, total_tol = 0, 0
    min_loss = float("inf")

    pbar = tqdm(total=num_epochs * math.ceil(len(train_data) / batch_size), desc='Training')

    for epoch in range(num_epochs):
        total_loss = 0.0

        # Generazione nu (aggregazione del TIL)
        with torch.no_grad():
            model.eval()
            weights.aggregate(train_data)
            model.train()

        model.train()
        for (x, t), (x_w, t_w) in zip(train_loader, weight_loader):
            user_id, pos_item_id, neg_item_id = x
            pos_tokens, neg_tokens = t

            # ============================================================
            # INNER OPTIMIZATION: Aggiorna Theta (parametri del modello)
            # ============================================================

            # Forward pass con parametri correnti
            user_embs, pos_item_emb, neg_item_emb = model(
                user_id, pos_item_id, neg_item_id,
                user_tokens=(pos_tokens, neg_tokens),
                pos_tokens=pos_tokens,
                neg_tokens=neg_tokens,
                p=True,
                dual_user=dual_user
            )

            # BPR loss
            BPR_loss = model.compute_loss(
                user_embs, user_id,
                pos_item_emb, pos_item_id,
                neg_item_emb, neg_item_id,
                dual_user
            )

            # Pesi TIL
            w_uij = weights(user_id, user_embs, pos_item_emb, neg_item_emb, dual_user=dual_user)

            # Loss inner pesata (Solo per il modello)
            L_inner = ((w_uij.detach() + epsilon) * BPR_loss).mean()

            total_loss += L_inner.item() * len(user_id)

            # Backward e aggiornamento di Theta (Il modello)
            optimizer_theta.zero_grad()
            L_inner.backward()
            optimizer_theta.step()

            # ============================================================
            # OUTER OPTIMIZATION: Aggiorna Lambda (modulo TIL)
            # ============================================================

            # Calcolo dei gradienti del passo virtuale
            user_embs, pos_item_emb, neg_item_emb = model(
                user_id, pos_item_id, neg_item_id,
                user_tokens=(pos_tokens, neg_tokens),
                pos_tokens=pos_tokens,
                neg_tokens=neg_tokens,
                p=True,
                dual_user=dual_user
            )
            BPR_loss = model.compute_loss(
                user_embs, user_id,
                pos_item_emb, pos_item_id,
                neg_item_emb, neg_item_id,
                dual_user
            )

            w_uij = weights(user_id, user_embs, pos_item_emb, neg_item_emb, dual_user=dual_user)
            L_inner_for_grad = ((w_uij + epsilon) * BPR_loss).mean()

            # Calcolo dei gradienti di Theta (Modello) rispetto a L_inner
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                L_inner_for_grad,
                params,
                create_graph=True,
                allow_unused=True
            )

            # Parametri virtuali > Theta' = Theta - alpha * grad
            named_params = dict(model.named_parameters())
            theta_virtual = {
                name: (param - lr * (g.clamp(-1, 1) if g is not None else torch.zeros_like(param)))
                for (name, param), g in zip(named_params.items(), grads)
            }

            user_id_o, pos_item_id_o, neg_item_id_o = x_w
            pos_tokens_o, neg_tokens_o = t_w

            # Forward pass virtuale con Theta' usando il loader a parte
            user_embs_v, pos_item_emb_v, neg_item_emb_v = functional_call(
                model,
                theta_virtual,
                (user_id_o, pos_item_id_o, neg_item_id_o),
                {
                    'user_tokens': (pos_tokens_o, neg_tokens_o),
                    'pos_tokens': pos_tokens_o,
                    'neg_tokens': neg_tokens_o,
                    'p': True,
                    'dual_user': dual_user
                }
            )

            # Loss outer (senza pesi, BPR standard)
            J_outer = model.compute_loss(
                user_embs_v, user_id_o,
                pos_item_emb_v, pos_item_id_o,
                neg_item_emb_v, neg_item_id_o,
                dual_user
            ).mean()

            # Backward e aggiornamento di Lambda (TIL)
            optimizer_lambda.zero_grad()
            J_outer.backward()
            optimizer_lambda.step()

            pbar.update(1)

        # Validazione alla fine di ogni epoch
        model.eval()
        neg_recall, _ = predict_bpr(model, data=val_data, k=10)
        model.train()
        print("\n" + "*" * 30)
        if neg_recall < min_loss:
            total_tol = 0
            current_tol = 0
            torch.save(model.state_dict(), path + names['model'])
            torch.save(weights.state_dict(), path + names['weight'])
            min_loss = neg_recall
            print(f"\nModel saved with [Recall: {min_loss:.4f}] and [Loss: {total_loss:.4f}]")
        else:
            current_tol += 1
            if current_tol == decay_tol:
                print("\nReducing learning on both model & TIL")
                scheduler_theta.step()
                scheduler_lambda.step()
                current_tol = 0
                total_tol += 1
            if total_tol == early_stop + 1:
                print("\nEarly stop at epoch %s with latest_recall=%.4f and best_recall=%.4f" % (epoch, neg_recall, min_loss))
                break

        # Report sulla loss media
        avg_loss = total_loss / len(train_data)
        print(f"\tTraining data: total loss: {total_loss:.4f}, avg loss: {avg_loss:.4f}")

    pbar.close()
    model.eval()

    model.load_state_dict(torch.load(path + names['model']))

    print("----" * 20)
    print(f"Inizio calcolo Recall [{now()}]")
    neg_recall, _ = predict_bpr(model, data=val_data, k=10, U=-1, N=-1)
    print(f"Fine calcolo Recall [{now()}]")
    print("----" * 20)
    print(f"{dataset} - best_res: {neg_recall}")
    print("----" * 20)

    return neg_recall


def evaluate_recommendation(model, data, k=10, N=200, U=128):

    # Se N o U sono -1, li trattiamo come "tutti"
    if N == -1:
        N = len(data.all_items)
    if U == -1:
        U = len(data.users)

    recalls = []
    precisions = []
    hits = []
    aucs = []

    model.eval()

    with torch.no_grad():
        users = data.users
        # Seleziona un sottoinsieme casuale di utenti se U < totale utenti
        if len(users) > U:
            users = np.random.choice(users, size=U, replace=False)

        interacted_train = getattr(data, "interacted_train", None)
        interacted_valid = getattr(data, "interacted_valid", None)

        for user_id in users:
            # 1. Identifica i positivi (ground truth per la validazione)
            positives = set(data.positive_items[user_id])
            if not positives:
                continue

            # 2. Identifica gli item da escludere (già visti in train/valid ma non target corrente)
            interacted = []
            if interacted_train is not None and user_id in interacted_train:
                interacted.extend(interacted_train[user_id])
            if interacted_valid is not None and user_id in interacted_valid:
                interacted.extend(interacted_valid[user_id])

            interacted_set = set(interacted)

            # 3. Generazione dei candidati negativi
            # Filtriamo: non devono essere negli interagiti e non devono essere i positivi target
            candidate_negatives = [item for item in data.all_items
                                   if item not in interacted_set and item not in positives]

            # Campionamento negativi
            if len(candidate_negatives) > N:
                candidate_negatives = random.sample(candidate_negatives, N)

            # La lista finale dei candidati è Negativi + Positives
            candidates = candidate_negatives + list(positives)

            if not candidates:
                continue

            # 4. Preparazione Tensor
            user_t = torch.tensor([user_id], dtype=torch.long).to(model.device)
            item_t = torch.tensor(candidates, dtype=torch.long).to(model.device)

            # 5. Inference (Embedding + Scoring)
            # In context_BPR generiamo prima gli embedding, poi calcoliamo lo score
            user_emb = model.embedding("user", user_t)  # [1, emb_dim]
            item_embs = model.embedding("item", item_t)  # [n_candidates, emb_dim]

            # Espandiamo l'embedding utente per matchare il numero di candidati
            user_emb_exp = user_emb.expand(len(candidates), -1)
            user_t_exp = user_t.expand(len(candidates))

            # Calcolo score
            scores = model.score(user_emb_exp, item_embs, user_t_exp, item_t)
            Y_pred = scores.cpu().numpy()  # Score predetti

            # 6. Creazione Ground Truth (Y_real)
            # 1 se l'item è nei positivi, 0 altrimenti
            Y_real = np.array([1 if item in positives else 0 for item in candidates])

            # 7. Calcolo Metriche
            # Ordiniamo per score decrescente
            idx = np.argsort(-Y_pred)
            top_k_idx = idx[:k]

            relevant_k = Y_real[top_k_idx].sum()
            total_relevant = len(positives)

            # Recall@K
            recalls.append(relevant_k / total_relevant if total_relevant > 0 else 0)

            # Precision@K
            precisions.append(relevant_k / k)

            # Hit@K
            hits.append(1.0 if relevant_k > 0 else 0.0)

            # AUC
            # try/except necessario perché roc_auc_score crasha se c'è una sola classe in Y_real
            try:
                auc_val = roc_auc_score(Y_real, Y_pred)
                aucs.append(auc_val)
            except ValueError:
                pass

    model.train()

    avg_recall = np.nanmean(recalls) if recalls else 0.0
    avg_precision = np.nanmean(precisions) if precisions else 0.0
    avg_hit = np.nanmean(hits) if hits else 0.0
    avg_auc = np.nanmean(aucs) if aucs else 0.0

    return avg_recall, avg_precision, avg_hit, avg_auc

def predict_bpr(model : context_BPR, data : SimpleNEIDataset, k=10, U=2048, N=1024):
    if N == -1:
        N = len(data.all_items)
    if U == -1:
        U = len(data.users)

    recall_k = 0.0

    with torch.no_grad():
        users = data.users
        if len(users) > U:
            users = np.random.choice(users, size=U, replace=False)

        interacted_train = getattr(data, "interacted_train", None)
        interacted_valid = getattr(data, "interacted_valid", None)

        for user_id in users:
            interacted = []
            positives = data.positive_items[user_id]
            tmp_user_id = user_id

            if interacted_train is not None and user_id in interacted_train:
                interacted.extend(interacted_train[user_id])

            if interacted_valid is not None and user_id in interacted_valid:
                interacted.extend(interacted_valid[user_id])

            all_items = [item for item in data.all_items if item not in interacted and item not in positives]
            if len(all_items) > N:
                all_items = random.sample(all_items, N)
            all_items += positives

            if not all_items:  # Verifica che ci siano item validi
                continue

            # Faccio i tensori con gli ID
            user_id = torch.tensor([user_id], dtype=torch.long).to(model.device)
            item_ids = torch.tensor(all_items, dtype=torch.long).to(model.device)

            # Genero gli embedding
            user_emb = model.embedding("user", user_id)
            item_embs = model.embedding("item", item_ids)

            # Devo espandere lo user per come model.score() è fatto
            user_emb = user_emb.expand(len(all_items), -1)
            user_id = user_id.expand(len(all_items))

            scores = model.score(user_emb, item_embs, user_id, item_ids)
            tmp = torch.argsort(scores, descending=True)
            item_ids = item_ids[tmp]
            del tmp, scores

            top_k = item_ids[:k]
            total_relevant = len(data.positive_items[tmp_user_id])
            count = 0
            for item in top_k.tolist():
                if item in data.positive_items[tmp_user_id]:
                    count += 1
            recall_k += (count / total_relevant)

        avg_recall = recall_k / len(users)
        return -recall_k, -avg_recall

def test():

    path = TIL_CONFIG["folder"]

    seed, extra = None, None

    if "seed" in TIL_CONFIG:
        seed = TIL_CONFIG["seed"]
    if "extra" in TIL_CONFIG:
        extra = TIL_CONFIG["extra"]

    if TIL_CONFIG['integrate']:
        names = {
            'model': "/TIL_model_params",
            'grid': "/grid"
        }
    else:
        names = {
            'model': "/model_params",
            'grid': "/grid"
        }

    path += "/" + TIL_CONFIG['dataset'] + '/grid'
    if seed:
        names['model'] += '_seed-' + str(seed)
        names['grid'] += '_' + str(seed)
    if extra:
        names['model'] += '_extra-' + str(extra)
        names['grid'] += '_extra-' + str(extra)

    names['model'] += '_best.pth'
    names['grid'] += '.csv'

    TIL_CONFIG["model_pth"] = path + names['model']
    TIL_CONFIG['grid'] = path + names['grid']
    TIL_CONFIG['folder'] += "/" + TIL_CONFIG['dataset']
    TIL_CONFIG['path'] = path

    with open(TIL_CONFIG['folder'] + '/meta.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.read_csv(TIL_CONFIG['grid'])

    best_param = df.loc[df['loss'].idxmin(), 'param']
    best_param = json.loads(best_param)
    del df

    df_items = pd.read_csv(TIL_CONFIG['folder'] + '/item2idx.csv')
    df_users = pd.read_csv(TIL_CONFIG['folder'] + '/user2idx.csv')

    item2idx = dict(zip(df_items['item_id'], df_items['idx']))
    user2idx = dict(zip(df_users['user_id'], df_users['idx']))

    del df_items, df_users

    for key in best_param:
        if key in TIL_CONFIG:
            TIL_CONFIG[key] = best_param[key]

    emb_dim = TIL_CONFIG['emb_dim']
    num_att_layers = TIL_CONFIG['num_att_layers']
    num_heads = TIL_CONFIG['num_heads']
    ffn_mult = TIL_CONFIG['ffn_mult']
    null_p = TIL_CONFIG['null_p']
    dropout = TIL_CONFIG['dropout']

    encoder_user = CrossAttentionReview2User(
        num_obj=data['all_users'],
        emb_dim=emb_dim,
        device=device,
        num_att_layers=num_att_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        null_p=null_p
    )

    encoder_item = CrossAttentionReview2User(
        num_obj=data['all_items'],
        emb_dim=emb_dim,
        device=device,
        num_att_layers=num_att_layers,
        num_heads=num_heads,
        ffn_mult=ffn_mult,
        null_p=null_p
    )

    CARU_embed = context_BPR(encoder_user, encoder_item, device)
    CARU_embed.load_state_dict(torch.load(TIL_CONFIG['model_pth']))
    CARU_embed.eval()

    dataset = TIL_CONFIG['dataset']
    folder = TIL_CONFIG['folder']

    test_data = SimpleNEIDataset(folder, 'test')

    print("----" * 20)
    print(f"Inizio Testing [{now()}]")

    mean_recall, mean_precision, mean_hit, mean_auc = evaluate_recommendation(
        CARU_embed, test_data, k=10, N=2000, U=1280
    )

    print(f"Fine Testing [{now()}]")
    print("----" * 20)
    print(f"===== RISULTATI DEL TEST (Dataset: {dataset}) =====")
    print(f"Recall@{10}:    {mean_recall:.4f}")
    print(f"Precision@{10}: {mean_precision:.4f}")
    print(f"Hit@{10}:       {mean_hit:.4f}")
    print(f"AUC:            {mean_auc:.4f}")
    print("===================================================")

def grid(parameters):
    folder = train_params['folder']
    dataset = parameters['dataset']
    folder = folder + dataset

    grid_folder = folder + "/grid"
    if not os.path.exists(grid_folder):
        os.mkdir(grid_folder)

    global is_grid
    is_grid = True

    grid_param = {
        'emb_dim' : [train_params['emb_dim']],
        'num_att_layers' : [train_params['num_att_layers']],
        'lr': [train_params['lr']]
    }

    if "emb_dim" in parameters:
        grid_param["emb_dim"] = parameters["emb_dim"]

    if "num_att_layers" in parameters:
        grid_param["num_att_layers"] = parameters["num_att_layers"]

    if "lr" in parameters:
        grid_param["lr"] = parameters["lr"]

    filename = "/grid"

    if "seed" in parameters:
        filename += "_"+str(parameters["seed"])

    if "extra" in parameters:
        filename += "_"+str(parameters["extra"])

    filename += ".csv"

    if not os.path.exists(grid_folder + filename):
        header = ['loss', 'param']
        tmp = pd.DataFrame(columns=header)
        tmp.to_csv(grid_folder + filename, index=False)
        del tmp

    df = pd.read_csv(grid_folder + filename)
    stored_params = df['param'].tolist()

    best_loss = float('inf')
    best_params = None
    if not df.empty and 'loss' in df.columns:
        best_idx = df['loss'].idxmin()
        best_loss = df.loc[best_idx, 'loss']
        best_params = json.loads(df.loc[best_idx, 'param'])
        print(f"Best loss attuale: {best_loss}")
        print(f"Best params attuali: {best_params}")

    # Genera tutte le combinazioni
    keys = grid_param.keys()
    values = grid_param.values()

    all_params = [dict(zip(keys, v)) for v in product(*values)]

    if stored_params:
        stored_params = [json.loads(x) for x in stored_params]
        all_params = [x for x in all_params if x not in stored_params]

    train_params['dataset'] = dataset
    for combination in all_params:
        for key in combination:
            if key in train_params:
                train_params[key] = combination[key]

        valid_loss = pretrain()

        new_row = {
            'loss': valid_loss,
            'param': json.dumps(combination)
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(grid_folder + filename, index=False)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_params = combination
            tmp_name = "/model_params"
            if grid_seed:
                tmp_name += '_seed-' + str(grid_seed)
            if extra:
                tmp_name += '_extra-' + str(extra)

            best_grid_name = tmp_name + "_best"
            tmp_name += '.pth'
            best_grid_name += '.pth'

            shutil.move(grid_folder + tmp_name, grid_folder + best_grid_name)

            print(f"Nuova best loss: {best_loss}")
            print(f"Nuovi best params: {best_params}")

    return

def grid_weight(parameters):
    folder = train_params['folder']
    dataset = parameters['dataset']
    folder = folder + dataset

    grid_folder = folder + "/grid"
    if not os.path.exists(grid_folder):
        os.mkdir(grid_folder)

    global is_grid
    is_grid = True

    grid_param = {
        'lr_lambda' : [train_params['lr_lambda']],
        'lr_step_decay': [train_params['lr_step_decay']],
    }

    if "lr_step_decay" in parameters:
        grid_param["lr_step_decay"] = parameters["lr_step_decay"]

    if "lr_lambda" in parameters:
        grid_param["lr_lambda"] = parameters["lr_lambda"]

    filename = "/TIL_grid"

    if "seed" in parameters:
        filename += "_"+str(parameters["seed"])

    if "extra" in parameters:
        filename += "_"+str(parameters["extra"])

    filename += ".csv"

    if not os.path.exists(grid_folder + filename):
        header = ['loss', 'param']
        tmp = pd.DataFrame(columns=header)
        tmp.to_csv(grid_folder + filename, index=False)
        del tmp

    df = pd.read_csv(grid_folder + filename)
    stored_params = df['param'].tolist()

    best_loss = float('inf')
    best_params = None
    if not df.empty and 'loss' in df.columns:
        best_idx = df['loss'].idxmin()
        best_loss = df.loc[best_idx, 'loss']
        best_params = json.loads(df.loc[best_idx, 'param'])
        print(f"Best loss attuale: {best_loss}")
        print(f"Best params attuali: {best_params}")

    # Genera tutte le combinazioni
    keys = grid_param.keys()
    values = grid_param.values()

    all_params = [dict(zip(keys, v)) for v in product(*values)]

    if stored_params:
        stored_params = [json.loads(x) for x in stored_params]
        all_params = [x for x in all_params if x not in stored_params]

    tmp_name = {
        'model': "/TIL_model_params",
        'TIL': "/TIL",
    }

    if grid_seed:
        tmp_name['model'] += '_seed-' + str(grid_seed)
        tmp_name['TIL'] += '_seed-' + str(grid_seed)
    if extra:
        tmp_name['model'] += '_extra-' + str(extra)
        tmp_name['TIL'] += '_extra-' + str(extra)

    best_grid_name = {
        'model': tmp_name['model'] + "_best.pth",
        'TIL': tmp_name['TIL'] + "_best.pth"
    }

    tmp_name['model'] += '.pth'
    tmp_name['TIL'] += '.pth'

    train_params['dataset'] = dataset
    for combination in all_params:
        for key in combination:
            if key in train_params:
                train_params[key] = combination[key]
                print(f"{key}: {train_params[key]}\n")
        print("Setup")
        for key in train_params:
            print(f"{key}: {train_params[key]}")

        valid_loss = train_weight()

        new_row = {
            'loss': valid_loss,
            'param': json.dumps(combination)
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(grid_folder + filename, index=False)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_params = combination

            shutil.move(grid_folder + tmp_name['model'], grid_folder + best_grid_name['model'])
            shutil.move(grid_folder + tmp_name['TIL'], grid_folder + best_grid_name['TIL'])

            print(f"Nuova best loss: {best_loss}")
            print(f"Nuovi best params: {best_params}")

    return best_loss

def train_weight_setup(parameters, grid=False):
    global is_grid
    is_grid = True

    folder = train_params['folder']
    dataset = parameters['dataset']
    folder = folder + dataset

    grid_folder = folder + "/grid"

    filename = "/grid"

    if "seed" in parameters:
        filename += "_"+str(parameters["seed"])

    if "extra" in parameters:
        filename += "_"+str(parameters["extra"])

    filename += ".csv"

    if not os.path.exists(grid_folder + filename):
        raise FileNotFoundError(f"File '{grid_folder + filename}' does not exist")

    df = pd.read_csv(grid_folder + filename)
    best_param =  json.loads(df.loc[df['loss'].idxmin()]['param'])
    del df
    if 'num_epochs' in parameters:
        train_params['num_epochs'] = parameters['num_epochs']

    train_params['dataset'] = dataset
    for key in best_param:
        if key in train_params:
            train_params[key] = best_param[key]

    if grid:
        valid_loss = grid_weight(parameters)
    else:
        valid_loss = train_weight()
    print(f"Valid loss for TIL: {valid_loss}")

if __name__ == "__main__":
    fire.Fire()

