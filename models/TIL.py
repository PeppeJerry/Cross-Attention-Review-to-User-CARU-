from torch import nn
import torch as tc

from TIL.dataset import NEIDataset


class TIL(nn.Module):
    def __init__(self, encoder_item, device):
        super(TIL, self).__init__()

        self.nu = {}
        self.encoder_item = encoder_item

        self.w1 = nn.Sequential(
            nn.Linear(2 * encoder_item.emb_dim, encoder_item.emb_dim),
            nn.ReLU(),
        )

        self.norm1 = nn.LayerNorm(2 * encoder_item.emb_dim)

        self.w2 = nn.Sequential(
            nn.Linear(encoder_item.emb_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()
        self.to(device)
        self.device = device

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def aggregate(self, train_set : NEIDataset):
        item_ids = []
        user_ids = {}

        for user in train_set.positive_users:
            positives = train_set.positive_items[user]
            user_positives = [item['item_id'] for item in positives]
            user_ids[user] = user_positives
            item_ids.extend(user_positives)

        item_ids = list(set(item_ids))
        id2idx = {iid: idx for idx, iid in enumerate(item_ids)}

        item_embeddings = self.encoder_item(tc.tensor(item_ids).to(self.device))

        for user_id in train_set.users:

            # L'utente non ha item positivi nello storico
            if user_id not in user_ids:
                self.nu[user_id] = tc.zeros(item_embeddings.shape[1]).to(self.device)
                continue

            indexes = [id2idx[x] for x in user_ids[user_id] if x in id2idx]

            # Ci sono indici di item non corretti
            if not indexes:
                self.nu[user_id] = tc.zeros(item_embeddings.shape[1]).to(self.device)
                continue

            # Aggrego lo storico di item positivi
            user_embeddings = item_embeddings[tc.tensor(indexes).to(self.device)]
            self.nu[user_id] = user_embeddings.mean(dim=0).clone().to(self.device)


    def forward(self, user_id, user_embs, pos_item_emb, neg_item_emb, dual_user=False):

        user_id = user_id.to(self.device)
        if dual_user:
            pos_user_emb, neg_user_emb = user_embs[0].to(self.device),user_embs[1].to(self.device)
        else:
            pos_user_emb = neg_user_emb = user_embs.to(self.device)

        pos_item_emb = pos_item_emb.to(self.device)
        neg_item_emb = neg_item_emb.to(self.device)

        nu_batch = tc.stack([self.nu[int(u.item())].detach() for u in user_id]).to(self.device)

        a = pos_item_emb * pos_user_emb
        b = pos_item_emb * nu_batch
        c = neg_item_emb * neg_user_emb
        d = neg_item_emb * nu_batch

        s_uij = tc.cat([(a + b), (c + d)], dim=-1)
        s_uij = self.norm1(s_uij)
        z_uij = self.w1(s_uij)
        w_uij = self.w2(z_uij)
        return w_uij.squeeze(-1)