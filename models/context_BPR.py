import torch.nn as nn
import torch as tc
from TIL.models import CrossAttentionReview2User

class context_BPR(nn.Module):

    def __init__(self, encoder_user, encoder_item, device, dropout=0.2):
        super(context_BPR, self).__init__()

        self.encoder_item : CrossAttentionReview2User = encoder_item
        self.encoder_user : CrossAttentionReview2User = encoder_user

        self.b_users = nn.Parameter(tc.randn(encoder_user.num_obj, 1))
        self.b_items = nn.Parameter(tc.randn(encoder_item.num_obj, 1))

        self.mlp = nn.Sequential(
            nn.Linear(3 * encoder_user.emb_dim, encoder_user.emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_user.emb_dim, 1),
        )

        self.to(device)
        self.device = device

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.b_users)
        nn.init.xavier_uniform_(self.b_items)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def score(self, user_emb, item_emb, user_id, item_id):
        user_emb = user_emb.to(self.device)
        item_emb = item_emb.to(self.device)

        x = tc.cat([user_emb * item_emb, user_emb, item_emb], dim=-1)

        score = self.mlp(x) + self.b_users[user_id] + self.b_items[item_id]
        return score.squeeze(-1)

    def embedding(self, encoder, obj_id, obj_tokens = None, p = False):
        obj_id = obj_id.to(self.device)

        if obj_tokens is not None:
            obj_tokens = obj_tokens.to(self.device)

        obj_emb = None
        if encoder == "user":
            obj_emb = self.encoder_user(obj_id, obj_tokens, p = p)
        if encoder == "item":
            obj_emb = self.encoder_item(obj_id, obj_tokens, p = p)

        return obj_emb

    def forward(self, user_id, pos_item_id, neg_item_id,
                user_tokens = None, pos_tokens = None, neg_tokens = None,
                p = False, dual_user = False):

        # Generate 2 embeddings from the same user
        # pos_user uses positive tokens & neg_user uses negative tokens
        if dual_user and user_tokens is not None:
            pos_tokens, neg_tokens = user_tokens
            pos_user = self.embedding("user", user_id, pos_tokens, p = p)
            neg_user = self.embedding("user", user_id, neg_tokens, p = p)

            user_embs = (pos_user, neg_user)
        elif dual_user:
            # Same embedding duplicated without context
            tmp = self.embedding("user", user_id, None, p=p)
            user_embs = (tmp, tmp)
        else:
            # Either user_tokens is none or has tokens
            # NOTE: user_tokens = None is handled inside
            user_embs = self.embedding("user", user_id, user_tokens, p = p)

        pos_item_emb = self.embedding("item", pos_item_id, pos_tokens, p)
        neg_item_emb = self.embedding("item", neg_item_id, neg_tokens, p)

        return user_embs, pos_item_emb, neg_item_emb

    def compute_loss(self, user_emb, user_id, pos_item_emb, pos_item_id, neg_item_emb, neg_item_id, dual_user = False):
        user_id = user_id.to(self.device)

        if dual_user:
            pos_user_emb, neg_user_emb = user_emb[0].to(self.device), user_emb[1].to(self.device)
        else:
            pos_user_emb = neg_user_emb = user_emb.to(self.device)

        pos_item_emb = pos_item_emb.to(self.device)
        neg_item_emb = neg_item_emb.to(self.device)

        y_ui = self.score(pos_user_emb, pos_item_emb, user_id, pos_item_id)
        y_uj = self.score(neg_user_emb, neg_item_emb, user_id, neg_item_id)

        return (-tc.log(tc.sigmoid(y_ui - y_uj) + 1e-12)).squeeze(dim=-1)