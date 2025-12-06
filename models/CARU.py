import torch.nn as nn
import torch as tc
import torch.nn.functional as F


class CrossAttentionReview2User(nn.Module):

    def __init__(self, num_obj, emb_dim = 128, review_dim=1536, null_p=0.5, dropout=0.2,
                 num_att_layers=3, num_heads=8, ffn_mult=4,
                 device="cpu", min_=1, max_=5):
        super(CrossAttentionReview2User, self).__init__()

        self.object_embedding = nn.Embedding(num_obj, emb_dim)
        self.null = nn.Parameter(tc.randn(review_dim) * 0.02)

        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention_norm': nn.LayerNorm(emb_dim),
                'attention': nn.MultiheadAttention(
                    emb_dim, num_heads, dropout=dropout, batch_first=True, kdim= review_dim, vdim= review_dim
                ),
                'dropout': nn.Dropout(dropout),

                'ffn': nn.Sequential(
                    nn.Linear(emb_dim, ffn_mult * emb_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_mult * emb_dim, emb_dim),
                ),
                'ffn_norm': nn.LayerNorm(emb_dim),
                'ffn_dropout': nn.Dropout(dropout),
            })
            for _ in range(num_att_layers)
        ])

        self.num_att_layers = num_att_layers
        self.num_heads = num_heads

        self.max_ = max_
        self.min_ = min_
        self.num_obj = num_obj
        self.null_p = null_p

        self.emb_dim = emb_dim
        self.review_dim = review_dim

        self.to(device)
        self.device = device

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.MultiheadAttention):
                if m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if m.out_proj.weight is not None:
                    nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

        # opzionale: re-inizializza anche il parametro "null"
        with tc.no_grad():
            self.null.normal_(mean=0.0, std=0.02)

    def forward(self, obj_id, tokens=None, p = False):
        obj_id = obj_id.to(self.device)
        b_size = obj_id.size(0)

        if tokens is None:
            tokens = tc.zeros(b_size, self.review_dim, device=self.device)

        tokens = tokens.to(self.device)
        y_ = (tokens == 0).all(dim=1)
        x = self.object_embedding(obj_id)
        null_b = self.null.expand(b_size, -1)

        if p:
            y = tc.rand(b_size, device=self.device) > self.null_p
        else:
            y = tc.zeros(b_size, device=self.device, dtype=tc.bool)

        # If review tokens are not present then it is forced the null sequence
        if y_.any().item():
            y = y & ~y_

        review_emb = tc.where(
            y.unsqueeze(1),
            tokens, # y = True
            null_b  # y = False
        )

        x = x.unsqueeze(1)
        review_emb = review_emb.unsqueeze(1)

        for i, layer in enumerate(self.cross_layers):
            x_norm = layer['attention_norm'](x)
            ca, _ = layer['attention'](x_norm, review_emb, review_emb)
            x = x + layer['dropout'](ca)

            x_norm = layer['ffn_norm'](x)
            ffn_out = layer['ffn'](x_norm)
            x = x + layer['ffn_dropout'](ffn_out)

        x = x.squeeze(1)
        return x