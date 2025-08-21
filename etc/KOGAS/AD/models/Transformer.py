import torch.nn as nn
from layers.Embed import PositionalEmbedding
from layers.Transformer_Enc import Transformer, PreNorm


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.feature_num = params['feature_num']
        self.num_transformer_blocks = params['n_layer']
        self.num_heads = params['n_head']
        self.embedding_dims = params['hidden_size']
        self.attn_dropout = params['attn_pdrop']
        self.ff_dropout = params['resid_pdrop']

        self.position_embedding = PositionalEmbedding(d_model=self.embedding_dims)
        self.value_embedding = nn.Linear(self.feature_num, self.embedding_dims)

        self.transformer = Transformer(self.embedding_dims,
                                       self.num_transformer_blocks,
                                       self.num_heads,
                                       self.embedding_dims,
                                       self.attn_dropout,
                                       self.ff_dropout)

        self.dropout = nn.Dropout(self.ff_dropout)

        self.output_layer = nn.Linear(self.embedding_dims, self.feature_num)

    def forward(self, x, use_attn=False):
        b, w, f = x.shape
        x = self.value_embedding(x)

        position_emb = self.position_embedding(x)
        x += position_emb
        x = self.dropout(x)

        x, _ = self.transformer(x, use_attn=use_attn)
        x = self.output_layer(x)

        return x