import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Attention import AnomalyAttention, AttentionLayer
from layers.Embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence loss for anomaly detection."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # Handle both old params dict format and new config format
        if hasattr(config, 'win_size'):
            # New config format
            self.window_size = config.win_size
            self.feature_num = config.enc_in
            self.num_transformer_blocks = config.e_layers
            self.num_heads = config.n_heads
            self.embedding_dims = config.d_model
            self.attn_dropout = config.dropout
            self.ff_dropout = config.dropout
            self.k = getattr(config, 'k', 3)
            activation = config.activation
            output_attention = config.output_attention
        else:
            # Old params dict format (for backward compatibility)
            self.window_size = config.get('window_size', 100)
            self.feature_num = config.get('feature_num', 7)
            self.num_transformer_blocks = config.get('n_layer', 3)
            self.num_heads = config.get('n_head', 4)
            self.embedding_dims = config.get('hidden_size', 512)
            self.attn_dropout = config.get('attn_pdrop', 0.1)
            self.ff_dropout = config.get('resid_pdrop', 0.1)
            self.k = config.get('k', 3)
            activation = config.get('activation', 'gelu')
            output_attention = config.get('output_attention', True)

        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(self.feature_num, self.embedding_dims, self.ff_dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(self.window_size, False, attention_dropout=self.attn_dropout, output_attention=output_attention),
                        self.embedding_dims, self.num_heads),
                    self.embedding_dims,
                    self.embedding_dims,
                    dropout=self.attn_dropout,
                    activation=activation
                ) for l in range(self.num_transformer_blocks)
            ],
            norm_layer=torch.nn.LayerNorm(self.embedding_dims)
        )

        self.projection = nn.Linear(self.embedding_dims, self.feature_num, bias=True)

    def forward(self, x):
        """Forward pass for reconstruction."""
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input data."""
        if self.output_attention:
            output, _, _, _ = self.forward(x)
            return output
        else:
            return self.forward(x)
    
    def detect_anomaly(self, x: torch.Tensor) -> tuple:
        """
        Detect anomalies using AnomalyTransformer approach.
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            Tuple of (reconstructed_output, anomaly_scores)
        """
        output, series, prior, _ = self.forward(x)
        
        # Calculate reconstruction loss
        rec_loss = torch.mean((x - output) ** 2, dim=-1)
        
        # Calculate series loss (Association Discrepancy)
        series_loss = 0.0
        prior_loss = 0.0
        temperature = 50
        
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)
                ).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)
                ).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)),
                    series[u].detach()) * temperature

        # Calculate final anomaly scores
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        anomaly_scores = metric * rec_loss
        
        return output, anomaly_scores
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get anomaly scores for input data."""
        _, scores = self.detect_anomaly(x)
        return scores
    
    def compute_loss(self, x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Compute training loss for AnomalyTransformer.
        
        Args:
            x: Input tensor
            criterion: Loss function
            
        Returns:
            Combined loss tensor
        """
        output, series, prior, _ = self.forward(x)
        
        # Reconstruction loss
        rec_loss = criterion(output, x)
        
        # Series loss (Association Discrepancy)
        series_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)
            ).detach())) + torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)).detach(),
                          series[u])))
        series_loss /= len(prior)
        
        # Combined loss
        total_loss = rec_loss - self.k * series_loss
        return torch.mean(total_loss)