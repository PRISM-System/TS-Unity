import torch
import torch.nn as nn

## OmniAnomaly Model (KDD 19)
class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		
		# Handle both old params dict format and new config format
		if hasattr(config, 'enc_in'):
			# New config format
			self.beta = getattr(config, 'beta', 0.01)
			self.n_feats = config.enc_in
			self.n_hidden = getattr(config, 'hidden_size', 64)
			self.n_latent = getattr(config, 'latent_size', 32)
		else:
			# Old params dict format (for backward compatibility)
			self.beta = config.get('beta', 0.01)
			self.n_feats = config.get('feature_num', 7)
			self.n_hidden = config.get('hidden_size', 64)
			self.n_latent = config.get('latent_size', 32)

		self.lstm = nn.GRU(self.n_feats, self.n_hidden, 2, batch_first=True)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			#nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		#hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)

		return x, mu, logvar, hidden
	
	def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Reconstruct input data using OmniAnomaly VAE.
		
		Args:
			x: Input tensor (batch_size, seq_len, feature_size)
			
		Returns:
			Reconstructed output tensor (batch_size, seq_len, feature_size)
		"""
		x_recon, _, _, _ = self.forward(x)
		return x_recon
	
	def detect_anomaly(self, x: torch.Tensor) -> tuple:
		"""
		Detect anomalies using OmniAnomaly reconstruction error.
		
		Args:
			x: Input tensor (batch_size, seq_len, feature_size)
			
		Returns:
			Tuple of (reconstructed_output, anomaly_scores)
		"""
		x_recon, mu, logvar, _ = self.forward(x)
		
		# Calculate reconstruction error as anomaly score
		anomaly_scores = torch.mean((x - x_recon) ** 2, dim=(1, 2))
		
		return x_recon, anomaly_scores
	
	def get_anomaly_score(self, x: torch.Tensor, method: str = 'mse') -> torch.Tensor:
		"""
		Calculate anomaly scores using reconstruction error.
		
		Args:
			x: Input tensor (batch_size, seq_len, feature_size)
			method: Scoring method ('mse', 'mae', 'combined', 'vae')
			
		Returns:
			Anomaly scores tensor
		"""
		if method == 'vae':
			_, scores = self.detect_anomaly(x)
			return scores
		else:
			reconstructed = self.reconstruct(x)
			if method == 'mse':
				return torch.mean((x - reconstructed) ** 2, dim=(1, 2))
			elif method == 'mae':
				return torch.mean(torch.abs(x - reconstructed), dim=(1, 2))
			elif method == 'combined':
				mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
				mae = torch.mean(torch.abs(x - reconstructed), dim=(1, 2))
				return 0.5 * (mse + mae)
			else:
				raise ValueError(f"Unknown scoring method: {method}")
	
	def compute_loss(self, x: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
		"""
		Compute training loss for OmniAnomaly model.
		
		Args:
			x: Input tensor (batch_size, seq_len, feature_size)
			criterion: Loss function for reconstruction
			
		Returns:
			Combined loss tensor (reconstruction + KL divergence)
		"""
		x_recon, mu, logvar, _ = self.forward(x)
		
		# Reconstruction loss
		rec_loss = criterion(x_recon, x)
		
		# KL divergence loss
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
		kl_loss = torch.mean(kl_loss)
		
		# Combined VAE loss
		total_loss = torch.mean(rec_loss) + self.beta * kl_loss
		
		return total_loss