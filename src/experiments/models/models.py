import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from .layers import ConvBlock, UpConvBlock, AttentionBlock


def create_MLP(input_dim, output_dim, hidden_dim=128, n_layers=3):
    """
    Create a Multi-Layer Perceptron (MLP) model.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        hidden_dim (int): Hidden layer dimension
        n_layers (int): Number of hidden layers
    """
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)

def fourier_embedding(t: torch.Tensor, num_bands: int = 6) -> torch.Tensor:
    """
    t: A Tensor of shape (batch_size,)
    num_bands: Number of frequency bands L
    
    Returns:
        A tensor of shape (batch_size, 2 * num_bands)
    """
    assert t.ndim == 1, "t should be a 1D tensor"
    t = t.unsqueeze(1)  # shape (batch_size, 1)
    freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32, device=t.device) * math.pi  # shape (num_bands,)
    angles = t * freqs  # shape (batch_size, num_bands)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # shape (batch_size, 2 * num_bands)
    return emb

class TimeConditionedMLP(nn.Module):
    def __init__(self, x_dim, output_dim, num_fourier_bands=6, hidden_dim=128, n_layers=3):
        super().__init__()
        self.num_fourier_bands = num_fourier_bands
        input_dim = x_dim + 2 * num_fourier_bands
        self.mlp = create_MLP(input_dim, output_dim, hidden_dim, n_layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t: scalar tensor, shape []
        x: tensor of shape (batch_size, ...)
        Returns: tensor of shape (batch_size, output_dim)
        """
        t = t.to(x.device)
        x_shape = x.shape
        x = x.view(x.size(0), -1)
        gamma_t = fourier_embedding(t, self.num_fourier_bands)  # (bs, 2L,)
        full_input = torch.cat([x, gamma_t], dim=1)                    # (bs, x_dim + 2L)
        return self.mlp(full_input).view(x_shape)  # (bs, x_dim)

class TimeAndLabelConditionedMLP(nn.Module):
    def __init__(self,
                 x_dim: int,
                 output_dim: int,
                 num_classes: int,
                 y_emb_dim: int = 16,
                 num_fourier_bands: int = 6,
                 hidden_dim: int = 128,
                 n_layers: int = 3):
        super().__init__()
        self.num_fourier_bands = num_fourier_bands
        # label embedding
        self.y_embed = nn.Embedding(num_classes, y_emb_dim)
        # total input size = x + time‐emb + y‐emb
        input_dim = x_dim + 2*num_fourier_bands + y_emb_dim
        self.mlp = create_MLP(input_dim, output_dim, hidden_dim, n_layers)

    def forward(self,
                t: torch.Tensor,               # (,) or (bs,)
                x: torch.Tensor,               # (bs, x_dim...)
                cond_mask: torch.Tensor,       # (bs,) boolean
                y: torch.LongTensor = None):   # (bs,) labels
        
        bs = x.shape[0]
        device = x.device

        # flatten x
        x_flat = x.view(bs, -1)           # (bs, x_dim)

        # time embedding
        t = t.to(device)
        # if t is scalar, expand to batch
        if t.ndim == 0:
            t = t.repeat(bs)
        gamma_t = fourier_embedding(t, self.num_fourier_bands)  # (bs, 2L)

        # label embedding (masked)
        if y is None:
            raise ValueError("y must be provided for label-conditioned MLP")
        y_emb = self.y_embed(y.to(device))                      # (bs, y_emb_dim)
        mask  = cond_mask.to(device).float().unsqueeze(1)       # (bs,1)
        y_emb = y_emb * mask                                     # zero-out where cond_mask==False

        # concat everything
        inp = torch.cat([x_flat, gamma_t, y_emb], dim=1)        # (bs, total_dim)
        out = self.mlp(inp)                                     # (bs, output_dim)

        # restore original x shape if needed
        return out.view(x.shape[0], *out.shape[1:])

# === FiLM layer ===
class FiLM(nn.Module):
    def __init__(self, time_dim, num_features):
        super().__init__()
        self.linear = nn.Linear(time_dim, 2 * num_features)

    def forward(self, x, t_emb):
        gamma_beta = self.linear(t_emb)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return gamma * x + beta



# === Convolutional block with FiLM ===
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, gn=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels) if gn else nn.Identity(),
            nn.ReLU()
        )
        self.film = FiLM(time_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv(x)
        return self.film(h, t_emb)


# === UNet architecture ===
class TimeConditionedUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_fourier_bands=6):
        super().__init__()
        self.time_dim = 2 * num_fourier_bands
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, 128)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, 128)
        self.down = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 2, 128)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels, 128)
        self.dec2 = ConvBlock(base_channels + in_channels, in_channels, 128, gn=False)

    def forward(self, t, x):
        """
        Args:
            t: Tensor of shape (B,)
            x: Tensor of shape (B, C, 28, 28)
        Returns:
            Output: Tensor of shape (B, C, 28, 28)
        """
        B = x.shape[0]
        if t.ndim == 0:
            t = t.expand(B)
        t_emb = self.embedding_mlp(fourier_embedding(t, self.time_dim // 2))

        # Encoder
        h1 = self.enc1(x, t_emb)               # (B, C, 28, 28)
        h2 = self.enc2(self.down(h1), t_emb)   # (B, 2C, 14, 14)

        # Bottleneck
        h3 = self.bottleneck(h2, t_emb)        # (B, 2C, 14, 14)

        # Decoder
        h3_up = self.up(h3)                    # (B, 2C, 28, 28)
        u1 = self.dec1(torch.cat([h3_up, h1], dim=1), t_emb)  # (B, C, 28, 28)

        out = self.dec2(torch.cat([u1, x], dim=1), t_emb)     # (B, 1, 28, 28)
        return out


def load_model(path_checkpoint, modelClass: torch.nn.Module, **kwargs):
    model = modelClass(**kwargs)
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


def save_model(path_checkpoint, model):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path_checkpoint,
    )
