import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock, UpConvBlock, AttentionBlock


# Adapted from https://www.kaggle.com/code/truthisneverlinear/attention-u-net-pytorch
class AttentionUNet(nn.Module):
    """
    Configurable implementation of the Attention U-Net model.
    
    Args:
        channels (list): List of channel numbers for each level, including input and output
                        e.g., [1, 64, 128, 256, 512, 1024] for original architecture
        n_classes (int): Number of output classes
        attention_levels (list, optional): List of booleans indicating whether to use 
                                        attention at each decoder level. If None, use
                                        attention at all levels.
    """
    def __init__(self, channels, n_classes=1, attention_levels=None, device=None):
        super().__init__()
        self.device = device

        if len(channels) < 3:
            raise ValueError("channels list must have at least 3 elements (in, hidden, out)")
            
        self.depth = len(channels) - 1
        self.channels = channels
        
        # Default: attention at all levels if not specified
        if attention_levels is None:
            attention_levels = [True] * (self.depth - 1)
        elif len(attention_levels) != self.depth - 1:
            raise ValueError(f"attention_levels must have {self.depth - 1} elements")
            
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(self.depth):
            self.encoder_blocks.append(
                ConvBlock(ch_in=channels[i], ch_out=channels[i+1])
            )
        
        # Decoder blocks with attention
        self.up_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(self.depth - 1, 0, -1):
            # Upsampling block
            self.up_blocks.append(
                UpConvBlock(ch_in=channels[i+1], ch_out=channels[i])
            )
            
            # Attention block (if specified for this level)
            if attention_levels[i-1]:
                self.attention_blocks.append(
                    AttentionBlock(
                        f_g=channels[i],
                        f_l=channels[i],
                        f_int=channels[i]//2
                    )
                )
            else:
                self.attention_blocks.append(None)
            
            # Decoder conv block
            self.decoder_blocks.append(
                ConvBlock(ch_in=channels[i]*2, ch_out=channels[i])
            )
        
        # Final 1x1 convolution
        self.conv_1x1 = nn.Conv2d(channels[1], n_classes,
                                 kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_features = []
        
        # Encoder path
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            if i < self.depth - 1:  # Don't store the last encoding
                encoder_features.append(x)
                x = self.maxpool(x)
        
        # Decoder path
        for i in range(self.depth - 1):
            # Upsample
            x = self.up_blocks[i](x)
            
            # Get corresponding encoder features
            enc_features = encoder_features[-(i+1)]
            
            # Apply attention if specified for this level
            if self.attention_blocks[i] is not None:
                enc_features = self.attention_blocks[i](g=x, x=enc_features)
            
            # Concatenate and apply convolution
            x = torch.cat([enc_features, x], dim=1)
            x = self.decoder_blocks[i](x)
        
        # Final 1x1 convolution
        return self.conv_1x1(x)

# Example usage:
def create_attention_unet(in_channels=1, n_classes=1, base_channels=64, depth=5, attention_levels=None, device=None):
    """
    Helper function to create an AttentionUNet with specified parameters.
    
    Args:
        in_channels (int): Number of input channels
        n_classes (int): Number of output classes
        base_channels (int): Number of channels in first layer
        depth (int): Number of down/up-sampling steps
        attention_levels (list, optional): List of booleans for attention layers
    """
    channels = [in_channels]
    for i in range(depth):
        channels.append(base_channels * (2**i))
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return AttentionUNet(
        channels=channels,
        n_classes=n_classes,
        attention_levels=attention_levels,
        device=device
    ).to(device)


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

if __name__ == "__main__":
    # 1. Original architecture
    model = create_attention_unet(
        in_channels=3,
        n_classes=1,
        base_channels=64,
        depth=5
    )

    # 2. Custom channel configuration
    channels = [3, 32, 64, 128, 256, 512]  # Custom channel sizes
    model = AttentionUNet(
        channels=channels,
        n_classes=1
    )

    # 3. Selective attention
    attention_levels = [True, False, True, False]  # Only use attention at certain levels
    model = create_attention_unet(
        in_channels=3,
        n_classes=1,
        base_channels=64,
        depth=5,
        attention_levels=attention_levels
    )
    print("Model created successfully!")