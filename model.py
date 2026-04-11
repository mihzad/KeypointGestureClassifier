import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# -------------------------
# 1. Pre-processing Module
# -------------------------
class HandNormalizer(nn.Module):
    """
    Normalizes coordinates to be 'Hand-Centric' and 'Scale-Invariant'.
    Input: (B, 16, 42) -> Output: (B, 16, 42)
    """
    def __init__(self, jitter_strength=0.02):
        super().__init__()
        self.jitter_strength = jitter_strength
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Reshape to (B, T, 21 joints, 2 coords)
        x = x.view(batch_size, seq_len, 21, 2)

        # 1. Wrist-Centering (Wrist is index 0)
        # Subtract wrist (x,y) from all joints across all frames
        wrist = x[:, :, 0:1, :] 
        x = x - wrist

        # 2. Scale Normalization (Using distance from wrist to middle finger MCP, index 9)
        # We use a small epsilon to avoid division by zero
        ref_dist = torch.norm(x[:, :, 9, :] - x[:, :, 0, :], dim=-1, keepdim=True).unsqueeze(-1)
        x = x / (ref_dist + 1e-6)

        # Flatten back to (B, 16, 42)
        x = x.view(batch_size, seq_len, 42)

        #post normalization jitter for training phase stability
        if self.training:
            x = x + self.jitter_strength * torch.randn_like(x)
        return x

# -------------------------
# 2. Refined Residual Block
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, inner_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, inner_ch, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(inner_ch)
        self.conv2 = nn.Conv1d(inner_ch, out_ch, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# -------------------------
# 3. Temporal Encoder
# -------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, in_channels=42, stem_channels=64, inner_residuals_dim=128,
                  between_residuals_dim=128, out_features=256):
        super().__init__()
        # Initial projection to a higher feature space
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU()
        )

        # Using fewer downsamples to preserve 16-frame resolution
        self.layer1 = ResidualBlock(in_ch=stem_channels,inner_ch=inner_residuals_dim, out_ch=between_residuals_dim, stride=1)
        self.layer2 = ResidualBlock(in_ch=between_residuals_dim, inner_ch=inner_residuals_dim, out_ch=out_features, stride=2) # 16 -> 8 frames

        # Dual Pooling captures "Peak" and "Average" movement
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # (B, 16, 42) -> (B, 42, 16)
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Combine poolings for (B, 256)
        return (self.avg_pool(x)).squeeze(-1) #(self.avg_pool(x) + self.max_pool(x)).squeeze(-1)

# -------------------------
# 4. Final Gesture Model
# -------------------------
class GestureNetV1(nn.Module):
    def __init__(self, num_classes, train_postnorm_jitter=0.02,
                TE_stem_channels=32,
                TE_inner_residuals_dim=32, TE_between_residuals_dim=64, features_per_encoder=64,
                classifier_hidden_dim=128):
        super().__init__()
        self.normalizer = HandNormalizer(jitter_strength=train_postnorm_jitter)
        
        self.pos_encoder = TemporalEncoder(in_channels=42, stem_channels=TE_stem_channels,
                                            inner_residuals_dim=TE_inner_residuals_dim,
                                            between_residuals_dim=TE_between_residuals_dim,
                                            out_features=features_per_encoder)
        self.vel_encoder = TemporalEncoder(in_channels=42, stem_channels=TE_stem_channels,
                                            inner_residuals_dim=TE_inner_residuals_dim,
                                            between_residuals_dim=TE_between_residuals_dim,
                                            out_features=features_per_encoder)

        self.classifier = nn.Sequential(
            nn.Linear(features_per_encoder*2, classifier_hidden_dim), #256 - 256
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(classifier_hidden_dim, num_classes) #256
        )

    def forward(self, x):
        # 1. Coordinate Normalization (Invariant to hand position/size)
        x_norm = self.normalizer(x)

        # 2. Velocity Calculation (Delta between frames)
        # Note: We use x_norm for velocity so speed is also relative to hand size
        vel = torch.zeros_like(x_norm)
        vel[:, 1:, :] = x_norm[:, 1:, :] - x_norm[:, :-1, :]

        # 3. Feature Extraction
        pos_feat = self.pos_encoder(x_norm)
        vel_feat = self.vel_encoder(vel)

        # 4. Fusion & Classification
        combined = torch.cat([pos_feat, vel_feat], dim=1)
        return self.classifier(combined)

def create_model(num_classes, additional_model_params: dict | None):
    if additional_model_params is None:
        additional_model_params = {}
    model = GestureNetV1(num_classes, **additional_model_params)

    model.cuda(0)

    summary(
        model,
        input_size=(64, 16, 42), # (B,T,J*2)
        depth=10,
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names", "depth"],
    )
    return model