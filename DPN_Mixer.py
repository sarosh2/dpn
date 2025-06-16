import torch
from torch import nn
from dpn_3.dpn import DPN
class Patch_Embedding(nn.Module):
    """Patch Embedding layer is nothing but a convolutional layer
       with kernelsize and stride equal to patch size."""
    def __init__(self, in_channels, embedding_dim, patch_size):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels, embedding_dim, patch_size, patch_size)

    def forward(self, x):
        return self.patch_embedding(x)

class MLP(nn.Module):
    """This network applies 2 consecutive fully connected layers and is used
       in Token Mixer and Channel Mixer modules."""
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.mlp = DPN(dim, intermediate_dim + dim, dim, True)

    def forward(self, x):
        return self.mlp(x)

class T1(nn.Module):
    """The transformation that is used in Mixer Layer (the T)
       which just swithes the 2nd and the 3rd dimensions and is
       applied before and after Token Mixing MLPs"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))

class T2(nn.Module):
    """The transformation that is applied right after the patch embedding layer
       and convert it's shape from (batch_size, embedding_dim, sqrt(num_patches), sqrt(num_patches))
       to (batch_size, num_patches, embedding_dim)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 3, 1)).reshape(x.shape[0], -1, x.shape[1])

class MixerLayer(nn.Module):
    """Mixer layer which consists of Token Mixer and Channel Mixer modules
       in addition to skip connections.
       intermediate_output = Token Mixer(input) + input
       final_output = Channel Mixer(intermediate_output) + intermediate_output"""
    def __init__(self, embedding_dim, num_patch, token_intermediate_dim, channel_intermediate_dim):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            T1(),
            MLP(num_patch, token_intermediate_dim),
            T1()
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            MLP(embedding_dim, channel_intermediate_dim),
        )

    def forward(self, x):

        x = x + self.token_mixer(x)    # Token mixer and skip connection
        x = x + self.channel_mixer(x)  # channel mixer and skip connection

        return x

class MLPMixer(nn.Module):
    """MLP-Mixer Architecture:
       1-Applies 'Patch Embedding' at first.
       2-Applies 'Mixer Layer' N times in a row.
       3-Performs 'Global Average Pooling'
       4-The learnt features are then passed to the classifier"""
    def __init__(self, in_channels, embedding_dim, num_classes, patch_size, image_size, depth, token_intermediate_dim, channel_intermediate_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.pe = nn.Sequential(
            Patch_Embedding(in_channels, embedding_dim, patch_size),
            T2(),
        )

        self.mixers = nn.ModuleList([MixerLayer(embedding_dim, self.num_patch, token_intermediate_dim * depth // 3, channel_intermediate_dim * depth // 3) for _ in range(3)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, num_classes))

    def forward(self, x):

        x = self.pe(x)              # Patch Embedding layer
        for mixer in self.mixers:   # Applying Mixer Layer N times
            x = mixer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)           # Global Average Pooling

        return self.classifier(x)