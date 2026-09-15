import torch
from torch import nn

"""
UBAC CNN Model Based on residual connections and layer normalisations.
ELU Activation used due to self normalisation properties,
and ability to produce -1 outputs.

ResidualBlocks: Are the blocks that define UBAC,
residual connections allow gradients to flow 
uninterrupted, combined with layer norm prevent
explosions of values. Dropout and BN used for regularisation.
"""
class ResidualBlocks(nn.Module):
    def __init__(self, kernel=3, padding=1, channels=32):
        super(ResidualBlocks, self).__init__()
        self.device_param = nn.Parameter(torch.zeros([1]))
        self.conv_b = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=padding, padding_mode="replicate"),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=padding, padding_mode="replicate"),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.4)

    def forward(self, X):
        return self.conv_b(X) + X

"""
The complete UBAC agent. Conv_b0 is the channel scaling head, purely
to raise the channel count. A limitation of UBAC is the fixed channel count.
"""
class ConvolutionalAgent(nn.Module):
    def __init__(self, action_space):
        super(ConvolutionalAgent, self).__init__()

        self.conv_b0 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2, padding_mode="replicate", dilation=2),
            nn.GELU(),
        )
        self.conv_k5 = ResidualBlocks(kernel=3, padding=1, channels=16)
        self.conv_k3 = ResidualBlocks(channels=16)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.mlp_b1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=action_space),
            nn.Tanh()
        )

    def forward(self, x):
        conv_channel_scale = self.conv_b0(x)
        conv_y1 = self.max_pool(self.conv_k5(conv_channel_scale))
        conv_y2 = self.max_pool(self.conv_k5(conv_y1))
        conv_y3 = self.max_pool(self.conv_k3(conv_y2))
        conv_y4 = self.max_pool(self.conv_k3(conv_y3))
        mlp_y1 = self.mlp_b1(conv_y4)
        return mlp_y1

if __name__ == "__main__":
    model = ConvolutionalAgent(action_space=1)
    print(f"Model created, number of parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")
    print(model(torch.randn(1, 3, 160, 80)).shape)