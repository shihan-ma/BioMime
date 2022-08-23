import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.residual = cfg.residual
        self.num_layers = cfg.num_layers

        act = nn.LeakyReLU(cfg.negative_slope)
        if isinstance(cfg.hidden_channels, list):
            h = cfg.hidden_channels
        else:
            h = [cfg.hidden_channels] * (self.num_layers - 1)
        if isinstance(cfg.kernel, list):
            kernel = cfg.kernel
        else:
            kernel = [cfg.kernel] * self.num_layers
        if isinstance(cfg.pad, list):
            pad = cfg.pad
        else:
            pad = [cfg.pad] * self.num_layers
        if isinstance(cfg.stride, list):
            stride = cfg.stride
        else:
            stride = [cfg.stride] * self.num_layers
        self.enc_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(i, o, k, s, p),
                act,
            ) for i, o, k, s, p in zip([cfg.in_channel] + [h[0] + cfg.num_conds] + h[1:], h + [cfg.out_channel], kernel, stride, pad)
        )
        self.conv11 = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(i, i, 1, 1, 0),
                act,
            ) for i in h + [cfg.out_channel]
        )

        self.avg_final = nn.AvgPool3d(cfg.final_dim)
        self.conv_final = nn.Conv3d(cfg.out_channel, 1, 1, 1)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)

            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.1)

    def forward(self, x, c):
        """
        Args:
            x: input batch (Tensor) [bs, 1, t, h, w]
            c: specified absolute conditions [bs, num_specified_conditions]
        """

        if self.residual:
            x = self.enc_conv[0](x)
            x = self.conv11[0](x) + x
            bs, _, T, H, W = x.shape
            c = c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, T, H, W)
            x = torch.cat((x, c), dim=1)
            for i in range(1, self.num_layers):
                x = self.enc_conv[i](x)
                x = self.conv11[i](x) + x
        else:
            x = self.enc_conv[0](x)
            x = self.conv11[0](x)
            x = torch.cat((x, c), dim=1)
            for i in range(1, self.num_layers):
                x = self.enc_conv[i](x)
                x = self.conv11[i](x)

        x = self.avg_final(x)
        x = self.conv_final(x).squeeze()

        return x
