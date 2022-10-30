import torch
import torch.nn as nn


class Conv3dEnc(nn.Module):
    def __init__(self, num_layers, in_channel, out_channel, hidden_channels, kernel, stride, pad, act=nn.PReLU()):
        super(Conv3dEnc, self).__init__()

        self.num_layers = num_layers
        if isinstance(hidden_channels, list):
            h = hidden_channels
        else:
            h = [hidden_channels] * (num_layers - 1)
        if not isinstance(kernel, list):
            kernel = [kernel] * num_layers
        elif len(kernel) != num_layers:
            kernel = [kernel[-1]] * num_layers
        if not isinstance(pad, list):
            pad = [pad] * num_layers
        elif len(pad) != num_layers:
            pad = [pad[-1]] * num_layers

        self.enc = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(i, o, k, s, p),
                act,
            ) for i, o, k, s, p in zip([in_channel] + h, h + [out_channel], kernel, stride, pad)
        )
        self.conv11 = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(i, i, 1, 1, 0),
                act,
            ) for i in h + [out_channel]
        )
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        """
        Args:
            x: input batch (Tensor) [bs, 1, t, h, w]
        """
        for i in range(self.num_layers):
            x = self.enc[i](x)
            x = self.conv11[i](x) + x
        return x


class ExpertClip(nn.Module):
    def __init__(self, scale_factor, tgt_sample):
        super(ExpertClip, self).__init__()

        self.scale = scale_factor
        self.tgt_sample = tgt_sample

    def forward(self, x):
        t, w, h = self.tgt_sample
        tgt_time_sp = int(t * self.scale)
        x = nn.functional.interpolate(x, size=(tgt_time_sp, w, h))
        if (tgt_time_sp > t):
            x = x[:, :, :t, :, :]
        elif (tgt_time_sp < t):
            paddings = torch.zeros(x.shape[0], x.shape[1], t - tgt_time_sp, w, h, device=x.device)
            x = torch.cat((x, paddings), dim=2)

        return x


class TimeScaling(nn.Module):
    def __init__(self, num_conds, tgt_sample):
        super(TimeScaling, self).__init__()

        self.scale = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        self.num_experts = len(self.scale)
        self.tgt_sample = tgt_sample

        self.proj = nn.Sequential(
            nn.Linear(num_conds, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_experts),
            nn.Softmax(dim=1)
        )

        self.expert = []
        for scale in self.scale:
            self.expert.append(
                ExpertClip(scale, tgt_sample)
            )

    def forward(self, x, conds):
        bs, ch, _, _, _ = x.shape
        res = torch.zeros(self.num_experts, bs, ch, self.tgt_sample[0], self.tgt_sample[1], self.tgt_sample[2], device=x.device)
        for i in range(self.num_experts):
            res[i] = self.expert[i](x)

        expert_weight = self.proj(conds)
        res = res.permute(1, 0, 2, 3, 4, 5)
        res = expert_weight.reshape(bs, self.num_experts, 1, 1, 1, 1) * res
        return res.sum(dim=1), expert_weight


class Conv3dDec(nn.Module):
    def __init__(self, num_conds, final_dim, cond_proj_dim, num_pre_conv, num_rescale, rescale, hidden_channels, args, final_interpolate, act=nn.PReLU()):
        super(Conv3dDec, self).__init__()

        self.num_conds = num_conds
        self.init_feat = final_dim + cond_proj_dim

        self.proj = nn.Linear(self.num_conds, cond_proj_dim)

        self.num_pre_conv = num_pre_conv
        self.pre_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(self.init_feat, self.init_feat, 3, 1, 1),
                act,
            ) for _ in range(num_pre_conv)
        )
        self.num_rescale = num_rescale
        self.rescales = nn.ModuleList(
            TimeScaling(self.num_conds, rescale[i]) for i in range(num_rescale)
        )
        self.up_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(i, o, *arg),
                act,
                nn.Conv3d(o, o, 1, 1, 0),
                act,
            ) for i, o, arg in zip([self.init_feat] + hidden_channels[:-1], hidden_channels, [args] * len(hidden_channels))
        )
        self.final_interpolate = final_interpolate
        self.final_conv = nn.Conv3d(hidden_channels[-1], 1, *args)

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, cond):
        """
        Args:
            x: input batch (Tensor) [bs, in_channel, t, h, w]
            cond: condition (Tensor) [bs, num_conds]
        Return: (Tensor) [bs, t, h, w], weight
        """
        cond2 = self.proj(cond)
        bs, _, T, H, W = x.shape
        cond2 = cond2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, T, H, W)

        x = torch.cat((x, cond2), dim=1)

        for i in range(self.num_pre_conv):
            x = self.pre_conv[i](x)

        w = []
        for i in range(self.num_rescale):
            x, cur_w = self.rescales[i](x, cond)
            w.append(cur_w)
            if i == self.num_rescale - 1:
                x = nn.functional.interpolate(x, size=self.final_interpolate)
            x = self.up_conv[i * 2](x)
            x = self.up_conv[i * 2 + 1](x)

        x = self.final_conv(x)
        return x.squeeze(), w
