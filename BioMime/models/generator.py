import torch
import torch.nn as nn
import numpy as np

from BioMime.modules import Conv3dEnc, Conv3dDec


class Generator(nn.Module):

    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.latent_dim = cfg.Latent
        self.num_conds = cfg.num_conds
        enc = cfg.Enc
        self.encoder = Conv3dEnc(enc.num_layers, enc.in_channel, enc.out_channel, enc.hidden_channels, enc.kernel, enc.stride, enc.pad)

        self.out_channel = enc.out_channel
        self.final_dim = enc.final_dim
        self.final_feat = enc.out_channel * np.prod(enc.final_dim)

        self.fc_mu = nn.Linear(self.final_feat, self.latent_dim)
        self.fc_var = nn.Linear(self.final_feat, self.latent_dim)

        self.decode_input = nn.Linear(self.latent_dim, self.final_feat)

        dec = cfg.Dec
        self.decoder = Conv3dDec(self.num_conds, enc.out_channel, dec.cond_proj_dim, dec.num_pre_conv, dec.num_rescale, dec.rescale, dec.up_conv.hidden_channels, dec.up_conv.args, dec.final_interpolate)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        Args:
            input: input batch (Tensor) [bs, 1, t, h, w]
        Return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var, result]

    def decode(self, z, c):
        """
        Decodes the sampled latent feature to MUAP.
        """
        result = self.decode_input(z)
        result, w = self.decoder(result.reshape(-1, self.out_channel, *self.final_dim), c)
        return result, w

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, c):
        mu, log_var, _ = self.encode(input)
        z = self.reparameterize(mu, log_var)
        if self.training:
            return [self.decode(z, c)[0], mu, log_var]
        else:
            return [self.decode(mu, c)[0], mu, log_var]

    def sample(self, num_samples, c, device, *args):
        """
        Sample from the latent space and return the corresponding MUAP space.
        """
        if (len(args) > 0):
            z = args[0]
        else:
            z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        assert(c.shape[0] == num_samples or (num_samples % c.shape[0] == 0))
        if c.shape[0] < num_samples:
            c = c.unsqueeze(1).expand(-1, num_samples // c.shape[0], -1)
        samples, _ = self.decode(z, c.reshape(num_samples, -1))

        return samples

    def generate(self, input, c):
        """
        Given a MUAP, return the morphed MUAP
        """
        return self.forward(input, c)[0]

    def latent(self, input):
        """
        Provide latent features
        """
        feat = self.encoder(input)
        return feat
