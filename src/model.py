import torch.nn as nn
import torch


class VariationalAutoEncoder(nn.Module):
    def __init__(self, features_d):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # input: N x channels_img x 224 x 224
            self._enc_block(
                in_channels=3,
                out_channels=features_d,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            self._enc_block(
                in_channels=features_d,
                out_channels=features_d * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            self._enc_block(
                in_channels=features_d * 2,
                out_channels=features_d * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.conv_mu = nn.Sequential(
            nn.Conv2d(
                features_d * 4, features_d * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(features_d * 4),
            nn.ReLU(),
            nn.Conv2d(
                features_d * 4, features_d * 4, kernel_size=3, stride=1, padding=1
            ),  # Output a single channel for mu
        )

        self.conv_logvar = nn.Sequential(
            nn.Conv2d(
                features_d * 4, features_d * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(features_d * 4),
            nn.ReLU(),
            nn.Conv2d(
                features_d * 4, features_d * 4, kernel_size=3, stride=1, padding=1
            ),  # Output a single channel for logvar
        )

        self.decoder = nn.Sequential(
            self._dec_block(
                in_channels=features_d * 4,
                out_channels=features_d * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            self._dec_block(
                in_channels=features_d * 2,
                out_channels=features_d,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(
                features_d,
                3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def _enc_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _dec_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, imgs):
        x = self.encoder(imgs)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_imgs = self.decoder(z)
        return reconstructed_imgs


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def get_model(features):
    model = VariationalAutoEncoder(features)

    return model


def get_essentials(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.9))

    criterion = nn.MSELoss()

    return criterion, optimizer
