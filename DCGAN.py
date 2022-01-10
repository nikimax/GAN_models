# DCGAN это GAN, в архитектуре Генератора и Дискриминатора которой используется CNN, а не полносвязная нейронка
# 1. Do not use Conv layers
# 2. No Linear layers
# 3. Relu for all layers in generator except output (THAN in the end)
# 4. Leaky Relu for all in discriminator

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N * channels_img * 64 * 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            # img size after conv = (w - f + 2p)/s + 1 = (64 - 4 + 2)/2 + 1 = 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            # img size = (32 - 4 + 2)/2 + 1 = 16
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            # img size = (16 - 4 + 2)/2 + 1 = 8
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            # img size = (8 - 4 + 2)/2 + 1 = 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # N * z_dim * 1 * 1
            self._block(z_dim, features_g * 16, 4, 1, 0),  # N * f_g * 16 * 4 * 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 2e-4
    noise_dim = 100
    batch_size = 128
    image_size = 64
    num_epochs = 5
    features_disc = 64
    features_gen = 64
    img_channels = 1

    transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])
        ])

    dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gen = Generator(z_dim=noise_dim, channels_img=img_channels, features_g=features_gen).to(device)
    disc = Discriminator(channels_img=img_channels, features_d=features_disc).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, noise_dim, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)

            # Train Discriminator
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_D_real = criterion(disc_real, torch.ones_like(disc_real))  # max log(D(real))
            loss_D_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # max log(1 - D(G(z)))
            lossD = (loss_D_real + loss_D_fake)/2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()
            # Train Generator
            output = disc(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1



