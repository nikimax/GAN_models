# Wassershein GAN (we use structure of layers from DCGAN as base)
# We change loss function for GAN model to make training process more stable. In classical GAN loss function
# is not representative.
# New LOSS: (|f| <= 1): E(real~pr)[f(real)] - E(fake~pg)[f(fake)]
# pr - real prob distribution, pg - generated distribution, f() - discriminator
# So discriminator max LOSS (it want to distinguish real and fake)
# Generator min LOSS as it want to minimise difference between fake and real
# Our main target is to make Pg distribution similar to Pr distribution
# We also Clip all parameters so that they are in range [-0.01, 0.01]

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from DCGAN import Generator, initialize_weights


class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
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
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0))

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.disc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITER = 5  # in WGAN for one Generator iteration we have CRITIC_ITER iterations for Discriminator
WEIGHT_CLIP = 0.01  # range of normalising parameters [-weight_clip; weight_clip]

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ])

# If you train on MNIST, remember to set channels_img to 1
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,
                       download=True)

# comment mnist above and uncomment below if train on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
"We do not use BCE loss as ve do not need log function any more. Just use torch.mean instead"

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        # we make additional for loop because we have different num of iterations for Gen and Disc (1 gen = 5 desc)
        ### Train Discriminator (Critic)
        for _ in range(CRITIC_ITER):
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            # we use minus to swap from max to min
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_disc.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        ### Train Generator
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
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


