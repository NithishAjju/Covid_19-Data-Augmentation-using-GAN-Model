import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_GAN3 import netG, netD


class Client:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_dir = 'CovidDataset'

        self.batch_size = 16
        self.local_epochs = 2
        self.imageSize = 64
        self.nc = 1  # Grayscale images
        self.nz = 100  # Latent vector size
        self.ngf = 64
        self.ndf = 64
        self.nb_label = 2  # Expected labels: [0, 1]

        # Loss functions
        self.s_criterion = nn.BCELoss().to(self.device)
        self.c_criterion = nn.NLLLoss().to(self.device)

        # Initialize noise and labels
        self.noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1).to(self.device)

        # Data transformations
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Filter the dataset to include only valid labels
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        valid_indices = [i for i, (_, label) in enumerate(dataset) if label < self.nb_label]
        filtered_dataset = torch.utils.data.Subset(dataset, valid_indices)

        self.train_loader = torch.utils.data.DataLoader(
            filtered_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True  # Ensure consistent batch size
        )

        # Models and optimizers
        self.generator = netG(self.nz, self.ngf, self.nc).to(self.device)
        self.discriminator = netD(self.ndf, self.nc, self.nb_label).to(self.device)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def local_training(self):
        self.discriminator.train()
        self.generator.train()

        for epoch in range(self.local_epochs):
            for i, (img, label) in enumerate(self.train_loader):
                img, label = img.to(self.device), label.to(self.device)

                # Update D network
                self.discriminator.zero_grad()
                s_label = torch.full((img.size(0),), 1.0, dtype=torch.float, device=self.device)  # Real labels
                c_label = label

                # Train on real data
                s_output, c_output = self.discriminator(img)
                s_errD_real = self.s_criterion(s_output, s_label.view(-1, 1))
                c_errD_real = self.c_criterion(c_output, c_label)
                errD_real = s_errD_real + c_errD_real
                errD_real.backward()

                # Train on fake data
                self.noise.normal_(0, 1)
                fake = self.generator(self.noise)
                s_label.fill_(0.0)  # Fake labels
                s_output, c_output = self.discriminator(fake.detach())
                s_errD_fake = self.s_criterion(s_output, s_label.view(-1, 1))
                c_errD_fake = self.c_criterion(c_output, c_label)
                errD_fake = s_errD_fake + c_errD_fake
                errD_fake.backward()
                self.optimizerD.step()

                # Update G network
                self.generator.zero_grad()
                s_label.fill_(1.0)  # Real labels for generator's perspective
                s_output, c_output = self.discriminator(fake)
                s_errG = self.s_criterion(s_output, s_label.view(-1, 1))
                c_errG = self.c_criterion(c_output, c_label)
                errG = s_errG + c_errG
                errG.backward()
                self.optimizerG.step()

                print(f"Epoch [{epoch+1}/{self.local_epochs}], Batch [{i+1}/{len(self.train_loader)}], "
                      f"Loss_D: {errD_real.item() + errD_fake.item():.4f}, Loss_G: {errG.item():.4f}")

        return self.discriminator.state_dict(), self.generator.state_dict(), errD_real.item(), errG.item()
    def client_update(self, global_weights_d, global_weights_g):
        """Update the local model weights with the global weights."""
        self.discriminator.load_state_dict(global_weights_d)
        self.generator.load_state_dict(global_weights_g)
