# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:45 2020
This is the standalone GAN code for COVID-19 data augmentation, as part of the paper publication: 
    "Federated Learning for COVID-19 Detection with Generative Adversarial Networks in Edge Cloud Computing", 
    IEEE Internet of Things Journal, Nov. 2021, Accepted (https://ieeexplore.ieee.org/abstract/document/9580478)
@author: Dinh C. Nguyen 
"""

import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pandas as pd  # Import pandas
from model_GAN3 import netG, netD

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and Hyperparameters
data_dir = 'CovidDataset'  # Path to your dataset
batch_size = 16
num_epochs = 500
imageSize = 64  # Image size
nc = 1  # Number of channels (for grayscale, 1)
nz = 100  # Size of z latent vector (input to generator)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
lr = 0.001  # Learning rate
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer
beta2 = 0.999  # Beta2 hyperparameter for Adam optimizer

# Fixed label settings for generating images during training
real_label = 1.0
fake_label = 0.0

# Define Loss functions
s_criterion = nn.BCELoss().to(device)  # Binary cross entropy loss
c_criterion = nn.CrossEntropyLoss().to(device)  # Cross entropy loss

# Initialize noise and labels
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)  # Latent vector (16x100x1x1)

# Define transformations for dataset
mu, sigma = (0.5), (0.5)  # Mean and standard deviation for normalization
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((imageSize, imageSize)),
    transforms.ToTensor(),
    transforms.Normalize(mu, sigma)
])

# Dataset loading
train_set = datasets.ImageFolder(data_dir, transform=transform)

# Dynamically determine number of labels
nb_label = len(train_set.classes)  # Automatically set nb_label
print(f"Number of classes in dataset: {nb_label}")
print(f"Classes: {train_set.classes}")

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Initialize models (Generator and Discriminator)
generator = netG(nz, ngf, nc).to(device)
discriminator = netD(ndf, nc, nb_label).to(device)

# Optimizers
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

# Training function
def training():
    for epoch in range(num_epochs + 1):
        for i, (img, label) in enumerate(train_loader):
            ###########################
            # (1) Update Discriminator
            ###########################
            discriminator.zero_grad()
            batch_size = img.size(0)
            input1 = img.to(device)

            # Normalize labels
            c_label = label.to(device)
            c_label -= c_label.min()  # Normalize to [0, nb_label-1]
            assert c_label.max().item() < nb_label, f"Label value out of range: {c_label.max().item()} >= {nb_label}"

            s_label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            s_output, c_output = discriminator(input1)
            s_errD_real = s_criterion(s_output.view(-1), s_label)  # Match shapes for BCELoss
            c_errD_real = c_criterion(c_output, c_label)
            errD_real = s_errD_real + c_errD_real
            errD_real.backward()

            # Train with fake images
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            s_label.fill_(fake_label)  # Fake labels for discriminator
            s_output, c_output = discriminator(fake.detach())
            s_errD_fake = s_criterion(s_output.view(-1), s_label)  # Match shapes for BCELoss
            c_errD_fake = c_criterion(c_output, c_label)
            errD_fake = s_errD_fake + c_errD_fake
            errD_fake.backward()

            optimizerD.step()

            ###########################
            # (2) Update Generator
            ###########################
            generator.zero_grad()
            s_label.fill_(real_label)  # Fake images are real for generator
            s_output, c_output = discriminator(fake)
            s_errG = s_criterion(s_output.view(-1), s_label)  # Match shapes for BCELoss
            c_errG = c_criterion(c_output, c_label)
            errG = s_errG + c_errG
            errG.backward()

            optimizerG.step()

            print(f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}] Loss_D: {errD_real.item() + errD_fake.item():.4f} Loss_G: {errG.item():.4f}")

            # Save images periodically
            if epoch % 50 == 0 and i == 0:
                vutils.save_image(img, f'./0_output_images/real_samples.png', normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.data, f'./0_output_images/fake_samples_epoch_{epoch:03d}.png', normalize=True)

        # Save the models
        torch.save(generator.state_dict(), f'./0_saved_model/netG_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'./0_saved_model/netD_epoch_{epoch}.pth')

        # Logging losses
        saved_training(errD_real.item() + errD_fake.item(), errG.item())

# Function to save training loss graphs
def saved_training(Loss_D1, Loss_G1):
    file = 'training_loss.csv'

    # Check if the file exists and create it if it doesn't
    if not os.path.exists(file):
        df = pd.DataFrame({'Loss_D': [Loss_D1], 'Loss_G': [Loss_G1]})
        df.to_csv(file, mode='w', header=True)
    else:
        df = pd.read_csv(file)
        new_row = pd.DataFrame({'Loss_D': [Loss_D1], 'Loss_G': [Loss_G1]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file, mode='w', header=True, index=False)

    # Now read the CSV file and plot the graph
    df = pd.read_csv(file)
    z1 = df['Loss_D']
    z2 = df['Loss_G']
    plt.plot(z1, label='Loss_D')
    plt.plot(z2, label='Loss_G')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Training Loss')
    plt.legend(bbox_to_anchor=(0.75, 0.95), loc='upper left')
    plt.savefig(f"0_plot/graph_loss_{datetime.datetime.now().strftime('%H_%M_%S')}.png")
    plt.show()

# Run training
training()