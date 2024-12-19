import torch.nn as nn
from torchvision import transforms, datasets
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *

discriminator = Discriminator()
encoder = Encoder()

optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)

discriminator_loss = nn.BCELoss()

num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator.to(device)
encoder.to(device)
#discriminator.load_state_dict(torch.load("discriminator.pth"))

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    discriminator.train()
    epoch_loss = 0  
    num_batches = len(dataloader)
    for images, attributs in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        images, attributs = images.to(device), attributs.to(device)
        latent_images = encoder(images)
        pred_attributs = discriminator(latent_images)
        disc_loss = discriminator_loss(pred_attributs, attributs)

        optimizer_discriminator.zero_grad()
        disc_loss.backward()
        optimizer_discriminator.step()
        epoch_loss += disc_loss.item()

    epoch_loss /= num_batches
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Discriminator Loss: {epoch_loss:.4f}")