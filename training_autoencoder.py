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

torch.autograd.set_detect_anomaly(True)

autoencoder = AutoEncoder()


optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=0.001)


reconstruction_loss = nn.MSELoss()

num_epochs = 10


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#autoencoder.to(device)
#autoencoder.load_state_dict(torch.load("autoencoder.pth"))

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    autoencoder.train()
    epoch_loss = 0  
    num_batches = len(dataloader)
    for images, attributs in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        # Autoencoder
 #       images, attributs = images.to(device), attributs.to(device)
        pred_images = autoencoder(images, attributs)

        recon_loss = reconstruction_loss(pred_images, images)

        optimizer_autoencoder.zero_grad()
        recon_loss.backward()
        optimizer_autoencoder.step()
        epoch_loss += recon_loss.item()

    epoch_loss /= num_batches
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Reconstruction Loss: {epoch_loss:.4f}")

