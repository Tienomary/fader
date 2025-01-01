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
from train_adversarial import *
from train_disc import *


# Définiton des modèles
discriminator = Discriminator()
autoencoder = AutoEncoder()
encoder = Encoder()

# Définition des optimiseurs
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)
optimizer_autoencoder = optim.Adam(discriminator.parameters(), lr=0.001)

#Définition des loss
discriminator_loss = nn.BCELoss()
reconstruction_loss = reconstruction_loss = nn.MSELoss() 

#Définition du nombre d'épochs d'entrainement
num_epochs = 10

#Initialisation du paramètre d'entrainement adversariale lambda_e
lambda_e = 0

#Initialisation du nombre d'itération de l'entrainement
nb_iteration = 0

#Définition du paramètre à modifier
y = 6
# Détection du GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator.to(device)
encoder.to(device)
autoencoder.to(device)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    discriminator.train()
    epoch_loss = 0  
    num_batches = len(dataloader)
    for images, attributs in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        images, attributs = images.to(device), attributs.to(device)
        if nb_iteration >= 500:
            #Définition du paramètre d'entrainement adversariale lambda_e
            lambda_e += 0.0001
        train_adversarial(autoencoder, encoder, discriminator, y, images, attributs, reconstruction_loss, discriminator_loss, optimizer_autoencoder, lambda_e)
        train_disc(discriminator, encoder, images, attributs, discriminator_loss, optimizer_discriminator)
        nb_iteration += 1
    epoch_loss /= num_batches
    torch.save(discriminator.state_dict(), "training.pth")
    torch.save(autoencoder.state_dict(), "training_auto.pth")
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Discriminator Loss: {epoch_loss:.4f}")