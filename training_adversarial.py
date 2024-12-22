
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

# Initialisation du modèle et de l'optimiseur
discriminator = Discriminator()
autoencoder = AutoEncoder()
encoder = Encoder()

# Définition des optimisateurs et des pertes
optimizer_autoencoder = optim.Adam(discriminator.parameters(), lr=0.001)
reconstruction_loss = nn.MSELoss()  
discriminator_loss = nn.BCELoss()

#Définition du nombre d'épochs d'entrainement
num_epochs = 10

#Initialisation du paramètre d'entrainement adversariale lambda_e
lambda_e = 0

#Initialisation du nombre d'itération de l'entrainement
nb_iteration = 0

#Définition du paramètre à modifier
y = 6

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    discriminator.train()
    epoch_loss = 0  
    num_batches = len(dataloader)
    for images, attributs in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        if nb_iteration >= 500:
            #Définition du paramètre d'entrainement adversariale lambda_e
            lambda_e += 0.0001
        train_adversarial(autoencoder, encoder, discriminator, y, images, attributs, reconstruction_loss, discriminator_loss, optimizer_autoencoder, lambda_e)
        nb_iteration += 1
    epoch_loss /= num_batches
    torch.save(discriminator.state_dict(), "training_adv.pth")
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Discriminator Loss: {epoch_loss:.4f}")



