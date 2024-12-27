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
optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=0.001)

#Définition des loss
discriminator_loss = nn.BCELoss()
reconstruction_loss = reconstruction_loss = nn.MSELoss() 

#Définition du nombre d'épochs d'entrainement
num_epochs = 10

#Initialisation du paramètre d'entrainement adversariale lambda_e
lambda_e = 0

#Valeur finale de lambda_e
l = 0.0001

#Valeur finale pour lambda_e
s = 500000

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
    
    #Entrainement
    disc_train_loss = 0.0 
    ae_train_loss = 0.0 
    num_batches = len(dataloader_train)
    discriminator.train()
    autoencoder.train()
    for images, attributs in tqdm(dataloader_train, desc=f"Training Epoch {epoch + 1}"):
        images, attributs = images.to(device), attributs.to(device)
        #Définition du paramètre d'entrainement adversariale lambda_e
        if nb_iteration < s:
            lambda_e += l*nb_iteration/s
        else :
            lambda_e = l
        ae_loss = train_adversarial(autoencoder, encoder, discriminator, y, images, attributs, reconstruction_loss, discriminator_loss, optimizer_autoencoder, lambda_e)
        disc_loss = train_disc(discriminator, encoder, images, attributs, discriminator_loss, optimizer_discriminator)
        nb_iteration += 1
        ae_train_loss += ae_loss.item()
        disc_train_loss += disc_loss.item()
    # Phase de validation
 #   autoencoder.eval()
  #  discriminator.eval()
 #   disc_val_loss = 0.0 
   # with torch.no_grad():
    #    for images, attributs in tqdm(dataloader_val, desc=f"Training Epoch {epoch + 1}"):
     #       images, attributs = images.to(device), attributs.to(device)
      #      with torch.no_grad(): 
       #         latent_images = encoder(images)
        #        pred_attributs = discriminator(latent_images)
         #       disc_loss = discriminator_loss(pred_attributs, attributs)
          #      disc_val_loss += disc_loss.item()

    torch.save(discriminator.state_dict(), "discriminator.pth")
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss Disc: {disc_train_loss / len(dataloader_train):.4f}, "
          f"Epoch [{epoch+1}/{num_epochs}], Train Loss Autoencoder: {ae_train_loss / len(dataloader_train):.4f}, ")