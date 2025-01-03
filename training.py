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

# Définition des optimiseurs
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.5, 0.999))
optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=2e-3, betas=(0.5, 0.999))

#Définition des loss
discriminator_loss = nn.BCELoss()
reconstruction_loss = reconstruction_loss = nn.MSELoss() 

#Définition du nombre d'épochs d'entrainement
num_epochs = 500

#Initialisation du paramètre d'entrainement adversariale lambda_e
lambda_e = 0

#Valeur finale de lambda_e
l = 0.0001

#Valeur finale pour lambda_e
s = 500000

#Initialisation du nombre d'itération de l'entrainement
nb_iteration = 0

#Définition du paramètre à modifier
y = 20
# Détection du GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator.to(device)
autoencoder.to(device)

#vecteur calculs loss
vect_train_ae_loss = []
vect_train_disc_loss = []
vect_val_ae_loss = []
vect_val_disc_loss = []


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    #Entrainement
    disc_train_loss = 0.0 
    ae_train_loss = 0.0 
    num_batches = len(dataloader_train)
    discriminator.train()
    autoencoder.train()
    for images, attributs in tqdm(dataloader_train, desc=f"Training Epoch {epoch + 1}"):
        attributs = attributs[:, y:y+1]
        images, attributs = images.to(device), attributs.to(device)
        #Définition du paramètre d'entrainement adversariale lambda_e
        if nb_iteration < s:
            lambda_e += l*nb_iteration/s
        else :
            lambda_e = l
        ae_loss = train_adversarial(autoencoder, discriminator, images, attributs, reconstruction_loss, discriminator_loss, optimizer_autoencoder, lambda_e)
        disc_loss = train_disc(discriminator, autoencoder, images, attributs, discriminator_loss, optimizer_discriminator)
        nb_iteration += 1
        ae_train_loss += ae_loss.item()
        disc_train_loss += disc_loss.item()

    # Phase de validation
    autoencoder.eval()
    discriminator.eval()
    disc_val_loss = 0.0 
    ae_val_loss = 0.0
    with torch.no_grad():
        for images, attributs in tqdm(dataloader_val, desc=f"Training Epoch {epoch + 1}"):
            attributs = attributs[:, y:y+1]
            images, attributs = images.to(device), attributs.to(device)
            latent_images = autoencoder.encoder(images)
            pred_attributs = discriminator(latent_images)
            disc_loss = discriminator_loss(pred_attributs, attributs)
            disc_val_loss += disc_loss.item()

            recon_images = autoencoder.decoder(latent_images, attributs)
            recon_loss = reconstruction_loss(images, recon_images)
            ae_val_loss += recon_loss.item()
    with open('vect_train_ae_loss.txt', 'w') as f:
        for item in vect_train_ae_loss:
            f.write("%s\n" % item)

    with open('vect_train_disc_loss.txt', 'w') as f:
        for item in vect_train_disc_loss:
            f.write("%s\n" % item)

    with open('vect_val_ae_loss.txt', 'w') as f:
        for item in vect_val_ae_loss:
            f.write("%s\n" % item)

    with open('vect_val_disc_loss.txt', 'w') as f:
        for item in vect_val_disc_loss:
            f.write("%s\n" % item)

    vect_train_ae_loss.append(ae_train_loss / len(dataloader_train))
    vect_train_disc_loss.append(disc_train_loss / len(dataloader_train))
    vect_val_ae_loss.append(ae_val_loss / len(dataloader_val))
    vect_val_disc_loss.append(disc_val_loss / len(dataloader_val))

    torch.save(discriminator.state_dict(), "discriminator.pth")
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    #torch.save(optimizer_autoencoder.state_dict(), "opt_autoencoder.pth")
    #torch.save(optimizer_discriminator.state_dict(), "opt_discriminator.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss Disc: {disc_train_loss / len(dataloader_train):.4f}, "
          f"Epoch [{epoch+1}/{num_epochs}], Train Loss Autoencoder: {ae_train_loss / len(dataloader_train):.4f}, "
          f"Epoch [{epoch+1}/{num_epochs}], Validation Loss Autoencoder: {ae_val_loss / len(dataloader_val):.4f}, "
          f"Epoch [{epoch+1}/{num_epochs}], Validation Loss Discriminator: {disc_val_loss / len(dataloader_val):.4f}, ")