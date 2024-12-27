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

#entrainement pour 1 batch:
#y est le numéro de l'attribut qui doit être changé
def train_adversarial(autoencoder, encoder, discriminator, y, images, attributs, reconstruction_loss, discriminator_loss, optimizer_autoencoder, lambda_e) :
# Entraînement
    
    # Préparation des données
#    images, attributs = images.to(device), attributs.to(device)

    # Générer les représentations latentes avec l'encodeur
 

    # Prédire les attributs à partir des représentations latentes
    pred_images = autoencoder(images, attributs)
    recon_loss = reconstruction_loss(pred_images, images)
    latent_images = encoder(images)
    with torch.no_grad():  # Pas de gradient pour le discriminateur
        pred_attributs = discriminator(latent_images)
    attributs[y]=1-attributs[y]
    disc_loss = discriminator_loss(pred_attributs, attributs) # On a bien l'attribut qui nous intéresse qui est modifié 
    # Calcul de la perte adversariale
    adversarial_loss = recon_loss+lambda_e*disc_loss

        # Mise à jour des paramètres du discriminateur
    optimizer_autoencoder.zero_grad()
    adversarial_loss.backward()
    optimizer_autoencoder.step()

    return adversarial_loss



