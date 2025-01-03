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

#Uniquement la fonction qui entraine le discriminateur sur 1 batch
def train_disc(discriminator, autoencoder, images, attributs, discriminator_loss, optimizer_discriminator) :

    # Préparation des données
#    images, attributs = images.to(device), attributs.to(device)

    with torch.no_grad(): 
        latent_images = autoencoder.encoder(images)
    pred_attributs = discriminator(latent_images)
    disc_loss = discriminator_loss(pred_attributs, attributs)

    optimizer_discriminator.zero_grad()
    disc_loss.backward()
    optimizer_discriminator.step()

    return disc_loss
