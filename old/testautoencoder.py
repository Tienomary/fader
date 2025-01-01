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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder()
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

            
image_path = "datas/img_align_celeba/img_align_celeba/000001.jpg"
image = Image.open(image_path).convert("RGB")  


# afficher une image et ses attributs
def affiche(dataset, image, attributs):
    attribute_names = dataset.get_attribute_names()

    # Afficher l'image
    image = image.permute(1, 2, 0).numpy()  
    plt.imshow(image)
    plt.axis('off')  
    plt.show()

    # Afficher les attributs associés à l'image
    print(f"Attributs pour cette image :")
    for i, attr in enumerate(attributs):
        print(f"{attribute_names[i]} : {'Oui' if attr.item() == 1 else 'Non'}")

for images, attrs in dataloader:
    print(images.shape) 
    print(attrs.shape)
    image = images[0] 
    attributs = attrs[0]
    affiche(dataset, image, attributs)
    output = model(images, attrs)
    print(output.shape)
    pred_images = output
    # Après avoir obtenu `pred_images` et `images` :
    # Sélectionner une image (par exemple, la première du batch)
    generated_image = pred_images[0].detach().cpu()  # Supprime le gradient et déplace sur CPU
    original_image = images[0].detach().cpu()

    # Si les images sont en format [C, H, W], les permuter pour [H, W, C]
    if generated_image.dim() == 3 and generated_image.shape[0] in [1, 3]:  # Vérifie les canaux
        generated_image = generated_image.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        original_image = original_image.permute(1, 2, 0)

    # Si l'image est normalisée (valeurs entre -1 et 1), la ramener entre 0 et 1
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # Afficher les images côte à côte
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.numpy())
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_image.numpy())
    plt.title("Generated Image")
    plt.axis("off")

    plt.show()
    break
