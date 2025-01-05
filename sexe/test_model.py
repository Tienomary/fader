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
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

#CONFIG 
image_name = "000007.jpg"
invert = False #mettre sur True pour inverser le genre de la personne


#choix du device et chargement du modèle 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = AutoEncoder().to(device)
autoencoder.load_state_dict(torch.load("sexe/train_sexe70epoch/autoencoder.pth", map_location=device))
autoencoder.eval()
#transformation de l'image et chargement de celle-ci + affichage
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img_path = "datas/img_align_celeba/img_align_celeba/"+image_name
pil_img = Image.open(img_path).convert("RGB")
plt.imshow(pil_img)
plt.axis('off')
plt.show()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

#chargement des attributs originaux de l'image
original_attrs = pd.read_csv("datas/list_attr_celeba.csv", sep=",", header=0, index_col="image_id").replace(-1,0)
original_attrs = torch.from_numpy(original_attrs.loc[image_name].values.astype("float32"))
original_attrs = original_attrs.unsqueeze(0)
modified_attr = original_attrs.clone()
if invert == True:
    modified_attrs = 1-modified_attr[0][20] #ici on modifie l'attribut numéro 20 qui correspond au genre de la personne (0 pour femme et 1 pour homme) 
else:
    modified_attrs = modified_attr[0][20]

modified_attrs = modified_attrs.unsqueeze(0) #on rajoute une dimension pour correspondre à la taille de l'input du modèle

modified_attrs = modified_attrs.to(device)

print(original_attrs)
print(modified_attrs)

with torch.no_grad():
    z = autoencoder.encoder(input_tensor)
    output_tensor = autoencoder.decoder(z, modified_attrs)

# visualisation
output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0)
plt.imshow(output_image.numpy())
plt.axis('off')
plt.show()
