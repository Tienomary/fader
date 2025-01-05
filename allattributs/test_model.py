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

image_name = "001215.jpg"

#choix du device et chargement du modèle 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = AutoEncoder().to(device)
autoencoder.load_state_dict(torch.load("autoencoder.pth", map_location=device))
autoencoder.eval()
#transformation de l'image et chargement de celle-ci + affichage
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img_path = "./datas/img_align_celeba/img_align_celeba/"+image_name
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

modified_attrs = 1-modified_attr[0][20] #ici on modifie l'attribut numéro 20 qui correspond au genre de la personne (0 pour femme et 1 pour homme) 

modified_attrs = modified_attrs.unsqueeze(0) #on rajoute une dimension pour correspondre à la taille de l'input du modèle

#modified_attrs[0][0] = 1   # 5_o_Clock_Shadow
#modified_attrs[0][1] = 1   # Arched_Eyebrows
#modified_attrs[0][2] = 0   # Attractive
#modified_attrs[0][3] = 1   # Bags_Under_Eyes
#modified_attrs[0][4] = 1   # Bald
#modified_attrs[0][5] = 1   # Bangs
#modified_attrs[0][6] = 1   # Big_Lips
#modified_attrs[0][7] = 1   # Big_Nose
#modified_attrs[0][8] = 0   # Black_Hair
#modified_attrs[0][9] = 1   # Blond_Hair
#modified_attrs[0][10] = 1  # Blurry
#modified_attrs[0][11] = 0  # Brown_Hair
#modified_attrs[0][12] = 1  # Bushy_Eyebrows
#modified_attrs[0][13] = 1  # Chubby
#modified_attrs[0][14] = 1  # Double_Chin
#modified_attrs[0][15] = 1  # Eyeglasses
#modified_attrs[0][16] = 1  # Goatee
#modified_attrs[0][17] = 0  # Gray_Hair
#modified_attrs[0][18] = 1  # Heavy_Makeup
#modified_attrs[0][19] = 1  # High_Cheekbones
#modified_attrs[0][20] = 1  # Male
#modified_attrs[0][21] = 1  # Mouth_Slightly_Open
#modified_attrs[0][22] = 1  # Mustache
#modified_attrs[0][23] = 1  # Narrow_Eyes
#modified_attrs[0][24] = 1  # No_Beard
#modified_attrs[0][25] = 1  # Oval_Face
#modified_attrs[0][26] = 1  # Pale_Skin
#modified_attrs[0][27] = 1  # Pointy_Nose
#modified_attrs[0][28] = 1  # Receding_Hairline
#modified_attrs[0][29] = 1  # Rosy_Cheeks
#modified_attrs[0][30] = 1  # Sideburns
#modified_attrs[0][31] = 0  # Smiling
#modified_attrs[0][32] = 1  # Straight_Hair
#modified_attrs[0][33] = 1  # Wavy_Hair
#modified_attrs[0][34] = 1  # Wearing_Earrings
#modified_attrs[0][35] = 1  # Wearing_Hat
#modified_attrs[0][36] = 1  # Wearing_Lipstick
#modified_attrs[0][37] = 1  # Wearing_Necklace
#modified_attrs[0][38] = 1  # Wearing_Necktie
#modified_attrs[0][39] = 1  # Young

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
