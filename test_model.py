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

from model import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Charger le modèle entraîné
autoencoder = AutoEncoder().to(device)
autoencoder.load_state_dict(torch.load("training_auto.pth", map_location=device))
autoencoder.eval()

# 2) Préparer l'image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img_path = "./datas/img_align_celeba/img_align_celeba/000001.jpg"

pil_img = Image.open(img_path).convert("RGB")
plt.imshow(pil_img)
plt.axis('off')
plt.show()
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# 3) Attributs initiaux (ex: 40 attributs CelebA)
original_attrs = pd.read_csv("datas/list_attr_celeba.csv", sep=",", header=0, index_col="image_id").replace(-1,0)
original_attrs = torch.from_numpy(original_attrs.loc['000001.jpg'].values.astype("float32"))

original_attrs = original_attrs.unsqueeze(0)
print(original_attrs)
modified_attrs = original_attrs.clone()
modified_attrs[0][7] = 1
modified_attrs = modified_attrs.to(device)
print(modified_attrs)

# 5) Encoder + Décoder
with torch.no_grad():
    z = autoencoder.encoder(input_tensor)
    output_tensor = autoencoder.decoder(z, modified_attrs)

# 6) Conversion et visualisation
output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0)
plt.imshow(output_image.numpy())
plt.axis('off')
plt.show()
