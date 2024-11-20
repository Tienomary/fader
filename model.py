import torch.nn as nn
from torchvision import transforms, datasets
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_path, partition_path, transform=None, split="train"):
        self.img_dir = img_dir
        self.transform = transform

        partition = pd.read_csv(partition_path, sep=",", skiprows=1, header=None, names=["image", "partition"])

        self.images = partition[partition["partition"] == {"train": 0, "valid": 1, "test": 2}[split]]["image"].values

        self.attributes = pd.read_csv(attr_path, sep=",", header=0, index_col="image_id").replace(-1,0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        attr = self.attributes.loc[img_name].values.astype("float32")
        return image, attr

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# Créer le DataLoader
dataset = CelebADataset(
    img_dir="datas/img_align_celeba/img_align_celeba/",
    attr_path="datas/list_attr_celeba.csv",
    partition_path="datas/list_eval_partition.csv",
    transform=transform,
    split="train"
)
print(f"Nombre d'images dans le dataset : {len(dataset)}")

batch_size = 32
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, attrs in dataloader:
    print(images.shape) 
    print(attrs.shape)  
    break

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.fc = nn.Linear(2048, latent_dim)

    def forward(self, x):
        print('Inti : ', x.shape)
        x = self.model(x)
        print('befo flaaten encoder : ', x.shape)
        x = self.fc(x.view(x.size(0), -1))
        print('sortie encoder : ', x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=512, num_attributes=40):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim + num_attributes, 2048)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, z, attributes):
        z = torch.cat([z, attributes], dim=1)
        print('Entree decoder avec attributs en + : ',z.shape)
        z = self.fc(z)
        print(z.shape)
        z = z.view(z.size(0), 512, 2, 2)
        print(z.shape)
        z = self.model(z)
        print(z.shape)
        return z
    
class Discriminator(nn.Module):
    def __init__(self, latent_dim=512, num_attributes=40):
        super(Discriminator, self).__init__()
        # c512: une  couche convol
        self.conv = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1) 
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512), 
            nn.ReLU(inplace=False),
            nn.Linear(512, num_attributes),  
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

    def forward(self, z):
        print("bef Conv2d:", z.shape)  
        z = self.conv(z)
        print("After Conv2d:", z.shape)  
        z = z.view(z.size(0), -1) 
        z = self.fc(z)
        return z
    



encoder = Encoder()
print(encoder)
optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.001)

decoder = Decoder()
print(decoder)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=0.001)

discriminator = Discriminator()
print(discriminator)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)

reconstruction_loss = nn.MSELoss()
discriminator_loss = nn.BCELoss()

num_epochs=10
for epoch in range(num_epochs):
    for images, attributs in dataloader:
        # Encode
        E = encoder(images)
        # Decode
        R = decoder(E, attributs)

        recon_loss = reconstruction_loss(R, images)

        #ERREUR PAR ICI 

        pred_attributes = discriminator(E.view(32, 512, 1, 1))
        disc_loss = discriminator_loss(pred_attributes, attributs)

        lambda_adv = 0.0001
        encoder_loss = recon_loss - lambda_adv * disc_loss

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        recon_loss.backward(retain_graph=True)

        optimizer_encoder.step()
        optimizer_decoder.step()
        optimizer_discriminator.zero_grad()
        disc_loss.backward()
        optimizer_discriminator.step()

