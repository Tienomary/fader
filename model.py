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

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_path, partition_path, transform=None, split="train", limit= None):
        self.img_dir = img_dir
        self.transform = transform

        partition = pd.read_csv(partition_path, sep=",", skiprows=1, header=None, names=["image", "partition"])

        self.images = partition[partition["partition"] == {"train": 0, "valid": 1, "test": 2}[split]]["image"].values
        
        if limit:
            self.images = self.images[:limit]

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

    def get_attribute_names(self):
        # Retourner les noms des attributs en excluant la colonne 'image_id'
        return self.attributes.columns.tolist()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# Créer le DataLoader trop cool
dataset = CelebADataset(
    img_dir="datas/img_align_celeba/img_align_celeba/",
    attr_path="datas/list_attr_celeba.csv",
    partition_path="datas/list_eval_partition.csv",
    transform=transform,
    split="train",
    limit = 32 * 5000
)
print(f"Nombre d'images dans le dataset : {len(dataset)}")

batch_size = 32
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



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
    #affiche(dataset, image, attributs)
    break

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        x = self.model(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=512, num_attributes=40):
        super(Decoder, self).__init__()
        self.num_attributes = num_attributes
        self.latent_dim = 2 * num_attributes
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(512 + self.latent_dim, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512 + self.latent_dim, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256 + self.latent_dim, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128 + self.latent_dim, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64 + self.latent_dim, 32, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(32 + self.latent_dim, 16, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(16 + self.latent_dim, 3, kernel_size=4, stride=2, padding=1)
        

    def forward(self, z, attributs):
        att = attributs
        attributs_transformed = torch.stack([(att == 1).float(), (att == 0).float()], dim=-1)
        latent_code = attributs_transformed.view(att.size(0), -1)
        latent_code = latent_code.unsqueeze(-1).unsqueeze(-1)
        latent_code0 = latent_code.expand(-1, -1, z.shape[2], z.shape[3])

        # Append latent code to the feature maps at every layer.
        z = torch.cat([z, latent_code0], dim=1)
        z = self.relu(self.deconv1(z))

        latent_code1 = latent_code.expand(-1, -1, z.shape[2], z.shape[3])

        z = torch.cat([z, latent_code1], dim=1)
        z = self.relu(self.deconv2(z))
        
        latent_code1 = latent_code.expand(-1, -1, z.shape[2], z.shape[3])

        z = torch.cat([z, latent_code1], dim=1)
        z = self.relu(self.deconv3(z))
        
        latent_code1 = latent_code.expand(-1, -1, z.shape[2], z.shape[3])

        z = torch.cat([z, latent_code1], dim=1)
        z = self.relu(self.deconv4(z))
        
        latent_code1 = latent_code.expand(-1, -1, z.shape[2], z.shape[3])

        z = torch.cat([z, latent_code1], dim=1)
        z = self.relu(self.deconv5(z))
        
        latent_code1 = latent_code.expand(-1, -1, z.shape[2], z.shape[3])

        z = torch.cat([z, latent_code1], dim=1)
        z = self.relu(self.deconv6(z))
        
        latent_code1 = latent_code.expand(-1, -1, z.shape[2], z.shape[3])

        z = torch.cat([z, latent_code1], dim=1)
        z = self.relu(self.deconv7(z)) 
        return z

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()  # Match latent_dim with Encoder's output size

    def forward(self, x, attr):
        z = self.encoder(x)
        output_image = self.decoder(z, attr)
        return output_image
    
class Discriminator(nn.Module):
    def __init__(self, latent_dim=512, num_attributes=40):
        super(Discriminator, self).__init__()
        # c512: une  couche convol
        self.conv = nn.Conv2d(latent_dim, 512, kernel_size=4, stride=2, padding=1) 
        self.fc = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(inplace=False),
            nn.Linear(512, num_attributes),  
            nn.Dropout(0.3),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.conv(z)
        z = z.view(32, -1)
        z = self.fc(z)
        return z

class EncoderAdversarial(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.discriminator = Discriminator()

    def forward(self, x, attr):
        z = self.encoder(x)
        attributs = self.discriminator(z)
        return attributs









