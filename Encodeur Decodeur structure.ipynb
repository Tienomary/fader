import torch
import torch.nn as nn
import torch.nn.functional as F

# Définition de l'encodeur
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Couches Convolution-BatchNorm-LeakyReLU avec k filtres pour chaque couche
        # Les convolutions ont une taille de noyau 4x4, un pas (stride) de 2 et un padding de 1
        # Chaque couche divise la taille de l'entrée par 2
        
        # Première couche avec 16 filtres
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        
        # Couche suivante avec 32 filtres
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        # Couche suivante avec 64 filtres
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # Couche suivante avec 128 filtres
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Couche suivante avec 256 filtres
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Couche suivante avec 512 filtres
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Dernière couche avec 512 filtres
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Passage de l'entrée à travers chaque couche de l'encodeur
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


# Définition du décodeur
class Decoder(nn.Module):
    def __init__(self, num_attributes):
        super(Decoder, self).__init__()
        
        # Ajout des attributs comme canaux constants pour chaque couche du décodeur
        self.num_attributes = num_attributes
        
        # Couche transposée pour reconstruire avec 512+2*num_attributes filtres
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512 + 2 * num_attributes, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512 + 2 * num_attributes, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(512 + 2 * num_attributes, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256 + 2 * num_attributes, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128 + 2 * num_attributes, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(64 + 2 * num_attributes, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(32 + 2 * num_attributes, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Utilisation de Tanh pour une sortie entre -1 et 1 (image normalisée)
        )

    def forward(self, x, attributes):
        # Combinaison du code latent avec les attributs (sous forme de canaux constants)
        # pour chaque couche du décodeur
        attributes = attributes.view(attributes.size(0), -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        
        x = torch.cat([x, attributes], 1)
        x = self.layer1(x)
        
        x = torch.cat([x, attributes], 1)
        x = self.layer2(x)
        
        x = torch.cat([x, attributes], 1)
        x = self.layer3(x)
        
        x = torch.cat([x, attributes], 1)
        x = self.layer4(x)
        
        x = torch.cat([x, attributes], 1)
        x = self.layer5(x)
        
        x = torch.cat([x, attributes], 1)
        x = self.layer6(x)
        
        x = torch.cat([x, attributes], 1)
        x = self.layer7(x)
        
        return x


# Définition du discriminateur
class Discriminator(nn.Module):
    def __init__(self, num_attributes):
        super(Discriminator, self).__init__()
        
        # Couche de convolution initiale
        self.conv = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Réseau fully-connected de deux couches (512 et nombre d'attributs)
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_attributes)

    def forward(self, x):
        # Passage à travers la couche de convolution et aplatissement
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        
        # Passage à travers les couches fully-connected
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        
        return x
