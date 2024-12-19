
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Discriminator, AutoEncoder  # Assurez-vous que les modèles sont bien définis ici
from dataset import CelebADataset  # Classe personnalisée pour gérer CelebA

# Initialisation du modèle et de l'optimiseur
discriminator = Discriminator()
autoencoder = AutoEncoder()

# Définition des optimisateurs et des pertes
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)
adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss pour les attributs binaires

# Détection du GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator.to(device)
autoencoder.to(device)

# Chargement des poids de l'autoencodeur déjà entraîné
autoencoder.load_state_dict(torch.load("autoencoder.pth"))
autoencoder.eval()  # On ne modifie pas les paramètres de l'autoencodeur ici

# Chargement des données
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CelebADataset(
    img_dir="datas/img_align_celeba/img_align_celeba/",
    attr_path="datas/list_attr_celeba.csv",
    partition_path="datas/list_eval_partition.csv",
    transform=transform,
    split="train"
)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Entraînement
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    discriminator.train()
    epoch_loss = 0  
    num_batches = len(dataloader)

    for images, attributes in tqdm(dataloader, desc=f"Training Discriminator Epoch {epoch + 1}"):
        # Préparation des données
        images, attributes = images.to(device), attributes.to(device)

        # Générer les représentations latentes avec l'encodeur
        with torch.no_grad():  # Pas de gradient pour l'encodeur
            latent_vectors = autoencoder.encoder(images)

        # Prédire les attributs à partir des représentations latentes
        pred_attributes = discriminator(latent_vectors)

        # Calcul de la perte adversariale
        loss = adversarial_loss(pred_attributes, attributes)

        # Mise à jour des paramètres du discriminateur
        optimizer_discriminator.zero_grad()
        loss.backward()
        optimizer_discriminator.step()

        epoch_loss += loss.item()

    # Afficher la perte moyenne pour l'époque
    epoch_loss /= num_batches
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Adversarial Loss: {epoch_loss:.4f}")

    # Sauvegarde du discriminateur
    torch.save(discriminator.state_dict(), "discriminator.pth")

