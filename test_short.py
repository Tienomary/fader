import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model_short import *

# Modèles

ae = AutoEncoder().cuda()        
disc = Discriminator().cuda() 
# Optimizers
opt_ae = optim.Adam(ae.parameters(), lr=2e-3, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=2e-3, betas=(0.5, 0.999))

criterion_bce = torch.nn.BCELoss()   
criterion_recon = torch.nn.MSELoss()

#ae.load_state_dict(torch.load('training_auto_short.pth'))
#disc.load_state_dict(torch.load('training_disc_short.pth'))


lambda_adv = 0 

# Fonctions d'entraînement
def train_discriminator_step(ae, disc, images, attrs):
    """
    entrainer le discriminateur pour qu'il devine attrs depuis z = encoder(images).
    """
    disc.train()
    ae.eval()
    with torch.no_grad():  
        z = ae.encoder(images) 

    # pred du discriminateur
    pred_attrs = disc(z)  
    
    loss_disc = criterion_bce(pred_attrs, attrs)

    opt_disc.zero_grad()
    loss_disc.backward()
    opt_disc.step()
    
    return loss_disc.item()

def train_autoencoder_step(ae, disc, images, attrs, lambda_adv):
    """
    Entraîner l'auto-encodeur (encoder+decoder).
    - disc est en eval (pas de mise à jour du disc)
    - ae est en train
    """
    ae.train()
    disc.eval()
    recon = ae(images, attrs)
    loss_recon = criterion_recon(recon, images)
    z = ae.encoder(images)
    pred_attrs = disc(z)
    loss_adv = criterion_bce(pred_attrs, 1.0 - attrs)
    
    loss_ae =  loss_recon + lambda_adv * loss_adv
    
    # Backprop seulement sur l'auto-encodeur
    opt_ae.zero_grad()
    loss_ae.backward()
    opt_ae.step()
    
    return loss_recon.item(), loss_adv.item()

# Boucle d'entraînement globale
num_epochs = 500

for epoch in range(num_epochs):
    for i, (images, attrs) in enumerate(dataloader):
        images = images.cuda()
        attrs = attrs.cuda()
        
        loss_disc = train_discriminator_step(ae, disc, images, attrs)
        
        loss_rec, loss_adv = train_autoencoder_step(ae, disc, images, attrs, lambda_adv=lambda_adv)
        lambda_adv += 0.0001 / 500
        if i % 50 == 0:
            print(f"Epoch {epoch} / {num_epochs} - Batch {i}/{len(dataloader)}")
            print(f"  Discriminator loss: {loss_disc:.4f}")
            print(f"  AE reconstruction loss: {loss_rec:.4f}, AE adv loss: {loss_adv:.4f}")
    
    torch.save(ae.state_dict(), "training_auto_short.pth")
    torch.save(disc.state_dict(), "training_disc_short.pth")

print("Fin de l'entraînement.")