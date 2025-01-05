import matplotlib.pyplot as plt

def read_losses(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [float(line.strip()) for line in lines]

# Chemins vers les fichiers
train_ae_loss_file = "train_sexe/vect_train_ae_loss.txt"
train_disc_loss_file = "train_sexe/vect_train_disc_loss.txt"
val_ae_loss_file = "train_sexe/vect_val_ae_loss.txt"
val_disc_loss_file = "train_sexe/vect_val_disc_loss.txt"

# Lecture des pertes
train_ae_loss = read_losses(train_ae_loss_file)
train_disc_loss = read_losses(train_disc_loss_file)
val_ae_loss = read_losses(val_ae_loss_file)
val_disc_loss = read_losses(val_disc_loss_file)

# Tracer les courbes
plt.figure(figsize=(10, 6))
#plt.plot(train_ae_loss, label='Train AE Loss', linestyle='-', marker='o')
plt.plot(train_disc_loss, label='Train Disc Loss', linestyle='-', marker='o')
#plt.plot(val_ae_loss, label='Validation AE Loss', linestyle='--', marker='s')
plt.plot(val_disc_loss, label='Validation Disc Loss', linestyle='--', marker='s')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Courbes de Loss - Entraînement et Validation')
plt.legend()
plt.grid(True)
plt.show()

