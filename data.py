from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader

# Set up the transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the CelebA dataset
celeba_dataset = CelebA(root='./data', split='train', transform=transform, download=True)

# Use DataLoader to load the dataset in batches
data_loader = DataLoader(celeba_dataset, batch_size=4, shuffle=True)

# Fetch one batch of data
data_iter = iter(data_loader)
images, labels = data_iter.next()

print("Images shape:", images.shape)  # Expecting: (batch_size, channels, height, width)
print("Labels shape:", labels.shape)  # This will depend on the target_type in CelebA

# Example: printing individual image and label shapes
for i in range(len(images)):
    print(f"Image {i+1} shape:", images[i].shape)
    print(f"Label {i+1}:", labels[i])

#TYPE 1 




# TYPE 2 




#TYPE 3 





#GENERAL