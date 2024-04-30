import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from segment_anything_local import sam_model_registry

# Define the paths to the crack images and their labels
data_path = "SemanticSegmentationScars"
image_path = data_path + "/ImageDatastore"
label_path = data_path + "/PixelLabelDatastore"

# Load the pre-trained model checkpoint file
sam_checkpoint = 'sam_vit_b_01ec64.pth'
# Set the device to use for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_model = torch.load(sam_checkpoint)

model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Remove the last classification layer of the pre-trained model to get feature vectors
#new_model = nn.Sequential(*list(sam_model.children())[:-1])
new_model = nn.Sequential(*list(sam.children())[:-1])

# Define the transformations to apply to the images
transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])

# Load the wall crack images and their labels
dataset = ImageFolder(image_path, transform=transform)
labels = ImageFolder(label_path, transform=transform)

# Create the data loader for the wall crack images and their labels
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(new_model.parameters(), lr=0.001)

# Train the new model on the wall crack images and their labels
for epoch in range(10):
    running_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        # Move the images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        
        
        #Make boxes and masks
        #boxes, masks = sam(images)
        
        outputs = new_model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
