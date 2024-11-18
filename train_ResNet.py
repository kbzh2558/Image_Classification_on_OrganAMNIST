import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from itertools import product
import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader

# Config
data_flag = 'organamnist'
download = True

BATCH_SIZE = 64

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Temporary transformation to load data as tensors without normalization
temp_transform = transforms.Compose([transforms.ToTensor()])
size = 128
# Load the dataset with temporary transformation
train_dataset = DataClass(split='train', transform=temp_transform, download=download, size=size)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Update mean and std calculation
mean128 = 0.0
std128 = 0.0
for images, _ in train_loader:
    batch_samples = images.size(0)
    images = images
    mean128 += images.mean([0, 2, 3]) * batch_samples
    std128 += images.std([0, 2, 3], unbiased=False) * batch_samples

mean128 /= len(train_loader.dataset)
std128 /= len(train_loader.dataset)
mean128 = mean128.item()
std128 = std128.item()

# Update the transformation with calculated mean and std
data_transform128 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[mean128] * 3, std=[std128] * 3)
])

# Reload the datasets with the updated transformation
train_dataset128 = DataClass(split='train', transform=data_transform128, download=download, size=size)
val_dataset128 = DataClass(split='val', transform=data_transform128, download=download, size=size)

# Create DataLoaders
train_loader128 = DataLoader(dataset=train_dataset128, batch_size=BATCH_SIZE, shuffle=True)
val_loader128 = DataLoader(dataset=val_dataset128, batch_size=2*BATCH_SIZE, shuffle=False)

# Tracking the best model and parameters
global_best_val_loss = float('inf')
global_best_params = None
global_best_model_state = None

# Define the parameter grid
# this is just the first trial grid
param_grid = {
    'fc_layers_config': [
        [256, 128, 64],
        [512, 256, 128, 64],
        [512, 256, 256, 128],
        [512, 256, 256, 128, 64]
    ]
}

# Get all combinations of parameters
param_combinations = list(product(*param_grid.values()))

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda")

# Load the pre-trained ResNet model
resnet = models.resnet101(pretrained=True)

# Freeze all convolutional layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the avgpool layer with AdaptiveAvgPool2d
resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

# Remove the last fully connected layer
resnet.fc = nn.Identity()

# Define a custom model with new FC layers
class CustomResNet(nn.Module):
    def __init__(self, resnet, num_classes, fc_layers_config):
        super(CustomResNet, self).__init__()
        self.resnet_features = resnet
        self.fc_layers_config = fc_layers_config
        self.num_classes = num_classes

        # Build custom fully connected layers
        fc_layers = []
        in_features = 2048  # Output features from ResNet's avgpool layer
        for neurons in self.fc_layers_config:
            fc_layers.append(nn.Linear(in_features, neurons))
            fc_layers.append(nn.ReLU())
            in_features = neurons
        fc_layers.append(nn.Linear(in_features, self.num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.resnet_features(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

# Training loop for grid search
for params in param_combinations:
    # Unpack parameter values
    fc_layers_config = params[0]
    print(f'Training with FC layers configuration: {fc_layers_config}')
    print('=================================================')

    # Instantiate the custom model
    modelCNN = CustomResNet(resnet, n_classes, fc_layers_config)
    modelCNN.to(device)  # Move model to device

    # Define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Only parameters of final layers are being optimized
    trainable_params = [p for p in modelCNN.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=0.001)

    # Define early stopping parameters
    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training and Validation loop for current configuration
    for epoch in range(70):
        # Training phase
        modelCNN.train()
        train_loss = 0.0
        for inputs, targets in train_loader128:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = modelCNN(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.float()
                loss = criterion(outputs, targets)
            else:
                targets = targets.long()
                targets = targets.view(-1)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader128)
        print(f'Config {params} - Epoch [{epoch+1}/70], Training Loss: {avg_train_loss:.4f}')

        # Validation phase (for early stopping)
        modelCNN.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader128:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = modelCNN(inputs)

                if task == 'multi-label, binary-class':
                    targets = targets.float()
                    loss = criterion(outputs, targets)
                else:
                    targets = targets.long()
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader128)
        print(f'Config {params} - Epoch [{epoch+1}/70], Validation Loss: {avg_val_loss:.4f}')

        # Check if current configuration has the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save model state with best validation loss for this configuration
            best_model_state = modelCNN.state_dict()
        else:
            epochs_no_improve += 1

        # Early stopping for this configuration
        if epochs_no_improve >= patience:
            print(f'Config {params} - Early stopping after {epoch+1} epochs.')
            break

    # Check if this configuration is the best so far
    if best_val_loss < global_best_val_loss:
        global_best_val_loss = best_val_loss
        global_best_params = params
        # Save the model state with the lowest validation loss
        global_best_model_state = {k: v.cpu() for k, v in best_model_state.items()}

    del modelCNN
    torch.cuda.empty_cache()
    print('=================================================')

# Save the best model and parameters
torch.save(global_best_model_state, 'ResNet_best_model.pth')
print(f'Best Model Parameters: {global_best_params} with Validation Loss: {global_best_val_loss:.4f}')




