# import pacakges
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import medmnist
from medmnist import INFO, Evaluator
from itertools import product

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

# Calculate mean and std for the training dataset
mean128 = 0.0
std128 = 0.0

min_pixel = float('inf')
max_pixel = float('-inf')

for images, _ in train_loader:
    batch_samples = images.size(0)  # batch size (the last batch can have fewer samples)
    images = images.view(batch_samples, images.size(1), -1)  # reshape to [batch_size, channels, pixels]
    mean128 += images.mean(2).sum(0)
    std128 += images.std(2).sum(0)

    # Update min and max pixel values
    min_pixel = min(min_pixel, images.min().item())
    max_pixel = max(max_pixel, images.max().item())

mean128 /= len(train_loader.dataset)
std128 /= len(train_loader.dataset)

# print(f"Calculated mean: {mean128}")
# print(f"Calculated std: {std128}")
# print(f"Min Pixel: {min_pixel}")
# print(f"Max Pixel: {max_pixel}")

# Update the transformation with calculated mean and std
data_transform128 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean128.tolist(), std=std128.tolist())
])

# Reload the datasets with the updated transformation
train_dataset128 = DataClass(split='train', transform=data_transform128, download=download, size=size)
val_dataset128 = DataClass(split='val', transform=data_transform128, download=download, size=size)

# Create DataLoaders
train_loader128 = DataLoader(dataset=train_dataset128, batch_size=BATCH_SIZE, shuffle=True)
val_loader128 = DataLoader(dataset=val_dataset128, batch_size=2*BATCH_SIZE, shuffle=False)

# modified CNN class
class ModifiedCNN(nn.Module): 
    def __init__(self, in_channels, num_classes, conv1_channels=128, conv2_channels=256, 
                 fc_neurons=128, pool_kernel=2, pool_stride=2):
        super(ModifiedCNN, self).__init__()

        # Layer 1: Convolution + BatchNorm + ReLU + Pooling
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )

        # Layer 2: Convolution + BatchNorm + ReLU + Pooling
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(1, 1)  # Dummy initialization for flexibility
        self.fc_neurons = fc_neurons
        self.fc2 = nn.Linear(fc_neurons, num_classes)

        # Apply Xavier initialization
        self.apply(self._initialize_weights)

    def _initialize_fc(self, x):
        """Dynamically initialize the fully connected layer based on the input dimensions."""
        flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(flattened_size, self.fc_neurons).to(x.device)
        self._initialize_weights(self.fc1)

    def forward(self, x):
        # Convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Flatten
        if isinstance(self.fc1, nn.Linear) and self.fc1.in_features == 1:
            self._initialize_fc(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

                
# Tracking the best model and parameters
global_best_val_loss = float('inf')
global_best_params = None
global_best_model_state = None

# Define the parameter grid
param_grid = {
    'conv1_channels': [64, 128],
    'conv2_channels': [128, 256],
    'fc_neurons': [256, 512],
    'pool_kernel': [2, 3],
    'pool_stride': [2, 3]
}

# Get all combinations of parameters
param_combinations = list(product(*param_grid.values()))

# Training loop for grid search
for params in param_combinations:
    # Unpack parameter values
    conv1_channels, conv2_channels, fc_neurons, pool_kernel, pool_stride = params
    print(f'Training with params: conv1_channels={conv1_channels}, conv2_channels={conv2_channels}, fc_neurons={fc_neurons}, pool_kernel={pool_kernel}, pool_stride={pool_stride}')
    print('=================================================')

    # Initialize model with the current parameters
    modelCNN = ModifiedCNN(
        in_channels=n_channels,
        num_classes=n_classes,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        fc_neurons=fc_neurons,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride
    )

    # Define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(modelCNN.parameters(), lr=0.001)

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda")
    modelCNN.to(device)  # Move model to device

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
        # global_best_model_state = best_model_state  # Save the model state with the lowest validation loss
        global_best_model_state = {k: v.cpu() for k, v in best_model_state.items()}    
    
    del modelCNN
    torch.cuda.empty_cache()
    print('=================================================')

# Save the best model and parameters
torch.save(global_best_model_state, 'best_model.pth')
print(f'Best Model Parameters: {global_best_params} with Validation Loss: {global_best_val_loss:.4f}')
