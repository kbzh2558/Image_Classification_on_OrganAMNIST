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

BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# Temporary transformation to load data as tensors without normalization
temp_transform = transforms.Compose([transforms.ToTensor()])

# Load the dataset with temporary transformation
train_dataset = DataClass(split='train', transform=temp_transform, download=download)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate mean and std for the training dataset
mean = 0.0
std = 0.0

min_pixel = float('inf')
max_pixel = float('-inf')

for images, _ in train_loader:
    batch_samples = images.size(0)  # batch size (the last batch can have fewer samples)
    images = images.view(batch_samples, images.size(1), -1)  # reshape to [batch_size, channels, pixels]
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

    # Update min and max pixel values
    min_pixel = min(min_pixel, images.min().item())
    max_pixel = max(max_pixel, images.max().item())

mean /= len(train_loader.dataset)
std /= len(train_loader.dataset)

# print(f"Calculated mean: {mean}")
# print(f"Calculated std: {std}")
# print(f"Min Pixel: {min_pixel}")
# print(f"Max Pixel: {max_pixel}")

# Update the transformation with calculated mean and std
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

# Reload the datasets with the updated transformation
train_dataset = DataClass(split='train', transform=data_transform, download=download)
# test_dataset = DataClass(split='test', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# define class
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()

        # Layer 1: Convolution + BatchNorm + ReLU
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Layer 2: Convolution + BatchNorm + ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Placeholder for fully connected layers; dimensions will be determined later
        self.fc1 = nn.Linear(1, 1)  # Dummy initialization; will be overwritten in `_initialize_fc`

        # Output layer
        self.fc2 = nn.Linear(256, num_classes)

        # Apply Xavier initialization to layers
        self.apply(self._initialize_weights)

    def _initialize_fc(self, x):
        """Initialize the fully connected layer based on the input dimensions."""
        flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 256).to(x.device)  # Move to the same device as x
        self._initialize_weights(self.fc1)  # Initialize weights

    def forward(self, x):
        # Pass through convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Flatten the output and dynamically initialize fc1 if necessary
        if isinstance(self.fc1, nn.Linear) and self.fc1.in_features == 1:
            self._initialize_fc(x)
        x = x.view(x.size(0), -1)  # Flatten before fully connected layer

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Apply Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Initialize biases to zero



# Define early stopping parameters
# patience = 3  # Number of epochs to wait for improvement

param_grid = {
    'patience': [3, 5, 7],
    'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.05]
}

param_combinations = list(product(*param_grid.values()))

# Tracking the best model and parameters
global_best_val_loss = float('inf')
global_best_params = None
global_best_model_state = None

for params in param_combinations:
    patience, learning_rate = params

    # Instantiate the model
    modelCNN = SimpleCNN(in_channels=n_channels, num_classes=n_classes)

    # Define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda")
    modelCNN.to(device)  # Move model to device


    optimizer = optim.SGD(modelCNN.parameters(), lr=learning_rate, momentum=0.9)

    print(f'Training with params: patience={patience}, learning_rate={learning_rate}')
    print('=================================================')

    best_val_loss = float('inf')  # Initialize best validation loss
    epochs_no_improve = 0  # Counter for early stopping

    for epoch in range(70):
        # Training phase
        modelCNN.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = modelCNN(inputs)
            
            # Calculate loss
            if task == 'multi-label, binary-class':
                targets = targets.float()
                loss = criterion(outputs, targets)
            else:
                targets = targets.long()
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate training loss
            train_loss += loss.item()
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        # print(f'Epoch [{epoch+1}/70], Training Loss: {avg_train_loss:.4f}')
        print(f'Config {params} - Epoch [{epoch+1}/70], Training Loss: {avg_train_loss:.4f}')

        
        # Validation phase (for early stopping)
        modelCNN.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
                
                # Forward pass
                outputs = modelCNN(inputs)
                
                # Calculate loss
                if task == 'multi-label, binary-class':
                    targets = targets.float()
                    loss = criterion(outputs, targets)
                else:
                    targets = targets.long()
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
                
                # Accumulate validation loss
                val_loss += loss.item()
        
        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        # print(f'Epoch [{epoch+1}/70], Validation Loss: {avg_val_loss:.4f}')
        print(f'Config {params} - Epoch [{epoch+1}/70], Validation Loss: {avg_val_loss:.4f}')

        
        # Check early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset early stopping counter
            # Optionally save the model
            # torch.save(modelCNN.state_dict(), 'simpleCNN_best_model.pth')
            best_model_state = modelCNN.state_dict()
            # print(f'Validation loss improved; model saved.')

        else:
            epochs_no_improve += 1
            print(f'No improvement in validation loss for {epochs_no_improve} epochs.')
            # logger.info(f'No improvement in validation loss for {epochs_no_improve} epochs.')


        # If no improvement for `patience` epochs, stop training
        if epochs_no_improve >= patience:
            # print(f'Early stopping triggered after {epoch+1} epochs.')
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