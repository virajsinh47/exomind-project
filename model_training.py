import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Define the 1D-CNN Architecture
class ExoplanetCNN(nn.Module):
    def __init__(self, input_length):
        super(ExoplanetCNN, self).__init__()
        # Input shape: (batch_size, 1, input_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # After two max pools of kernel size 2, length is divided by 4
        self.flatten_length = (input_length // 4) * 32
        
        self.fc1 = nn.Linear(self.flatten_length, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 3) # 3 classes: 0: Planet, 1: Eclipsing Binary, 2: Noise
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Global model instance for inference
# We'll use 20000 to match the requested dummy data size
INPUT_LENGTH = 20000
global_model = ExoplanetCNN(input_length=INPUT_LENGTH)

def predict_exoplanet(flux_array: np.ndarray) -> dict:
    """
    Inference function that accepts a 1D NumPy array and returns prediction details.
    It expects a 1D array of shape (INPUT_LENGTH,).
    """
    global global_model
    global_model.eval()
    
    with torch.no_grad():
        # Convert numpy array to torch tensor, add batch and channel dimensions
        # Shape becomes (1, 1, input_length)
        input_tensor = torch.tensor(flux_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        logits = global_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        
        # Get prediction
        confidence, predicted_class = torch.max(probabilities, dim=0)
        class_idx = predicted_class.item()
        conf_val = confidence.item()
        
        # Map class index to label
        class_mapping = {
            0: "Planet",
            1: "Eclipsing Binary",
            2: "Noise"
        }
        
        return {
            "is_planet": class_idx == 0,
            "confidence": conf_val,
            "class_label": class_mapping[class_idx]
        }

def train_dummy_model():
    """
    Training loop using fake data to demonstrate loss reduction.
    """
    print("Generating fake data using np.random.normal(1, 0.01, 20000)...")
    num_samples = 300
    
    X = []
    y = []
    
    for i in range(num_samples):
        # Base noise
        flux = np.random.normal(1, 0.01, INPUT_LENGTH)
        
        if i < 100:
            # Fake Planet signature: slight dip
            dip_start = np.random.randint(5000, 15000)
            flux[dip_start:dip_start+500] -= 0.02
            label = 0
        elif i < 200:
            # Fake Eclipsing Binary signature: deep dip
            dip_start = np.random.randint(5000, 15000)
            flux[dip_start:dip_start+500] -= 0.15
            label = 1
        else:
            # Noise: no dip
            label = 2
            
        X.append(flux)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    # Convert to PyTorch tensors
    # Add channel dim: (batch_size, 1, input_length)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = ExoplanetCNN(input_length=INPUT_LENGTH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    # Update global model for inference
    global global_model
    global_model = model
    print("Training complete. Global model updated for inference.")

if __name__ == "__main__":
    # 1. Run the training loop with fake data to show loss decreasing
    train_dummy_model()
    
    # 2. Test the inference function on a new dummy sample
    print("\nTesting predict_exoplanet function on a new dummy array...")
    test_flux = np.random.normal(1, 0.01, INPUT_LENGTH)
    # Injecting a fake planet dip to test
    test_flux[10000:10500] -= 0.02
    
    prediction = predict_exoplanet(test_flux)
    print(f"Inference Output: {prediction}")
