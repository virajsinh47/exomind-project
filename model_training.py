import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Define the 1D-CNN Architecture
class ExoplanetCNN(nn.Module):
    def __init__(self, input_length):
        super(ExoplanetCNN, self).__init__()
        self.input_length = input_length
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self._to_linear = None
        self._get_conv_output_size()
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3) 
        )

    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_length)
            output = self.conv_layers(dummy_input)
            self._to_linear = output.numel()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

import os

# Global model instance for inference
# We'll use 20000 to match the requested dummy data size
INPUT_LENGTH = 20000
global_model = ExoplanetCNN(input_length=INPUT_LENGTH)

# Load the real, trained AI brain if Member 2 has provided it!
model_path = r"D:\exomind-project\models\exoplanet_model.pth"
if os.path.exists(model_path):
    global_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Loaded real AI brain successfully!")

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
