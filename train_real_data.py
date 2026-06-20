import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import the data pipeline and model definitions
from data_pipeline.data_pipeline import get_clean_lightcurve
from model_training import ExoplanetCNN, INPUT_LENGTH

def process_flux(flux_array, target_length):
    """Truncates or median-pads the flux array to exactly target_length."""
    if len(flux_array) == 0:
        return None
        
    # Convert astropy MaskedNDArray to standard numpy array to prevent np.pad crashes
    flux_array = np.array(flux_array, dtype=np.float32)
    
    if len(flux_array) > target_length:
        # Truncate
        return flux_array[:target_length]
    elif len(flux_array) < target_length:
        # Median pad
        pad_size = target_length - len(flux_array)
        pad_value = np.median(flux_array) if len(flux_array) > 0 else 1.0
        return np.pad(flux_array, (0, pad_size), 'constant', constant_values=pad_value)
    
    return flux_array

def main():
    # Define Training Data TIC IDs
    planet_tics = [
        "TIC 279741379", 
        "TIC 281541555", 
        "TIC 149090620", 
        "TIC 280053912", 
        "TIC 162239401"
    ]
    
    # Mix of Eclipsing Binaries and Noise/Non-planets
    noise_tics = [
        "TIC 149258286", # We'll assign label 1
        "TIC 149179973", # We'll assign label 1
        "TIC 149089947", # We'll assign label 2
        "TIC 280054000", # We'll assign label 2
        "TIC 149090630"  # We'll assign label 2
    ]
    
    X = []
    y = []
    
    print("Downloading and processing planet TICs (Label 0)...")
    for tic in planet_tics:
        result = get_clean_lightcurve(tic)
        # get_clean_lightcurve returns np.array([]) if it fails, else a tuple
        if isinstance(result, tuple) and len(result) == 2:
            time_arr, flux_arr = result
            flux_processed = process_flux(flux_arr, INPUT_LENGTH)
            if flux_processed is not None:
                X.append(flux_processed)
                y.append(0)  # 0: Planet
        else:
            print(f"Failed to process {tic}.")

    print("Downloading and processing noise/binary TICs (Labels 1 and 2)...")
    for i, tic in enumerate(noise_tics):
        result = get_clean_lightcurve(tic)
        if isinstance(result, tuple) and len(result) == 2:
            time_arr, flux_arr = result
            flux_processed = process_flux(flux_arr, INPUT_LENGTH)
            if flux_processed is not None:
                X.append(flux_processed)
                # Assign 1 to the first two, 2 to the rest
                label = 1 if i < 2 else 2
                y.append(label)
        else:
            print(f"Failed to process {tic}.")

    if len(X) == 0:
        print("No valid data collected. Exiting.")
        return

    # Convert lists to numpy arrays, then to PyTorch tensors
    X = np.array(X)
    y = np.array(y)
    
    # Our CNN expects input shape (batch_size, channels, input_length)
    # channels=1 for 1D CNN
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Use a small batch size since the dataset is very small
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize Model, Loss, and Optimizer
    model = ExoplanetCNN(input_length=INPUT_LENGTH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"\nStarting training for {epochs} epochs...")
    
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
        
    # Save the model weights
    save_path = 'exoplanet_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model weights successfully saved to '{save_path}'.")

if __name__ == "__main__":
    main()
