from pathlib import Path
import torch

def load_model(model_path):
    """Load the trained model from the specified path."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

def get_device():
    """Get the device to run the model on (CPU or GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_model(model_path):
    """Load the model and prepare it for inference."""
    device = get_device()
    model = load_model(model_path).to(device)
    return model, device