import torch

def save_model(model, model_path):
    """
    Save the trained model to a specified path.
    
    Parameters:
    - model: The trained model to save.
    - model_path: The file path where the model will be saved.
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
def load_model(model_path):
    """
    Load a trained model from a specified path.
    Parameters:
    - model_path: The file path from where the model will be loaded.
    Returns:
    - model: The loaded model.
    """
    model = torch.load(model_path)
    print(f"Model loaded from {model_path}")
    return model