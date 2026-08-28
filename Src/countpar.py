import torch

def count_parameters(model_path):
    print(f"Loading model from: {model_path}\n" + "-"*40)
    
    # Load the saved file safely on CPU
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Target the actual model weights dictionary found in your checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)

    # Calculate total elements in all weight tensors
    if isinstance(state_dict, dict):
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        trainable_params = "Unknown (state_dict weights do not store gradient status)"
    elif isinstance(checkpoint, torch.nn.Module):
        total_params = sum(p.numel() for p in checkpoint.parameters())
        trainable_params = sum(p.numel() for p in checkpoint.parameters() if p.requires_grad)
    else:
        raise ValueError("Unsupported PyTorch model format.")

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params}")

if __name__ == "__main__":
    MODEL_PATH = "/home/xavier/Sci-it/Src/best_model.pt"
    count_parameters(MODEL_PATH)
