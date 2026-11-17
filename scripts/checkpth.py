import torch

# Replace with the actual path to your .pth file
checkpoint_path = "C:/Users/hanna/Documents/Github-Clone1/DroneIdentifier/models/drone_cnn_4class.pth"

# Load the checkpoint (use map_location='cpu' if it was trained on GPU and you are inspecting on CPU)
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 1. Check if the checkpoint is a state_dict or a dictionary containing extra metadata
print("Type of checkpoint object:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys in checkpoint dictionary:")
    for key in checkpoint.keys():
        print(f" - {key}")
        
    # If it contains a 'state_dict' key, inspect that specifically
    state_dict = checkpoint.get('state_dict', checkpoint)
else:
    state_dict = checkpoint

# 2. Print layer names and their tensor shapes
print("\nLayer weights and shapes inside the model:")
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")