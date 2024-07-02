import torch

# Loading weights
state_dict = torch.load('blood_50k.pth', map_location=torch.device('cpu'))

# Checking keys in weight documents
print("Keys in the state_dict:")
for key in state_dict.keys():
    print(f"{key}:")
    # Printing sub-keys
    if isinstance(state_dict[key], dict):
        for subkey in state_dict[key].keys():
            print(f"  {subkey}")
    else:
        print("  (No subkeys)")

# Printing 'model_state_dict' sub-keys
if 'model_state_dict' in state_dict:
    model_state_dict = state_dict['model_state_dict']
    print("\nKeys in the model_state_dict:")
    for key in model_state_dict.keys():
        print(f"  {key}")
