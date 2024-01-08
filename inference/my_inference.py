from collections import OrderedDict
import yaml
from nvidia.github.FourCastNet.networks.afnonet import AFNONet
import torch

# Load the configuration from the YAML file
with open('AFNO.YAML', 'r') as stream:
    config = yaml.safe_load(stream)

# Update inf_data_path to your specific file for inference
config['afno_backbone']['inf_data_path'] = '/path/to/your/inference_file.h5'

# Load your pre-trained model
def load_model(checkpoint_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AFNONet()  # Instantiate your AFNONet model
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Load state dict
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint.items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    return model

# Usage example:
checkpoint_path = 'path/to/your/checkpoint.pth'
loaded_model = load_model(checkpoint_path)


# Load pre-trained weights
checkpoint = torch.load('path_to_your_pretrained_model_weights.pth')

model.load_state_dict(checkpoint['model_state_dict'])

# Set the model in evaluation mode
model.eval()

# Perform inference on your file
with torch.no_grad():
    # Assuming your model has a method `predict` for inference
    predictions = model.predict()

    # model von sample
    # 
    # for loop, t=0 
    # for batch in data loader
    #     for t in time period
    #           initial condition = batch t = 0
    #   14 ground truth
    #   14 predictions
    # xarray packen, zusätzliche dimension lead time, wie viele Schritte in der Zukunft vorhergesagt, "wie viele schritte vorher"

    # Handle or save your predictions here
    print(predictions)  # Replace this with your handling logic
