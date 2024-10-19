from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lite import LITEModel

def analyze_image(file_path):
    # Load the PNG file
    image = Image.open(file_path).convert('L')  # Convert to grayscale

    # Convert image to numpy array
    image_array = np.array(image)

    # Flatten the 2D image array to 1D time-series data
    time_series_data = image_array.flatten()

    # Convert the flattened array to a PyTorch tensor
    input_data = torch.tensor(time_series_data, dtype=torch.float32)

    # If the model expects a specific shape, reshape the input accordingly
    input_data = input_data.unsqueeze(0)  # Adding batch dimension

    # Load the model architecture
    model = LITEModel(length_TS=len(input_data), n_classes=4)

    # Load the model weights
    model_path = 'model/car_lite_model.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    # Make prediction using the model
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(input_data)

    # Get the predicted class index
    predicted_class = torch.argmax(prediction, dim=1).item()  

    classes = ['Sedan', 'Pickup', 'Minivan', 'SUV']

    return classes[predicted_class], F.softmax(prediction, dim=1)

