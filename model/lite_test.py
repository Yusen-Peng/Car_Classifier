from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from lite import LITEModel
def analyze_image(file_path):
  if __name__ == '__main__':
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
      model_path = 'car_lite_model.pth'
      state_dict = torch.load(model_path)
      model.load_state_dict(state_dict)
      model.eval()  # Set the model to evaluation mode

      #print("Model loaded successfully.")

      # Make prediction using the model
      with torch.no_grad():  # Disable gradient calculation for inference
          prediction = model(input_data)

      # Print the prediction
      #print("Prediction:", prediction)

        # Get the predicted class index
      predicted_class = torch.argmax(prediction, dim=1).item()


      classes = ['Sedan', 'Pickup', 'Minivan', 'SUV'] 

      return classes[predicted_class]
      # Print the predicted class
      #print("Predicted class:", classes[predicted_class])
