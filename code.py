import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from handwriting_features.features import HandwritingFeatures
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Image Preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (128, 32))
    return resized_image

# Step 2: Handwriting Analysis
def analyze_handwriting(image):
    data = {
        "x": np.arange(image.shape[1]),
        "y": np.arange(image.shape[0]),
        "pressure": image.flatten()
    }
    sample = HandwritingFeatures.from_numpy_array(data)
    
    # Extract features
    stroke_width = sample.stroke_width(statistics=["mean"])['mean']
    connected = sample.number_of_intra_stroke_intersections(statistics=["sum"])['sum']
    tilt = sample.tilt(statistics=["mean"])['mean']
    
    characteristics = {
        "stroke_width": stroke_width,
        "connected": connected,
        "tilt": tilt
    }
    
    return characteristics

# Step 3: RNN Model Definition
class HandwritingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HandwritingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 128 * 32  # Image size (128x32)
hidden_size = 256
output_size = input_size

model = HandwritingRNN(input_size, hidden_size, output_size)

# Step 4: Training the Model
def load_arial_font_image(text):
    # Generate Arial font image
    font_image = Image.new('L', (128, 32), color=255)
    draw = ImageDraw.Draw(font_image)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text((10, 5), text, font=font, fill=0)
    return np.array(font_image)

def train_model(model, train_images, text, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    arial_font_image = load_arial_font_image(text)
    arial_font_image = torch.tensor(arial_font_image.flatten(), dtype=torch.float32).unsqueeze(0)
    
    for epoch in range(num_epochs):
        for image in train_images:
            image = torch.tensor(image.flatten(), dtype=torch.float32).unsqueeze(0)
            
            output = model(image)
            loss = criterion(output, arial_font_image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Text Generation
def generate_handwritten_text(model, text, output_path):
    arial_font_image = load_arial_font_image(text)
    arial_font_image = torch.tensor(arial_font_image.flatten(), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        generated_image_flat = model(arial_font_image)
        generated_image = generated_image_flat.view(32, 128).cpu().numpy()
    
    # Create a new transparent image
    width, height = 800, 200  # Adjust size as needed
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    
    # Convert generated image to RGBA format
    generated_image_pil = Image.fromarray((generated_image * 255).astype(np.uint8), mode='L').convert('RGBA')
    
    # Paste the generated text image onto the transparent background
    image.paste(generated_image_pil, (0, 0), generated_image_pil)
    image.save(output_path, 'PNG')

# Step 6: Integration
def process_handwriting_images(image_paths, text, output_path):
    preprocessed_images = [preprocess_image(image_path) for image_path in image_paths]
    characteristics = [analyze_handwriting(image) for image in preprocessed_images]
    train_model(model, preprocessed_images, text, num_epochs=100)
    generate_handwritten_text(model, text, output_path)

# Example usage
# image_paths = ['image1.jpg', 'image2.jpg']
# process_handwriting_images(image_paths, "Hello", 'output.png')
