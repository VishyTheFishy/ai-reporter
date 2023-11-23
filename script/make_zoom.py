from PIL import Image
import os

# Create the new directory
output_directory = "zoom_CD29_MSC_20x"
os.makedirs(output_directory, exist_ok=True)

# Zoom factor (adjust as needed)
zoom_factor = 2.0  # 2.0 means 200% zoom

# Function to zoom and copy images
def zoom_and_copy(input_path, output_path):
    with Image.open(input_path) as img:
        # Calculate the new size after zooming
        new_size = tuple(int(dim * zoom_factor) for dim in img.size)
        # Resize (zoom) the image
        zoomed_img = img.resize(new_size, Image.ANTIALIAS)
        # Save the zoomed image to the output directory
        zoomed_img.save(output_path)

# Zoom and copy images from the input/train directory
input_train_directory = "path/to/CD29_MSC_20x/input/train"
for input_file in os.listdir(input_train_directory):
    if input_file.endswith(".tiff"):
        input_path = os.path.join(input_train_directory, input_file)
        output_path = os.path.join(output_directory, input_file)
        zoom_and_copy(input_path, output_path)

# Zoom and copy images from the input/test directory
input_test_directory = "path/to/CD29_MSC_20x/input/test"
for input_file in os.listdir(input_test_directory):
    if input_file.endswith(".tiff"):
        input_path = os.path.join(input_test_directory, input_file)
        output_path = os.path.join(output_directory, input_file)
        zoom_and_copy(input_path, output_path)
