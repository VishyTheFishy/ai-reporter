from PIL import Image
import os

# Create the new directory

input_d = "/home/vishrutsgoyal/Nucleus_MSC_20x_BF"
output_d = "/home/vishrutsgoyal/Nucleus_MSC_20x_BF_zoom"

x,y = 512,512

# Zoom factor (adjust as needed)
zoom = 2.0  # 2.0 means 200% zoom

# Function to zoom and copy images
def zoom_and_copy(input_path, output_path):
    with Image.open(input_path) as img:
        w, h = img.size
        zoom2 = zoom * 2
        img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))
        zoomed_img = img.resize((w, h), Image.LANCZOS)
        # Save the zoomed image to the output directory
        zoomed_img.save(output_path)

def copy_directory(input_directory, output_directory):
    for input_file in os.listdir(input_directory):
        if input_file.endswith(".tif"):
            input_path = os.path.join(input_directory, input_file)
            output_path = os.path.join(output_directory, input_file)
            zoom_and_copy(input_path, output_path)


# Zoom and copy images from the input/train directory
partitions = ["input/train", "output/train", "input/test", "output/test"]
for partition in partitions:
    copy_directory(os.path.join(input_d,partition),os.path.join(output_d,partition))


