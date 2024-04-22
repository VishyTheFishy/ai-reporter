from PIL import Image
import os
from random import randrange

# Create the new directory

org_d = "/home/vishrutsgoyal/CD29_MSC_20x_PC"
final_d = "/home/vishrutsgoyal/CD29_MSC_20x_PC_2x"

os.makedirs(final_d + "/input/train",exist_ok=True)
os.makedirs(final_d + "/output/train",exist_ok=True)
os.makedirs(final_d + "/input/test",exist_ok=True)
os.makedirs(final_d + "/output/test",exist_ok=True)


size = 1024
zoom = 2  # 2.0 means 200% zoom
matrix = size // zoom

# Function to zoom and copy images
def copy_pair(split, name):
        try:
                with Image.open(os.path.join(org_d, "input",split,name)) as img1:
                        with Image.open(os.path.join(org_d, "output",split,name)) as img2:
                                x1 = randrange(0, size - matrix)
                                y1 = randrange(0, size - matrix)
                                img1.crop((x1, y1, x1 + matrix, y1 + matrix)).resize((size,size)).save(os.path.join(final_d, "input",split,name))
                                img2.crop((x1, y1, x1 + matrix, y1 + matrix)).resize((size,size)).save(os.path.join(final_d, "output",split,name))
        except FileNotFoundError: 
                print("Not Found")

def copy(split):
    for input_file in os.listdir(os.path.join(org_d,"input",split)):
        if input_file.endswith(".tif"):
            copy_pair(split, input_file)


# Zoom and copy images from the input/train directory
copy("train")
copy("test")

