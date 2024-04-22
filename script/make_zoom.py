from PIL import Image
import os

# Create the new directory

org_d = "/home/vishrutsgoyal/Nucleus_MSC_20x_PC"
final_d = "/home/vishrutsgoyal/Nucleus_MSC_20x_PC_2x"

size = 1024
zoom = 2.0  # 2.0 means 200% zoom
matrix = size // zoom

# Function to zoom and copy images
def copy_pair(split, name):
        img1 = Image.open(os.path.join(org_d, "input",split,name))
        img2 = Image.open(os.path.join(org_d, "output",split,name))
        x1 = randrange(0, size - matrix)
        y1 = randrange(0, size - matrix)
        img1.crop((x1, y1, x1 + matrix, y1 + matrix))
        img2.crop((x1, y1, x1 + matrix, y1 + matrix))
        # Save the zoomed image to the output directory
        img1.save(os.path.join(final_d, "input",split,name))
        img2.save(os.path.join(final_d, "output",split,name))

def copy(split):
    for input_file in os.listdir(os.path.join(org_d,"input",split)):
        if input_file.endswith(".tif"):
            copy_pair(split, input_file)


# Zoom and copy images from the input/train directory
copy("train")
copy("test")

