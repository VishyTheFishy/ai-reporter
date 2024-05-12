from PIL import Image
import os
import numpy as np
from random import randrange

# Create the new directory

org_d = "/home/vishrutsgoyal/Nucleus_MSC_20x_PC"
final_d = "/home/vishrutsgoyal/Nucleus_MSC_20x_PC_grad-1"

os.makedirs(final_d + "/input/train",exist_ok=True)
os.makedirs(final_d + "/output/train",exist_ok=True)
os.makedirs(final_d + "/input/test",exist_ok=True)
os.makedirs(final_d + "/output/test",exist_ok=True)


size = 1024
slope = .1
mask = np.linspace(1-slope,1+slope, num=size)
mask = np.stack((mask,mask,mask)).T

# Function to zoom and copy images
def copy_pair(split, name):
        try:
          with Image.open(os.path.join(org_d, "input",split,name)) as img1:
            with Image.open(os.path.join(org_d, "output",split,name)) as img2:
              im1 = np.array(img1)
              scale = 255/(np.max(im1.flatten())*(1+slope))
              print(scale)
              for  i in range(len(im1)):
                im1[i] = scale*mask*im1[i]
              print(np.max(im1.flatten()))
              
              Image.fromarray(im1).save(os.path.join(final_d, "input",split,name))
              img2.save(os.path.join(final_d, "output",split,name))
        except FileNotFoundError: 
                print("Not Found")

def copy(split):
    for input_file in os.listdir(os.path.join(org_d,"input",split)):
        if input_file.endswith(".tif"):
            copy_pair(split, input_file)


# Zoom and copy images from the input/train directory
copy("train")
copy("test")
