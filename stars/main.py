import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import (opening,dilation,closing,erosion)

image = np.load("stars.npy")

struct = np.ones((3,3))
processed = opening(image, footprint=struct)

labeled_with_stars = label(image)
labeled_no_stars = label(processed)

print(f"{labeled_with_stars.max()-labeled_no_stars.max()}")