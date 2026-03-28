import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label
import os

def get_centre(img):
    labeled = label(img)
    centre = []
    for prop in measure.regionprops(labeled):
        centre.append(prop.centroid)
    return centre


path = "out"
count = len(os.listdir(path))
images = []
for i in range(count):
    img = np.load(os.path.join(path, f"h_{i}.npy"))
    images.append(img)

for i, img in enumerate(images):
    centre = get_centre(img)
    if i==0:
        trajectory = [[c] for c in centre]
    else:
        for traj in trajectory:
            last_centre = traj[-1]
            dists = [((c[0] - last_centre[0]) ** 2 + (c[1] - last_centre[1]) ** 2) ** 0.5 for c in centre]
            idx = np.argmin(dists)

            traj.append(centre[idx])
            centre.pop(idx)
plt.figure()

for traj in trajectory:
    ys, xs = zip(*traj)
    plt.plot(xs, ys, marker='o')

plt.grid()
plt.show()