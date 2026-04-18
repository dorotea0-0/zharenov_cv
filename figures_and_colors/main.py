import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.color import rgb2hsv

img = imread('balls_and_rects.png')
print(img.shape)
hsv = rgb2hsv(img)
h = hsv[:,:,0]

binary_all = np.any(img[:,:,:3] > 0, axis=2)
labeled_all = label(binary_all)

regions = regionprops(labeled_all, intensity_image=h)

circles = []
rectangles = []

for region in regions:
    if region.extent < 0.9:
        circles.append(region)
    else:
        rectangles.append(region)

print(f"\nВсего объектов: {len(regions)}")
print(f"Кружков: {len(circles)}")
print(f"Прямоугольников: {len(rectangles)}")

colours_circles = [r.intensity_mean for r in circles]
colours_rects = [r.intensity_mean for r in rectangles]

def group_hues(hues, delta=0.05):
    if not hues:
        return []
    hues = sorted(hues)
    groups = [[hues[0]]]
    for val in hues[1:]:
        if abs(val - groups[-1][-1]) < delta:
            groups[-1].append(val)
        else:
            groups.append([val])
    return groups

c_groups = group_hues(colours_circles)
r_groups = group_hues(colours_rects)

print("\n Кружки по цветам")
for i, grp in enumerate(c_groups):
    print(f"Группа {i}: тон: {np.mean(grp):.3f} количество: {len(grp)}")

print("\nПрямоугольники по цветам")
for i, grp in enumerate(r_groups):
    print(f"Группа {i}: тон: {np.mean(grp):.3f} количество: {len(grp)}")