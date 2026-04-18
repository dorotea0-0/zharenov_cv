import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

base_dir = Path(__file__).parent

def count_holes(reg):
    img_shape = reg.image.shape
    padded_arr = np.zeros((img_shape[0] + 2, img_shape[1] + 2))
    padded_arr[1:-1, 1:-1] = reg.image
    inverted_mask = np.logical_not(padded_arr)
    bg_labeled = label(inverted_mask)
    return np.max(bg_labeled) - 1

def count_lines(reg):
    img_shape = reg.image.shape
    pattern = reg.image
    v_count = (np.sum(pattern, 0) / img_shape[0] == 1).sum()
    h_count = (np.sum(pattern, 1) / img_shape[1] == 1).sum()
    return v_count, h_count

def simmetry(reg, transpose=False):
    pattern = reg.image
    if transpose:
        pattern = pattern.T
    img_shape = pattern.shape

    upper_part = pattern[:img_shape[0] // 2]
    if img_shape[0] % 2 != 0:
        lower_part = pattern[img_shape[0] // 2 + 1:]
    else:
        lower_part = pattern[img_shape[0] // 2:]

    lower_part = lower_part[::-1]
    match = lower_part == upper_part
    return match.sum() / match.size

def classificator(reg):
    hole_count = count_holes(reg)
    if hole_count == 2:
        v_cnt, _ = count_lines(reg)
        v_cnt /= reg.image.shape[1]
        if v_cnt > 0.2:
            return "B"
        else:
            return "8"
    elif hole_count == 1:
        h_sym = simmetry(reg)
        v_sym = simmetry(reg, transpose=True)
        if h_sym > 0.8 and v_sym > 0.8:
            return "O"
        elif h_sym > 0.9 and v_sym > 0.6:
            return "D"
        elif h_sym > 0.3 and v_sym > 0.8:
            return "A"
        else:
            return "P"

    elif hole_count == 0:
        h_sym = simmetry(reg)
        v_sym = simmetry(reg, transpose=True)

        if h_sym == 1 and v_sym == 1:
            return "-"
        elif h_sym > 0.7 and v_sym > 0.7:
            if reg.eccentricity > 0.85:
                return "1"
            else:
                return "X"
        elif h_sym > 0.3 and v_sym > 0.8:
            if reg.eccentricity > 0.55:
                return "W"
            else:
                return "*"
        else:
            return "/"

    return "?"


img_data = imread('symbols.png')[:,:,:-1]
binary_mask = img_data.mean(2) > 0
labeled_mask = label(binary_mask)
print(np.max(labeled_mask))

regions_list = regionprops(labeled_mask)

classification_stats = {}
output_folder = base_dir / "output"
output_folder.mkdir(exist_ok=True)

plt.figure(figsize=(5,7))

for reg in regions_list:
    predicted_char = classificator(reg)
    if predicted_char not in classification_stats:
        classification_stats[predicted_char] = 0
    classification_stats[predicted_char] += 1
    plt.cla()
    plt.title(f"Class: {predicted_char}")
    plt.imshow(reg.image)
    plt.savefig(output_folder / f"{reg.label}.png")
print(classification_stats)

plt.imshow(binary_mask)
plt.show()