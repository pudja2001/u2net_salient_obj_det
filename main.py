from source.Segmentator import ImageSegmentator
from skimage import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    weight_path = "./weights/u2net/u2net.pth"
    img_path = "./source/models/test_data/test_images/kunyit1.png"
    img_curr = Image.open(img_path)
    img_arr = np.array(img_curr)

    # Segment image
    results = ImageSegmentator(weight_path).segment(img_arr)
    print("output array:")
    print(results)
    print("shape:", results.shape)

    plt.imshow(results)
    plt.show()