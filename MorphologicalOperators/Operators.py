import numpy as np
import cv2

# Görüntüyü yükleme ve ikili hale getirme
def load_binary_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    return binary_image

# Erozyon işlemi
def erosion(image, kernel):
    output = np.zeros_like(image)
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            if np.array_equal(region & kernel, kernel):
                output[i - pad, j - pad] = 1
    return output

# Genişleme işlemi
def dilation(image, kernel):
    output = np.zeros_like(image)
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            if np.any(region & kernel):
                output[i - pad, j - pad] = 1
    return output

# Açma işlemi: Erozyon -> Genişleme
def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)

# Kapama işlemi: Genişleme -> Erozyon
def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)

