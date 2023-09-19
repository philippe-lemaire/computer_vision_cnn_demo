import numpy as np
from cv2 import rotate


def get_augmented_data(X, y):
    """Takes in a numpy array of images and creates 3 variants with 90 degrees, 180 degrees and 270 degrees rotations of it.
    Then stacks X + each rotation of X.
    Stacks the y labels so that each rotated image gets its label as well"""
    rotated_arrays = []
    for rotation in range(3):
        rotated_imgs = []
        for img in X:
            rotated_img = rotate(img, rotateCode=rotation)
            rotated_imgs.append(rotated_img)
        rotated_arrays.append(np.array(rotated_imgs))

    for ar in rotated_arrays:
        ar = ar.reshape(ar.shape[0], 28, 28, 1)
        X = np.vstack((X, ar))
    y = np.tile(y, 4)
    return X, y
