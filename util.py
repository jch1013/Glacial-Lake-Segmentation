import matplotlib.pyplot as plt
import numpy as np

"""
Utility file to store various useful methods for working with this code
"""
def compare_images(image, label):
    plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1)
    plt.title('Satellite Image')
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Lake Prediction')
    plt.imshow(label, cmap='gray')
    plt.axis('off')
    plt.show()


def compare_image_sets(image_set, label_set):
    for i in range(len(image_set)):
        compare_images(image_set[i], label_set[i])


def compare_image_label_prediction(image, label, prediction):
    plt.figure(figsize=(24, 8))

    plt.subplot(1, 3, 1)
    plt.title('Color Image')
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title('Label')
    plt.imshow(label, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Prediction')
    plt.imshow(prediction, cmap='gray')  # <-- replace with your third image variable
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def calculate_jaccard(y_true, y_pred):
    """
    y_true and y_pred should be binary masks (0 or 1), same shape.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 1.0


def calculate_dice(y_true, y_pred):
    """
    y_true and y_pred should be binary masks (0 or 1), same shape.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()

    return 2.0 * intersection / total if total > 0 else 1.0