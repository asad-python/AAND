import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='latin1')
    return data_dict


def display_all_images(all_images, all_labels, class_names):
    num_batches = len(all_images)
    num_images_per_batch = min(5, len(all_images[0]))  # Displaying at most 5 images per batch

    plt.figure(figsize=(15, 3 * num_batches))

    for i in range(num_batches):
        images = all_images[i]
        labels = all_labels[i]
        batch_number = i + 1
        total_images_in_batch = len(images)

        for j in range(num_images_per_batch):
            plt.subplot(num_batches, num_images_per_batch, i * num_images_per_batch + j + 1)
            plt.imshow(images[j])
            plt.title(f"Batch: {batch_number}\n{class_names[labels[j]]}")
            plt.axis('off')

        plt.text((num_images_per_batch - 1) * 0.5, -0.2, f"Total Images: {total_images_in_batch}", ha='center')

    plt.suptitle("All Batches")
    plt.show()


cifar10_batch_folder = 'cifar-10-batches-py 2'

# List of batch files
batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Accumulate images and labels for all batches
all_images = []
all_labels = []

# Iterate over each batch
for selected_batch in batch_files:
    # Construct the path to the selected batch
    batch_path = os.path.join(cifar10_batch_folder, selected_batch)

    # Unpickle the selected batch
    batch_dict = unpickle(batch_path)

    # Try to identify the correct key for images
    images_key = 'data' if 'data' in batch_dict else 'data_batch'

    # Extracting images and labels
    images = batch_dict[images_key].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch_dict['labels']  # Use 'labels' instead of b'labels'

    # Accumulate images and labels
    all_images.append(images)
    all_labels.append(labels)

# Displaying all accumulated images with labels and batch information in a single plot
display_all_images(all_images, all_labels, class_names)
