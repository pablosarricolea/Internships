import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def visualize_histories(**images):
    """Import as tuples: one = (a,b)"""
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.ylim(0,1)
        plt.xlabel('epoch')
        if i == 0:
            plt.ylabel('dice_score')
        plt.title(' '.join(name.split('_')).title())
        plt.plot(np.asarray(image[0]))
        plt.plot(np.asarray(image[1]))
        plt.legend(['train', 'val'])
    plt.show()

def load_images_from_hdf5(directory_images,begin,stop):
    
    with h5py.File(os.path.join(directory_images), "r") as f:
        images = f["images"]
        images_train = images[:,:,begin:stop]
        
    return images_train