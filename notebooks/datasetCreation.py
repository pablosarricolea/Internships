import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

def path_extraction(path_directory, h5py_file_directory):

    ImgDir = path_directory
    features_path = []
    count = 0

    for folder in os.listdir(ImgDir):
        
        count +=1
        
        if 'Training' in folder:
            
            new_dir = os.path.join(ImgDir,folder)
            data = os.listdir(new_dir)
            
            for files in data:
                
                if 't1.' in files and 't1_' in h5py_file_directory:
                    features_path.append(os.path.join(new_dir, files))
                if 't1ce.' in files and 't1ce_' in h5py_file_directory:
                    features_path.append(os.path.join(new_dir, files))
                if 't2.' in files and 't2_' in h5py_file_directory:
                    features_path.append(os.path.join(new_dir, files))
                if 'flair.' in files and 'flair_' in h5py_file_directory:
                    features_path.append(os.path.join(new_dir, files))
                if 'seg.' in files and 'seg_' in h5py_file_directory:
                    features_path.append(os.path.join(new_dir, files))

    return features_path

def concatenate_images(path, features = None):
        
    IMG_HEIGHT = 240
    IMG_WIDTH = 240
    IMG_DEPTH = 155

    img_conc = np.zeros((len(path),IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH))

    for file, i in zip(path,range(len(path))):

        img = nib.load(file)
        imgarr = img.get_fdata()

        if features:

            max_value = np.max(imgarr)
            min_value = np.min(imgarr)
            imgarr_std = (imgarr - min_value) / (max_value - min_value)
            imgarr = imgarr_std

        img_conc[i,:,:,:] = imgarr

    img_conc = np.concatenate(img_conc,axis=2)

    return img_conc
    
def create_h5py(img_conc_features, h5py_file_directory):

    with h5py.File(h5py_file_directory, 'a') as f:
        
        f.create_dataset("images", data=img_conc_features, compression="gzip")
        
def create_dataset(path_directory, h5py_file_directory, features = None):
    
    features_path = path_extraction(path_directory, h5py_file_directory)
    
    img_conc_features = concatenate_images(features_path, features)
    
    create_h5py(img_conc_features, h5py_file_directory)
    
if __name__ == "__main__":
    create_dataset(path_directory, h5py_file_directory)
    