    
#import os
import numpy as np
import h5py
    

def create_dataset(path_directory, h5py_file_directory):
    
    features_path, labels_path = path_extraction(path_directory)
    img_conc_features = concatenate_images(features_path, features = True)
    img_conc_labels = concatenate_images(labels_path, features = False)
    create_h5py(img_conc_features, img_conc_labels, h5py_file_directory)
    
def path_extraction(path_directory):

    ImgDir = path_directory
    features_path = []
    labels_path = []
    count = 0

    for folder in os.listdir(ImgDir):
        count +=1
        if 'Training' in folder:
            new_dir = os.path.join(ImgDir,folder)
            data = os.listdir(new_dir)
            for files in data:
                if 'flair' in files:
                    features_path.append(os.path.join(new_dir, files))
                if 'seg' in files:
                    labels_path.append(os.path.join(new_dir, files))

    #print(len(features_path))
    #print(len(labels_path))

    return features_path, labels_path
    
def concatenate_images(path, features = None):

    IMG_HEIGHT = 240
    IMG_WIDTH = 240
    IMG_DEPTH = 155

    img_conc = np.zeros((len(path),IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH))

    for file, i in zip(path,range(len(path))):
        img = nib.load(file)
        imgarr = img.get_fdata()

        if features:

            mean = np.mean(imgarr)
            std = np.std(imgarr)
            imgarr_norm = (imgarr - mean) / std
            imgarr_norm_clip = np.clip(imgarr_norm,-5,5)
            imgarr_std = imgarr_norm_clip/np.max(imgarr_norm_clip)
            imgarr = imgarr_std

        img_conc[i,:,:,:] = imgarr

    img_conc = np.concatenate(img_conc,axis=2)
    #print(np.shape(img_conc_features))

    return img_conc
    
def create_h5py(img_conc_features, img_conc_labels, h5py_file_directory):

    with h5py.File(h5py_file_directory, 'a') as f:
        f.create_dataset("features", data=img_conc_features, compression="gzip")
        f.create_dataset("labels", data=img_conc_labels, compression="gzip")
            
            
if __name__ == "__main__":

    create_dataset(path_directory, h5py_file_directory)