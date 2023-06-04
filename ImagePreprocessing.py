import matplotlib.pyplot as plt 
import glob
from skimage.color import rgb2lab
from skimage.transform import resize 
from collections import namedtuple
import numpy as np 
from tqdm import tqdm

np.random.seed(101)
#%matplotlib inline
Dataset = namedtuple('Dataset', ['X', 'y'])

def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis = 0).astype(np.float32)

def read_dataset_ppm(path, n_labels, resize_to):
    images = []
    labels = []

    for c in tqdm(range(n_labels)):
        full_path = path + '/' + format(c, '05d') + '/'
        for img_name in glob.glob(full_path + "*.ppm"):
            img = plt.imread(img_name).astype(np.float32)
            img = rgb2lab(img/255.0)[:, :, 0]
            if resize_to:
                img = resize(img, resize_to, mode = 'reflect')
            label = np.zeros((n_labels, ), dtype = np.float32)
            label[c] = 1.0
            images.append(img.astype(np.float32))
            labels.append(label)
    return Dataset(X = to_tf_format(images).astype(np.float32), y = np.matrix(labels).astype(np.float32))

def preprocessing(N_Classes, Resized_Image):
    
    dataset = read_dataset_ppm('GTSRB/Final_Training/Images', N_Classes, Resized_Image)
    return dataset
