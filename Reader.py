'''
Fully Convolutional Networks for Semantic Segmentation: Test Data Reader
'''
__author__ = 'qhan'

import numpy as np
import scipy.misc as misc
import PIL.Image as Image

'''
:filename: test data list
:resize_size:
:return: np array of images, names, original size
'''
def read_test_data(listname, height, width):
    
    images, names = [], []

    with open(listname, 'r') as f:
        
        image_dir = f.readline()[:-1]

        for line in f:

            name = line[:-1]
            path = image_dir + '/' + name
            print('\rpath: ' + path, end='', flush=True)
            image = Image.open(path)
            (w, h) = image.size
            #max_edge = max(w, h)
            #image = np.array( image.crop((0, 0, max_edge, max_edge)) )
            resized_image = misc.imresize(image, [height, width], interp='nearest')

            names.append(name)
            images.append(resized_image)

        print('')
        #h, w, _ = image.shape

    return np.array(images), np.array(names), (h, w)
