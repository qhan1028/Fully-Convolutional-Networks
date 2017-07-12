'''
FCN_tensorflow: Test Data Reader
'''
__author__ = 'qhan'

from os.path import basename, splitext
import numpy as np
import scipy.misc as misc

'''
:filename: test data list
:resize_size:
:return: np array of images & names
'''
def read_test_data(listname, width, height):
    
    images, names = [], []

    with open(listname, 'r') as f:
        
        image_dir = f.readline()[:-1]

        for line in f:
            
            name = basename(line[:-1])
            names.append(splitext(name)[0])

            image = misc.imread(image_dir + '/' + name)
            resized_image = misc.imresize(image, [height, width], interp='nearest')
            images.append(resized_image)

    return np.array(images), np.array(names)
