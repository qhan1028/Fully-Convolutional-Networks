#
#   Fully Convolutional Networks for Semantic Segmentation: Test Data Reader
#   Written by Qhan
#

from __future__ import print_function

import numpy as np
import scipy.misc as misc
from PIL import Image

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


#
#   Fully Convolutional Networks: MIT Scene Parsing Data Reader
#

import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import tensorflow_utils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        print ("> [SPD] Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("> [SPD] Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("> [SPD] Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('> [SPD] No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("> [SPD] Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('> [SPD] No. of %s files: %d' % (directory, no_of_images))

    return image_list
