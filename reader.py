#
#   Fully Convolutional Networks for Semantic Segmentation: Test Data Reader
#   Written by Qhan
#

from __future__ import print_function

import argparse

# Argument parser
def parse_args():

    parser = argparse.ArgumentParser('Fully Convolutional Networks for Semantic Segmentation.')
    parser.add_argument('-m', '--mode', metavar='MODE', default='train', choices=['train', 'visualize', 'test'], help='Mode: train / visualize / test.')
    parser.add_argument('-b', '--batch-size', metavar='N', default=2, nargs='?', type=int, help='Batch size for training.')
    parser.add_argument('-e', '--learning-rate', '--eta', metavar='N', default=1e-4, nargs='?', type=float, help='Learning rate for Adam Optimizer.')
    parser.add_argument('-i', '--iter', metavar='N', default=int(1e5), nargs='?', type=int, help='Max iteration for training.')
    parser.add_argument('-si', '--start-iter', metavar='N', default=0, nargs='?', type=int, help='Start iteration for training.')
    parser.add_argument('-d', '--data-dir', metavar='DIR', default='Data_zoo/MIT_SceneParsing', nargs='?', help='Path to training & validation data.')
    parser.add_argument('-ld', '--logs-dir', metavar='DIR', default='logs', nargs='?', help='Path to logs directory.')
    parser.add_argument('-rd', '--res-dir', metavar='DIR', default='res', nargs='?', help='Path to result directory.')
    parser.add_argument('-md', '--model-dir', metavar='DIR', default='Model_zoo', nargs='?', help='Path to vgg pretrained model.')
    parser.add_argument('--debug', action='store_true', default=True, help='Debug mode.')
    parser.add_argument('--testlist', metavar='FILE', default='testlist.txt', nargs='?', help='Test list for testing.')
    parser.add_argument('-v', '--video', action='store_true', default=False, help='Resize back to original size.')
    args = parser.parse_args()

    args.data_dir += '/'
    args.logs_dir += '/'
    args.res_dir += '/'
    args.model_dir += '/'
    if not os.path.exists(args.res_dir): os.mkdir(args.res_dir)

    print('====================================================')
    for key, value in vars(args).items(): print('> [Args]', key + ':', value)
    
    return args


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
