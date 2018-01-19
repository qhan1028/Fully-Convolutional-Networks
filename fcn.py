#
#   Fully Convolutional Networks
#   Modified by Qhan
#

from __future__ import print_function
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import PIL.Image as Image
import scipy.misc as misc
import datetime
import tensorflow as tf

import tensorflow_utils as utils
import batch_datset_reader as dataset

import timer
from reader import read_test_data, read_dataset, parse_args
from augment import augment

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2 # original is 151
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
            print('conv ' + name[4:] + ':', current.shape)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if args.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
            print('pool ' + name[4:] + '  :', current.shape)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("> [FCN] Setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(args.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        print('----------------------------------------------------')
        print('conv 5_3:', conv_final_layer.get_shape())

        pool5 = utils.max_pool_2x2(conv_final_layer)
        print('pool 5  :', pool5.get_shape())

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6") # original is [7, 7, 512, 4096]
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        print('conv 6  :', conv6.get_shape())
        relu6 = tf.nn.relu(conv6, name="relu6")
        if args.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        print('conv 7  :', conv7.get_shape())
        relu7 = tf.nn.relu(conv7, name="relu7")
        if args.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        print('conv 8  :', conv8.get_shape())
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        print('conv t1 :', conv_t1.get_shape())
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        print('fuse 1  :', fuse_1.get_shape())

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        print('conv t2 :', conv_t2.get_shape())
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        print('fuse 2  :', fuse_2.get_shape())

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        print('conv t3 :', conv_t3.get_shape())

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        print('prediction:', annotation_pred.get_shape())
    
    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if args.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main():
    # tensorflow input and output
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)

    # Summary
    print('====================================================')
    if args.mode != 'test':
        tf.summary.image("input_image", image, max_outputs=4)
        tf.summary.image("ground_truth", tf.cast(annotation * 255, tf.uint8), max_outputs=4)
        tf.summary.image("pred_annotation", tf.cast(pred_annotation * 255, tf.uint8), max_outputs=4)
        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                              name="entropy")))
        tf.summary.scalar("train_entropy", loss)

        trainable_var = tf.trainable_variables()
        if args.debug:
            for var in trainable_var:
                utils.add_to_regularization_and_summary(var)
        train_op = train(loss, trainable_var)

        print("> [FCN] Setting up summary op...")
        summary_op = tf.summary.merge_all()

        # Validation summary
        val_summary = tf.summary.scalar("validation_entropy", loss)

        # Read data
        print("> [FCN] Setting up image reader...")
        train_records, valid_records = read_dataset(args.data_dir)
        print('> [FCN] Train len:', len(train_records))
        print('> [FCN] Val len:', len(valid_records))

    t = timer.Timer() # Qhan's timer

    if args.mode != 'test':
        print("> [FCN] Setting up dataset reader")
        image_options = {'resize': True, 'resize_height': IMAGE_HEIGHT, 'resize_width': IMAGE_WIDTH}
        if args.mode == 'train':
            t.tic(); train_dataset_reader = dataset.BatchDatset(train_records, image_options, mode='train')
            print('> [FCN] Train data set loaded. %.4f ms' % t.toc())
        t.tic(); validation_dataset_reader = dataset.BatchDatset(valid_records, image_options, mode='val')
        print('> [FCN] Validation data set loaded. %.4f ms' % t.toc())

    # Setup Session
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.90, allow_growth=True)
    sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options) )

    # Initialize model
    print("> [FCN] Setting up Saver...", flush=True)
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

    print("> [FCN] Initialize variables... ", flush=True, end='')
    t.tic(); sess.run(tf.global_variables_initializer())
    print('%.4f ms' % t.toc())

    t.tic()
    ckpt = tf.train.get_checkpoint_state(args.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("> [FCN] Model restored..." + ckpt.model_checkpoint_path + ', %.4f ms' % (t.toc()))

    print('==================================================== [%s]' % args.mode)

    if args.mode == 'train':
        np.random.seed(1028)
        start = args.start_iter
        end = start + args.iter + 1
        for itr in range(start, end):

            # Read batch data
            train_images, train_annotations = train_dataset_reader.next_batch(args.batch_size)
            images = np.zeros_like(train_images)
            annotations = np.zeros_like(train_annotations)

            # Data augmentation
            for i, (im, ann) in enumerate(zip(train_images, train_annotations)):
                flip_prob = np.random.random()
                aug_type = np.random.randint(0, 3)
                randoms = np.random.random(2)
                images[i] = augment(im, flip_prob, aug_type, randoms)
                annotations[i] = augment(ann, flip_prob, aug_type, randoms)

            t.tic()
            feed_dict = {image: images, annotation: annotations, keep_probability: 0.85}
            sess.run(train_op, feed_dict=feed_dict)
            train_time = t.toc()

            if itr % 10 == 0 and itr > 10:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, itr)
                print("[%6d], Train_loss: %g, %.4f ms" % (itr, train_loss, train_time), flush=True)

            if itr % 100 == 0 and itr != 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(args.batch_size * 2)
                val_feed_dict = { image: valid_images, annotation: valid_annotations, keep_probability: 1.0}
                
                t.tic(); val_loss, val_str = sess.run([loss, val_summary], feed_dict=val_feed_dict)
                print("[%6d], Validation_loss: %g, %.4f ms" % (itr, val_loss, t.toc()))

                summary_writer.add_summary(val_str, itr)
             
            if itr % 1000 == 0 and itr != 0:
                saver.save(sess, args.logs_dir + "model.ckpt", itr)

    elif args.mode == 'visualize':
        for itr in range(20):
            valid_images, valid_annotations = validation_dataset_reader.get_random_batch(1)
            
            t.tic(); pred = sess.run(pred_annotation, feed_dict={image: valid_images, keep_probability: 1.0})
            print("> [FCN] Saved image: %d, %.4f ms" % (itr, t.toc()))

            valid_annotations = np.squeeze(valid_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)

            utils.save_image(valid_images[0].astype(np.uint8), args.res_dir, name="inp_" + str(itr))
            utils.save_image(valid_annotations[0].astype(np.uint8), args.res_dir, name="gt_" + str(itr))
            utils.save_image(pred[0].astype(np.uint8), args.res_dir, name="pred_" + str(itr))

    elif args.mode == 'test':
        images, names, (H, W) = read_test_data(args.test_dir, IMAGE_HEIGHT, IMAGE_WIDTH)
        for i, (im, name) in enumerate(zip(images, names)):
            
            t.tic(); pred = sess.run(pred_annotation, feed_dict={image: im.reshape((1,) + im.shape), keep_probability: 1.0})
            print('> [FCN] Test: %d,' % (i) + ' Name: ' + name + ', %.4f ms' % t.toc())
            
            pred = pred.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            if args.video:
                save_video_image(im, pred, args.res_dir + '/pred_%05d' % (i) + '.png', H, W)
            else:
                misc.imsave(args.res_dir + '/inp_%d' % (i) + '.png', im.astype(np.uint8))
                misc.imsave(args.res_dir + '/pred_%d' % (i) + '.png', pred.astype(np.uint8))

    else:
        pass

def save_video_image(im, pred, name, oh, ow):
    
    bg = np.where(pred != 1)
    im[bg] = [0, 255, 0]
    max_edge = max(oh, ow)
    resized_im = misc.imresize(im, [oh, ow], interp='nearest')
    image = Image.fromarray(np.uint8(resized_im))
    #image = image.crop((0, 0, ow, oh))
    image.save(name)


if __name__ == "__main__":

    args = parse_args()
    print('====================================================')
    main()
