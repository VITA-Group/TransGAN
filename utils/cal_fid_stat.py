# -*- coding: utf-8 -*-
# @Date    : 2019-07-26
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0


import os
import glob
import argparse
import numpy as np
from imageio import imread
import tensorflow as tf

import utils.fid_score as fid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='set path to training set jpg images dir')
    parser.add_argument(
        '--output_file',
        type=str,
        default='fid_stat/fid_stats_cifar10_train.npz',
        help='path for where to store the statistics')

    opt = parser.parse_args()
    print(opt)
    return opt


def main():
    args = parse_args()

    ########
    # PATHS
    ########
    data_path = args.data_path
    output_path = args.output_file
    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path)  # download inception if necessary
    print("ok")

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" ", flush=True)
    image_list = glob.glob(os.path.join(data_path, '*.jpg'))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")


if __name__ == '__main__':
    main()
