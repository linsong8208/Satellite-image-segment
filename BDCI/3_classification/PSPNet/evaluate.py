from __future__ import print_function
import argparse
import os
import time

import tensorflow as tf
import numpy as np

from pspnet_model import PSPNet
from image_reader import ImageReader


class Tools(object):

    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def print_info(info):
        print("{} {}".format(time.strftime("%H:%M:%S", time.localtime()), info))
        pass

    pass


class Evaluate(object):

    def __init__(self, num_classes=5, ignore_label=255, is_flip=False, is_measure_time=True,
                 log_dir="./model_bdci_5", model_name="model.ckpt", save_dir="./output_bdci_5",
                 data_dir="data/bdci/train", eval_list="data/bdci/train_list.txt"):

        self.save_dir = Tools.new_dir(save_dir)
        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.img_mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        self.input_size = [1024, 2048]
        self.num_classes = num_classes

        self.ignore_label = ignore_label  # Don't care label
        self.num_steps = 500  # numbers of image in validation set
        self.time_list = []

        self.is_flip = is_flip
        self.is_measure_time = is_measure_time

        self.data_dir = data_dir
        self.data_eval_list = eval_list
        pass

    def calculate_time(self, sess, net):
        start = time.time()
        sess.run(net.layers['data'])
        data_time = time.time() - start

        start = time.time()
        sess.run(net.layers['conv6'])
        total_time = time.time() - start

        inference_time = total_time - data_time
        self.time_list.append(inference_time)
        Tools.print_info('average inference time: {}'.format(np.mean(self.time_list)))
        pass

    def run(self):
        coord = tf.train.Coordinator()

        tf.reset_default_graph()
        with tf.name_scope("create_inputs"):
            reader = ImageReader(self.data_dir, self.data_eval_list, self.input_size, None, None,
                                 self.ignore_label, self.img_mean, coord)
            image, label = reader.image, reader.label
        image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)

        # Create network.
        net = PSPNet({'data': image_batch}, is_training=False, num_classes=self.num_classes)

        with tf.variable_scope('', reuse=True):
            flipped_img = tf.image.flip_left_right(image)
            flipped_img = tf.expand_dims(flipped_img, dim=0)
            net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=self.num_classes)

        # Predictions.
        raw_output = net.layers['conv6']

        if self.is_flip:
            flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
            flipped_output = tf.expand_dims(flipped_output, dim=0)
            raw_output = tf.add_n([raw_output, flipped_output])

        raw_output_up = tf.image.resize_bilinear(raw_output, size=self.input_size, align_corners=True)
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        predictions_op = tf.expand_dims(raw_output_up, dim=3)

        # mIoU
        predictions_flatten = tf.reshape(predictions_op, [-1, ])
        raw_gt = tf.reshape(label_batch, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        predictions = tf.gather(predictions_flatten, indices)

        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(predictions, gt, num_classes=self.num_classes)

        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        sess.run(init)
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=tf.global_variables())
            loader.restore(sess, ckpt.model_checkpoint_path)
            Tools.print_info("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            Tools.print_info('No checkpoint file found.')

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for step in range(self.num_steps):
            predictions_result, _ = sess.run([predictions, update_op])
            if step > 0 and self.is_measure_time:
                self.calculate_time(sess, net)
            if step % 1 == 0:
                Tools.print_info('Finish {0}/{1} mIoU: {2}'.format(step, self.num_steps, sess.run(mIoU)))

        Tools.print_info('step {0} mIoU: {1}'.format(self.num_steps, sess.run(mIoU)))

        coord.request_stop()
        coord.join(threads)
        pass

    pass

if __name__ == '__main__':
    Evaluate().run()
