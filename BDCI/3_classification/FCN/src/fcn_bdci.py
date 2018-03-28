# encoding: UTF-8
import os
import time
from PIL import Image
import argparse
import collections
import numpy as np
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tcl


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    pass


# 切割数据
class PreData:

    @staticmethod
    def product_train_image(src_images, src_labels, result_root_path, stripe, crop_size):
        result_data_path = Tools.new_dir(os.path.join(result_root_path, "image"))
        result_label_path = Tools.new_dir(os.path.join(result_root_path, "label"))

        if len(glob(os.path.join(result_data_path, "*.*"))):
            Tools.print_info("train data existed...")
        else:
            for index, image in enumerate(src_images):
                new_image = np.copy(np.asarray(Image.open(image)))
                new_label = np.copy(np.asarray(Image.open(src_labels[index])))
                for crop_x in range(crop_size//2, len(new_image) - crop_size//2, stripe):
                    for crop_y in range(crop_size//2, len(new_image) - crop_size//2, stripe):
                        crop_image = new_image[crop_x: crop_size + crop_x, crop_y: crop_size + crop_y, :]
                        crop_label = new_label[crop_x: crop_size + crop_x, crop_y: crop_size + crop_y]
                        crop_name = "{}_{}_{}_{}_{}.bmp".format(np.random.randint(0, 1000000), index, crop_x, crop_y, stripe)
                        im = Image.fromarray(crop_image).convert("RGB").resize((crop_size, crop_size))
                        im.save(os.path.join(result_data_path, crop_name))
                        im = Image.fromarray(crop_label).convert("L").resize((crop_size, crop_size))
                        im.save(os.path.join(result_label_path, crop_name))
                    pass
                pass
        return result_data_path, result_label_path

    @staticmethod
    def product_test_image(src_images, result_root_path, cover, crop_size):
        Tools.new_dir(result_root_path)

        if len(glob(os.path.join(result_root_path, "*.*"))):
            Tools.print_info("test data existed...")
        else:
            for index, image in enumerate(src_images):
                new_image = np.copy(np.asarray(Image.open(image)))
                for crop_x in range(crop_size//2, len(new_image) - crop_size//2, crop_size - cover):
                    for crop_y in range(crop_size//2, len(new_image) - crop_size//2, crop_size - cover):
                        crop_image = new_image[crop_x: crop_size + crop_x, crop_y: crop_size + crop_y, :]
                        crop_name = "{}_{}_{}_{}_{}_{}.bmp".format(index, crop_x, crop_y, crop_size + crop_x, crop_size + crop_y, cover)
                        im = Image.fromarray(crop_image).convert("RGB").resize((crop_size, crop_size))
                        im.save(os.path.join(result_root_path, crop_name))
                    pass
                pass
        return result_root_path

    @staticmethod
    def train():
        return PreData.product_train_image(src_images=["data/CCF-training/1.png", "data/CCF-training/2.png"],
                                           src_labels=["data/CCF-training/1_class.png",
                                                       "data/CCF-training/2_class.png"],
                                           result_root_path="data/train_fcn", stripe=150, crop_size=256)

    @staticmethod
    def test():
        return PreData.product_test_image(src_images=["data/CCF-testing/1.png",
                                                      "data/CCF-testing/2.png", "data/CCF-testing/3.png"],
                                          result_root_path="data/test_fcn", cover=30, crop_size=256)

    pass


# 分两种使用情况：
# 1.训练
# 2.测试
class Data:

    def __init__(self, images_path, batch_size, type_number, image_size, image_channel):
        self.batch_size = batch_size
        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel

        self.images = glob(os.path.join(images_path, "*.bmp"))
        self.labels = [image.replace("image", "label") for image in self.images]

        # 训练相关的
        self.batch_offset = 0
        self.now_epochs = 0

        # 测试的batch number
        self.batch_number = int(np.ceil(len(self.images) / self.batch_size))
        pass

    # 训练时使用
    def next_batch(self):
        start = self.batch_offset
        self.batch_offset += self.batch_size

        if self.batch_offset > len(self.images):
            # Finished epoch
            self.now_epochs += 1
            Tools.print_info("Epochs completed: {}".format(self.now_epochs))

            # Shuffle the data
            # perm = np.arange(len(self.images))
            # np.random.shuffle(perm)
            # self.images = self.images[perm]
            # self.labels = self.labels[perm]

            # Start next epoch
            start = 0
            self.batch_offset = self.batch_size

        end = self.batch_offset
        return self.deal_image(self.images[start:end]), self.deal_label(self.labels[start:end])

    # 测试时使用
    # 测试图片个数必须大于batch size
    def next_batch_test(self, now_batch):
        start = len(self.images) - self.batch_size if now_batch == self.batch_number - 1 else now_batch * self.batch_size
        end = start + self.batch_size
        return self.deal_image(self.images[start: end]), self.images[start: end]

    @staticmethod
    def deal_image(images):
        return [np.asarray(Image.open(image)) / 255.0 for image in images]

    @staticmethod
    def deal_label(labels):
        return [np.asarray(Image.open(label), dtype=np.int32) for label in labels]

    pass


class CNNNet:
    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    # 网络
    def cnn_5(self, input_op, **kw):
        weight_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, self._image_channel, 64], stddev=5e-2))
        kernel_1 = tf.nn.conv2d(input_op, weight_1, [1, 1, 1, 1], padding="SAME")
        bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
        conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding="SAME")
        norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        weight_2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 128], stddev=5e-2))
        kernel_2 = tf.nn.conv2d(norm_1, weight_2, [1, 1, 1, 1], padding="SAME")
        bias_2 = tf.Variable(tf.constant(0.1, shape=[128]))
        conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
        norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding="SAME")

        weight_23 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
        kernel_23 = tf.nn.conv2d(pool_2, weight_23, [1, 2, 2, 1], padding="SAME")
        bias_23 = tf.Variable(tf.constant(0.1, shape=[256]))
        conv_23 = tf.nn.relu(tf.nn.bias_add(kernel_23, bias_23))
        norm_23 = tf.nn.lrn(conv_23, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_23 = tf.nn.max_pool(norm_23, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="SAME")

        reshape = tf.reshape(pool_23, [self._batch_size, -1])
        dim = reshape.get_shape()[1].value

        weight_4 = tf.Variable(tf.truncated_normal(shape=[dim, 192 * 2], stddev=0.04))
        bias_4 = tf.Variable(tf.constant(0.1, shape=[192 * 2]))
        local_4 = tf.nn.relu(tf.matmul(reshape, weight_4) + bias_4)

        weight_5 = tf.Variable(tf.truncated_normal(shape=[192 * 2, self._type_number], stddev=1 / 192.0))
        bias_5 = tf.Variable(tf.constant(0.0, shape=[self._type_number]))
        logits = tf.add(tf.matmul(local_4, weight_5), bias_5)

        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)

        return logits, softmax, prediction

    pass


class AlexNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    def alex_net(self, input_op, **kw):
        #  256 X 256 X 3
        with tf.name_scope("conv1") as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, self._image_channel, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(input=input_op, filter=kernel, strides=[1, 4, 4, 1], padding="SAME")  # 64 X 64 X 64
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32))
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn1")
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")  # 31 X 31 X 64

        with tf.name_scope("conv2") as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")  # 31 X 31 X 192
            biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
            conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn2")
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")  # 15 X 15 X 192

        with tf.name_scope("conv3") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")  # 15 X 15 X 384
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
            conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope("conv4") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")  # 15 X 15 X 256
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32))
            conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope("conv5") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")  # 15 X 15 X 256
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32))
            conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")  # 7 X 7 X 256

        dim = pool3.get_shape()[1].value * pool3.get_shape()[2].value * pool3.get_shape()[3].value
        reshape = tf.reshape(pool3, [-1, dim])

        with tf.name_scope("fc1") as scope:
            weights = tf.Variable(tf.truncated_normal([dim, 384], dtype=tf.float32, stddev=1e-1))
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
            fc1 = tf.nn.relu(tf.add(tf.matmul(reshape, weights), biases), name=scope)  # dim X 384

        with tf.name_scope("fc2") as scope:
            weights = tf.Variable(tf.truncated_normal([384, 192], dtype=tf.float32, stddev=1e-1))
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
            fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights), biases), name=scope)  # 384 X 192

        with tf.name_scope("fc3") as scope:
            weights = tf.Variable(tf.truncated_normal([192, self._type_number], dtype=tf.float32, stddev=1e-1))
            biases = tf.Variable(tf.constant(0.0, shape=[self._type_number], dtype=tf.float32))
            logits = tf.add(tf.matmul(fc2, weights), biases, name=scope)  # 192 X number_type

        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)
        return logits, softmax, prediction

    pass


class VGGNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    # 网络
    def vgg_16(self, input_op):
        first_out = 32

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_3 = self._conv_op(conv_3_2, "conv_3_3", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_3 = self._max_pool_op(conv_3_3, "pool_3", 2, 2, stripe_height=2, stripe_width=2)

        conv_4_1 = self._conv_op(pool_3, "conv_4_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_2 = self._conv_op(conv_4_1, "conv_4_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_3 = self._conv_op(conv_4_2, "conv_4_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        pool_4 = self._max_pool_op(conv_4_3, "pool_4", 2, 2, stripe_height=2, stripe_width=2)

        conv_5_1 = self._conv_op(pool_4, "conv_5_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_5_2 = self._conv_op(conv_5_1, "conv_5_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_5_3 = self._conv_op(conv_5_2, "conv_5_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)

        return conv_5_3

    # 网络
    def vgg_12(self, input_op):
        first_out = 16

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_3 = self._max_pool_op(conv_3_2, "pool_3", 2, 2, stripe_height=2, stripe_width=2)

        conv_4_1 = self._conv_op(pool_3, "conv_4_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_2 = self._conv_op(conv_4_1, "conv_4_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_3 = self._conv_op(conv_4_2, "conv_4_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)

        return conv_4_3

    # 网络
    def vgg_10(self, input_op):
        first_out = 8

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_3 = self._conv_op(conv_3_2, "conv_3_3", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)

        return conv_3_3

    # 创建卷积层
    @staticmethod
    def _conv_op(input_op, name, kernel_height, kernel_width, n_out, stripe_height, stripe_width):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name=name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[kernel_height, kernel_width, n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, filter=kernel, strides=(1, stripe_height, stripe_width, 1), padding="SAME")
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
            activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
            return activation
        pass

    # 最大池化层
    @staticmethod
    def _max_pool_op(input_op, name, kernel_height, kernel_width, stripe_height, stripe_width):
        return tf.nn.max_pool(input_op, ksize=[1, kernel_height, kernel_width, 1],
                              strides=[1, stripe_height, stripe_width, 1], padding="SAME", name=name)

    pass


class InceptionNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    @staticmethod
    def _inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection="moving_vars"):
        batch_norm_params = {
            "decay": 0.9997,
            "epsilon": 0.001,
            "updates_collections": tf.GraphKeys.UPDATE_OPS,
            "variables_collections": {
                "beta": None,
                "gamma": None,
                "moving_mean": [batch_norm_var_collection],
                "moving_variance": [batch_norm_var_collection]
            }
        }

        # slim.arg_scope 可以给函数的参数自动赋予某些默认值
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                normalizer_params=batch_norm_params) as sc:
                return sc
        pass

    @staticmethod
    def _inception_v3_base(inputs, scope):
        end_points = {}

        # 299
        with tf.variable_scope(scope, values=[inputs]):

            # 非Inception Module:5个卷积层和2个最大池化层
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding="VALID"):

                with tf.variable_scope("group_non"):
                    net = slim.conv2d(inputs, 32, [3, 3], stride=2)  # 149 X 149 X 32
                    net = slim.conv2d(net, 32, [3, 3])  # 147 X 147 X 32
                    net = slim.conv2d(net, 64, [3, 3], padding="SAME")  # 147 X 147 X 64
                    net = slim.max_pool2d(net, [3, 3], stride=2)  # 73 X 73 X 64
                    net = slim.conv2d(net, 80, [1, 1])  # 73 X 73 X 80
                    net = slim.conv2d(net, 192, [3, 3])  # 71 X 71 X 192
                    net = slim.max_pool2d(net, [3, 3], stride=2)  # 35 X 35 X 192
                pass

            # 非Inception Module的结果
            end_points["group_non"] = net

            # 共有3个连续的Inception模块组
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding="SAME"):

                # 第1个模块组包含了3个Inception Module
                with tf.variable_scope("group_1a"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 64, [1, 1])  # 35 X 35 X 64
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 48, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5])  # 35 X 35 X 64
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 64, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3])  # 35 X 35 X 96
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 32, [1, 1])  # 35 X 35 X 32
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 35 X 35 X 256
                    pass

                with tf.variable_scope("group_1b"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 64, [1, 1])  # 35 X 35 X 64
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 48, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5])  # 35 X 35 X 64
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 64, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3])  # 35 X 35 X 96
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1])  # 35 X 35 X 64
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 35 X 35 X 288
                    pass

                with tf.variable_scope("group_1c"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 64, [1, 1])  # 35 X 35 X 64
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 48, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5])  # 35 X 35 X 64
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 64, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3])  # 35 X 35 X 96
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1])  # 35 X 35 X 64
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 35 X 35 X 288
                    pass

                # 第1个模块组的结果
                end_points["group_1c"] = net  # 35 X 35 X 288

                # 第2个模块组包含了5个Inception Module
                with tf.variable_scope("group_2a"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding="VALID")  # 17 X 17 X 384
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 64, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3])
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding="VALID")  # 17 X 17 X 96
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding="VALID")  # 17 X 17 X 288
                    net = tf.concat([branch_0, branch_1, branch_2], axis=3)  # 17 X 17 X 768
                    pass

                with tf.variable_scope("group_2b"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 192, [1, 1], padding="VALID")  # 17 X 17 X 192
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 128, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 128, [1, 7])
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1])  # 17 X 17 X 192
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 128, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 128, [1, 7])
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7])  # 17 X 17 X 192
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1])  # 17 X 17 X 192
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 17 X 17 X 768
                    pass

                with tf.variable_scope("group_2c"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 192, [1, 1])  # 17 X 17 X 192
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 160, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7])
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1])  # 17 X 17 X 192
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 160, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7])
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7])  # 17 X 17 X 192
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1])  # 17 X 17 X 192
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 17 X 17 X 768
                    pass

                with tf.variable_scope("group_2d"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 192, [1, 1])  # 17 X 17 X 192
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 160, [1, 1],)
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7])
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1])  # 17 X 17 X 192
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 160, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7])
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7])  # 17 X 17 X 192
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1])  # 17 X 17 X 192
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 17 X 17 X 768
                    pass

                with tf.variable_scope("group_2e"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 192, [1, 1])  # 17 X 17 X 192
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 160, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7])
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1])  # 17 X 17 X 192
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 160, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7])
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7])  # 17 X 17 X 192
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1])  # 17 X 17 X 192
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 17 X 17 X 768
                    pass

                # 第2个模块组的结果
                end_points["group_2e"] = net  # 17 X 17 X 768

                # 第3个模块组包含了3个Inception Module
                with tf.variable_scope("group_3a"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 192, [1, 1])
                        branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding="VALID")  # 8 X 8 X 320
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 192, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7])
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1])
                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding="VALID")  # 8 X 8 X 192
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding="VALID")  # 8 X 8 X 768
                    net = tf.concat([branch_0, branch_1, branch_2], axis=3)  # 8 X 8 X 1280
                    pass

                with tf.variable_scope("group_3b"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 320, [1, 1])  # 8 X 8 X 320
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 384, [1, 1])
                        branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3]),
                                              slim.conv2d(branch_1, 384, [3, 1])], 3)  # 8 X 8 X 768
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 448, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3])
                        branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3]),
                                              slim.conv2d(branch_2, 384, [3, 1])], 3)  # 8 X 8 X 768
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1])  # 8 X 8 X 192
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 8 X 8 X 2048
                    pass

                with tf.variable_scope("group_3c"):
                    with tf.variable_scope("branch_0"):
                        branch_0 = slim.conv2d(net, 320, [1, 1])  # 8 X 8 X 320
                    with tf.variable_scope("branch_1"):
                        branch_1 = slim.conv2d(net, 384, [1, 1])
                        branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3]),
                                              slim.conv2d(branch_1, 384, [3, 1])], 3)  # 8 X 8 X 768
                    with tf.variable_scope("branch_2"):
                        branch_2 = slim.conv2d(net, 448, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3])
                        branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3]),
                                              slim.conv2d(branch_2, 384, [3, 1])], 3)  # 8 X 8 X 768
                    with tf.variable_scope("branch_3"):
                        branch_3 = slim.avg_pool2d(net, [3, 3])
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1])  # 8 X 8 X 192
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)  # 8 X 8 X 2048
                    pass

                # 第3个模块组的结果
                end_points["group_3c"] = net  # 8 X 8 X 2048

            pass

        return net, end_points

    # 网络
    # keep_prob=0.8
    def inception_v3(self, input_op, is_training=True, reuse=None, **kw):

        with tf.variable_scope("inception_v3", values=[input_op, self._type_number], reuse=reuse) as scope:

            # slim.arg_scope 可以给函数的参数自动赋予某些默认值
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net, end_points = self._inception_v3_base(inputs=input_op, scope=scope)

                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding="SAME"):

                    # 辅助分类节点：Auxiliary Logits，将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类
                    # 结果中，相当于做了模型的融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化。
                    with tf.variable_scope("aux_logits"):
                        aux_logits = end_points["group_2e"]  # 17 X 17 X 768
                        aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding="VALID")  # 5 X 5 X 768
                        aux_logits = slim.conv2d(aux_logits, 128, [1, 1])  # 5 X 5 X 128
                        # 299
                        # aux_logits = slim.conv2d(aux_logits, 768, [5, 5], padding="VALID",  # 1 X 1 X 768
                        #                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
                        # 256
                        aux_logits = slim.conv2d(aux_logits, 768, [4, 4], padding="VALID",  # 1 X 1 X 768
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
                        aux_logits = slim.conv2d(aux_logits, self._type_number, [1, 1],
                                                 activation_fn=None, normalizer_fn=None,   # 1 X 1 X num_classes
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
                        aux_logits = tf.squeeze(aux_logits, [1, 2])  # num_classes
                        end_points["aux_logits"] = aux_logits
                        pass

                    # 正常的Logits
                    with tf.variable_scope("logits"):
                        # 299
                        # net = slim.avg_pool2d(net, [8, 8], padding="VALID")  # 1 X 1 X 2048
                        # 256
                        net = slim.avg_pool2d(net, [6, 6], padding="VALID")  # 1 X 1 X 2048
                        net = slim.dropout(net, keep_prob=kw["keep_prob"])
                        end_points["pre_logits"] = net
                        logits = slim.conv2d(net, self._type_number, [1, 1],   # 1 X 1 X num_classes
                                             activation_fn=None, normalizer_fn=None)
                        logits = tf.squeeze(logits, [1, 2])  # num_classes
                        end_points["logits"] = logits
                        pass

                    pass

                softmax = slim.softmax(logits)
                end_points["softmax"] = softmax
                end_points["prediction"] = tf.argmax(softmax, 1)

            pass
        return logits, end_points["softmax"], end_points["prediction"]

    pass


class ResNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    # 残差基本单元
    class _Block(collections.namedtuple("Block", ["scope", "unit_fn", "args"])):
        pass

    # 卷积层
    @staticmethod
    def _bottleneck_conv2d(inputs, num_outputs, kernel_size, stride):
        if stride == 1:
            padding = "SAME"
        else:
            padding = "VALID"
            padding_begin = (kernel_size - 1) // 2
            padding_end = kernel_size - 1 - padding_begin
            inputs = tf.pad(inputs, [[0, 0], [padding_begin, padding_end], [padding_begin, padding_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding=padding)

    # Block中unit_fn的实现
    def _bottleneck(self, inputs, depth, depth_bottleneck, stride, scope=None):
        with tf.variable_scope(scope, "bottleneck_v2", [inputs]):
            pre_activation = slim.batch_norm(inputs, activation_fn=tf.nn.relu)

            # 定义直连的x：将两者的通道数和空间尺寸处理成一致
            depth_in = inputs.get_shape()[-1].value
            if depth == depth_in:
                # 输入和输出通道数相同的情况
                shortcut = inputs if stride == 1 else slim.max_pool2d(inputs, kernel_size=[1, 1], stride=stride,
                                                                      padding="SAME")
            else:
                # 输入和输出通道数不相同的情况
                shortcut = slim.conv2d(pre_activation, depth, [1, 1], stride=stride, normalizer_fn=None,
                                       activation_fn=None)

            residual = slim.conv2d(pre_activation, depth_bottleneck, kernel_size=[1, 1], stride=1)
            residual = self._bottleneck_conv2d(residual, depth_bottleneck, kernel_size=3, stride=stride)
            residual = slim.conv2d(residual, depth, kernel_size=[1, 1], stride=1, activation_fn=None)

            output = shortcut + residual
        return output

    # 堆叠Blocks
    @staticmethod
    # @slim.add_arg_scope  # Decorates a function with args so it can be used within an arg_scope
    def _stack_blocks_dense(net, blocks):
        for block in blocks:
            with tf.variable_scope(block.scope, "block", [net]) as sc:
                for i, unit in enumerate(block.args):
                    with tf.variable_scope("unit_%d" % (i + 1), values=[net]):
                        depth, depth_bottleneck, stride = unit
                        net = block.unit_fn(net, depth=depth, depth_bottleneck=depth_bottleneck, stride=stride)
                pass
        return net

    # 构造整个网络
    def _resnet_v2(self, inputs, blocks, global_pool=True, include_root_block=True):
        end_points = {}

        net = inputs
        # 是否加上ResNet网络最前面通常使用的7X7卷积和最大池化
        if include_root_block:
            net = slim.conv2d(net, 64, [7, 7], stride=2, padding="SAME", activation_fn=None, normalizer_fn=None)
            net = slim.max_pool2d(net, [3, 3], stride=2, padding="SAME")

        # 构建ResNet网络
        net = self._stack_blocks_dense(net, blocks)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        # 全局平均池化层
        if global_pool:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True)

        # 分类
        logits = slim.conv2d(net, self._type_number, kernel_size=[1, 1], activation_fn=None, normalizer_fn=None)
        logits = tf.squeeze(logits, [1, 2])  # batch_size X type_number

        softmax = slim.softmax(logits)
        end_points["softmax"] = softmax
        end_points["prediction"] = tf.argmax(softmax, 1)
        return logits, end_points["softmax"], end_points["prediction"]

    # 通用scope
    @staticmethod
    def _resnet_arg_scope(is_training=True, weight_decay=0.0001, bn_decay=0.997, bn_epsilon=1e-5, bn_scale=True):
        batch_norm_params = {
            "is_training": is_training,
            "decay": bn_decay,
            "epsilon": bn_epsilon,
            "scale": bn_scale,
            "updates_collections": tf.GraphKeys.UPDATE_OPS
        }
        conv2d_params = {
            "weights_regularizer": slim.l2_regularizer(weight_decay),
            "weights_initializer": slim.variance_scaling_initializer(),
            "activation_fn": tf.nn.relu,
            "normalizer_fn": slim.batch_norm,
            "normalizer_params": batch_norm_params
        }
        with slim.arg_scope([slim.conv2d], **conv2d_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc
        pass

    def resnet_v2_50(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    def resnet_v2_101(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    def resnet_v2_152(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    def resnet_v2_200(self, input_op, **kw):
        blocks = [
            self._Block("block1", self._bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            self._Block("block2", self._bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
            self._Block("block3", self._bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            self._Block("block4", self._bottleneck, [(2048, 512, 1)] * 3)
        ]
        with slim.arg_scope(self._resnet_arg_scope()):
            return self._resnet_v2(input_op, blocks, global_pool=True, include_root_block=True)
        pass

    pass


class FCN_VGGNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        self._vgg_net = VGGNet(self._type_number, self._image_size, self._image_channel, self._batch_size)
        pass

    # 网络
    # keep_prob=0.7
    def fcn_vgg_16(self, input_op, **kw):

        Tools.print_info("begin to build VGG net")
        input_op = self._vgg_net.vgg_16(input_op)

        Tools.print_info("begin to build FCN net")
        first_out = 256 * 2

        fcn_conv_1 = self._conv_op(input_op, "fcn_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        dropout_1 = tf.nn.dropout(fcn_conv_1, keep_prob=kw["keep_prob"])

        fcn_conv_2 = self._conv_op(dropout_1, "fcn_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        dropout_2 = tf.nn.dropout(fcn_conv_2, keep_prob=kw["keep_prob"])

        fcn_conv_3 = self._conv_op(dropout_2, "fcn_3", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        dropout_3 = tf.nn.dropout(fcn_conv_3, keep_prob=kw["keep_prob"])

        fcn_conv_4 = self._conv_op(dropout_3, "fcn_4", 3, 3, n_out=self._type_number, stripe_height=1, stripe_width=1)

        # transpose conv
        conv_transpose_1 = self._conv_transpose_op(fcn_conv_4, "conv_transpose_1", 3, 3,
                                                   n_out=self._type_number, stripe_height=2, stripe_width=2)
        dropout_1 = tf.nn.dropout(conv_transpose_1, keep_prob=kw["keep_prob"])

        conv_transpose_2 = self._conv_transpose_op(dropout_1, "conv_transpose_2", 3, 3,
                                                   n_out=self._type_number, stripe_height=2, stripe_width=2)
        dropout_2 = tf.nn.dropout(conv_transpose_2, keep_prob=kw["keep_prob"])

        logits = self._conv_transpose_op(dropout_2, "conv_transpose_3", 3, 3, n_out=self._type_number,
                                         stripe_height=4, stripe_width=4, has_activation=False)

        softmax = tf.nn.softmax(logits=logits, name="softmax")
        prediction = tf.argmax(logits, axis=3, name="prediction")

        return logits, softmax, prediction

    # 网络
    # keep_prob=0.7
    def fcn_vgg_12(self, input_op, **kw):
        Tools.print_info("begin to build VGG net")
        input_op = self._vgg_net.vgg_12(input_op)

        Tools.print_info("begin to build FCN net")
        first_out = 128 * 2

        fcn_conv_1 = self._conv_op(input_op, "fcn_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        dropout_1 = tf.nn.dropout(fcn_conv_1, keep_prob=kw["keep_prob"])

        fcn_conv_2 = self._conv_op(dropout_1, "fcn_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        dropout_2 = tf.nn.dropout(fcn_conv_2, keep_prob=kw["keep_prob"])

        fcn_conv_3 = self._conv_op(dropout_2, "fcn_3", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        dropout_3 = tf.nn.dropout(fcn_conv_3, keep_prob=kw["keep_prob"])

        fcn_conv_4 = self._conv_op(dropout_3, "fcn_4", 3, 3, n_out=self._type_number, stripe_height=1, stripe_width=1)

        # transpose conv
        conv_transpose_1 = self._conv_transpose_op(fcn_conv_4, "conv_transpose_1", 3, 3,
                                                   n_out=self._type_number, stripe_height=2, stripe_width=2)
        dropout_1 = tf.nn.dropout(conv_transpose_1, keep_prob=kw["keep_prob"])

        conv_transpose_2 = self._conv_transpose_op(dropout_1, "conv_transpose_2", 3, 3,
                                                   n_out=self._type_number, stripe_height=2, stripe_width=2)
        dropout_2 = tf.nn.dropout(conv_transpose_2, keep_prob=kw["keep_prob"])

        logits = self._conv_transpose_op(dropout_2, "conv_transpose_3", 3, 3, n_out=self._type_number,
                                         stripe_height=2, stripe_width=2, has_activation=False)

        softmax = tf.nn.softmax(logits=logits, name="softmax")
        prediction = tf.argmax(logits, axis=3, name="prediction")

        return logits, softmax, prediction

    # 网络
    # keep_prob=0.7
    def fcn_vgg_10(self, input_op, **kw):
        Tools.print_info("begin to build VGG net")
        input_op = self._vgg_net.vgg_10(input_op)

        Tools.print_info("begin to build FCN net")
        first_out = 64 * 2

        fcn_conv_1 = self._conv_op(input_op, "fcn_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        dropout_1 = tf.nn.dropout(fcn_conv_1, keep_prob=kw["keep_prob"])

        fcn_conv_2 = self._conv_op(dropout_1, "fcn_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        dropout_2 = tf.nn.dropout(fcn_conv_2, keep_prob=kw["keep_prob"])

        fcn_conv_3 = self._conv_op(dropout_2, "fcn_3", 3, 3, n_out=self._type_number, stripe_height=1, stripe_width=1)

        # transpose conv
        conv_transpose_1 = self._conv_transpose_op(fcn_conv_3, "conv_transpose_1", 3, 3,
                                                   n_out=self._type_number, stripe_height=2, stripe_width=2)
        dropout_1 = tf.nn.dropout(conv_transpose_1, keep_prob=kw["keep_prob"])

        logits = self._conv_transpose_op(dropout_1, "conv_transpose_2", 3, 3, n_out=self._type_number,
                                         stripe_height=2, stripe_width=2, has_activation=False)

        softmax = tf.nn.softmax(logits=logits, name="softmax")
        prediction = tf.argmax(logits, axis=3, name="prediction")

        return logits, softmax, prediction

    # 创建卷积层
    @staticmethod
    def _conv_transpose_op(input_op, name, kernel_height, kernel_width, n_out, stripe_height, stripe_width,
                           has_activation=True):
        batch_size = input_op.get_shape()[0].value
        height = input_op.get_shape()[1].value * stripe_height
        width = input_op.get_shape()[2].value * stripe_width
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name=name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[kernel_height, kernel_width, n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer_conv2d())
            outputs_shape = [batch_size, height, width, n_out]
            conv = tf.nn.conv2d_transpose(input_op, filter=kernel, output_shape=outputs_shape, strides=(1, stripe_height, stripe_width, 1))
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
            if not has_activation:
                return tf.nn.bias_add(conv, biases, name=scope)
            else:
                return tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        pass

    # 创建卷积层
    @staticmethod
    def _conv_op(input_op, name, kernel_height, kernel_width, n_out, stripe_height, stripe_width, has_activation=True):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name=name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[kernel_height, kernel_width, n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, filter=kernel, strides=(1, stripe_height, stripe_width, 1), padding="SAME")
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
            if not has_activation:
                return tf.nn.bias_add(conv, biases, name=scope)
            else:
                return tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        pass

    # 最大池化层
    @staticmethod
    def _max_pool_op(input_op, name, kernel_height, kernel_width, stripe_height, stripe_width):
        return tf.nn.max_pool(input_op, ksize=[1, kernel_height, kernel_width, 1],
                              strides=[1, stripe_height, stripe_width, 1], padding="SAME", name=name)

    pass


class FCN_VGGNet16:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size

        self.vgg_weights = np.load('pretrained/vgg16_weights.npz')
        pass

    def fcn_vgg_16(self, input_op, **kw):
        images = self.preprocess(input_op)
        conv1_1 = self._conv_layer(images, 'conv1_1_W', 'conv1_1_b', name='conv1_1')
        conv1_2 = self._conv_layer(conv1_1, 'conv1_2_W', 'conv1_2_b', name='conv1_2')
        pool1 = self._max_pool_layer(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool1')

        conv2_1 = self._conv_layer(pool1, 'conv2_1_W', 'conv2_1_b', name='conv2_1')
        conv2_2 = self._conv_layer(conv2_1, 'conv2_2_W', 'conv2_2_b', name='conv2_2')
        pool2 = self._max_pool_layer(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool2')

        conv3_1 = self._conv_layer(pool2, 'conv3_1_W', 'conv3_1_b', name='conv3_1')
        conv3_2 = self._conv_layer(conv3_1, 'conv3_2_W', 'conv3_2_b', name='conv3_2')
        conv3_3 = self._conv_layer(conv3_2, 'conv3_3_W', 'conv3_3_b', name='conv3_3')
        pool3 = self._max_pool_layer(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool3')

        conv4_1 = self._conv_layer(pool3, 'conv4_1_W', 'conv4_1_b', name='conv4_1')
        conv4_2 = self._conv_layer(conv4_1, 'conv4_2_W', 'conv4_2_b', name='conv4_2')
        conv4_3 = self._conv_layer(conv4_2, 'conv4_3_W', 'conv4_3_b', name='conv4_3')
        pool4 = self._max_pool_layer(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool4')

        conv5_1 = self._conv_layer(pool4, 'conv5_1_W', 'conv5_1_b', name='conv5_1')
        conv5_2 = self._conv_layer(conv5_1, 'conv5_2_W', 'conv5_2_b', name='conv5_2')
        conv5_3 = self._conv_layer(conv5_2, 'conv5_3_W', 'conv5_3_b', name='conv5_3')
        pool5 = self._max_pool_layer(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool5')

        fc_1 = self._fully_collected_layer(pool5, 'fc_1', kw["keep_prob"], num_classes=self._type_number)
        fc_2 = self._fully_collected_layer(fc_1, 'fc_2', kw["keep_prob"], num_classes=self._type_number)
        fc_3 = self._fully_collected_layer(fc_2, 'fc_3', kw["keep_prob"], num_classes=self._type_number)

        img_shape = tf.shape(input_op)
        dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self._image_channel])
        upsample_1 = self._upsample_layer(fc_3, dconv3_shape, self._type_number, 'upsample_1', 32, num_classes=self._type_number)

        skip_1 = self._skip_layer_connection(pool4, 'skip_1', 512, stddev=0.00001, num_classes=self._type_number)
        upsample_2 = self._upsample_layer(skip_1, dconv3_shape, self._type_number, 'upsample_2', 16, num_classes=self._type_number)

        skip_2 = self._skip_layer_connection(pool3, 'skip_2', 256, stddev=0.0001, num_classes=self._type_number)
        upsample_3 = self._upsample_layer(skip_2, dconv3_shape, self._type_number, 'upsample_3', 8, num_classes=self._type_number)

        logits = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))
        softmax = tf.nn.softmax(logits=logits, name="softmax")
        prediction = tf.argmax(logits, axis=3, name="prediction")
        return logits, softmax, prediction

    @staticmethod
    def _skip_layer_connection(parent, name, num_input_layers, num_classes, stddev=0.0005):
        with tf.variable_scope(name) as scope:
            initial = tf.truncated_normal([1, 1, num_input_layers, num_classes], stddev=stddev)
            kernel = tf.get_variable('kernel', initializer=initial)
            conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')

            bias_init = tf.constant(0.0, shape=[num_classes])
            bias = tf.get_variable('bias', initializer=bias_init)
            skip_layer = tf.nn.bias_add(conv, bias)

            return skip_layer

    def _upsample_layer(self, bottom, shape, n_channels, name, upscale_factor, num_classes):
        kernel_size = 2 * upscale_factor - upscale_factor % 2
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            output_shape = [shape[0], shape[1], shape[2], num_classes]
            filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
            weights = self._get_bilinear_filter(filter_shape, upscale_factor)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')

            bias_init = tf.constant(0.0, shape=[num_classes])
            bias = tf.get_variable('bias', initializer=bias_init)
            dconv_with_bias = tf.nn.bias_add(deconv, bias)

        return dconv_with_bias

    @staticmethod
    def _get_bilinear_filter(filter_shape, upscale_factor):
        kernel_size = filter_shape[1]
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights, dtype=tf.float32)

        bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init, shape=weights.shape)
        return bilinear_weights

    def _fully_collected_layer(self, parent, name, dropout, num_classes):
        with tf.variable_scope(name) as scope:
            if name == 'fc_1':
                kernel = self._reshape_fc_weights('fc6_W', [7, 7, 512, 4096])
                conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')
                bias = self._get_bias('fc6_b')
                output = tf.nn.bias_add(conv, bias)
                output = tf.nn.relu(output, name=scope.name)
                return tf.nn.dropout(output, dropout)

            if name == 'fc_2':
                kernel = self._reshape_fc_weights('fc7_W', [1, 1, 4096, 4096])
                conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')
                bias = self._get_bias('fc7_b')
                output = tf.nn.bias_add(conv, bias)
                output = tf.nn.relu(output, name=scope.name)
                return tf.nn.dropout(output, dropout)

            if name == 'fc_3':
                initial = tf.truncated_normal([1, 1, 4096, num_classes], stddev=0.0001)
                kernel = tf.get_variable('kernel', initializer=initial)
                conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')
                initial = tf.constant(0.0, shape=[num_classes])
                bias = tf.get_variable('bias', initializer=initial)
                return tf.nn.bias_add(conv, bias)

            raise RuntimeError('{} is not supported as a fully connected name'.format(name))
        pass

    def _reshape_fc_weights(self, name, new_shape):
        w = self.vgg_weights[name]
        w = w.reshape(new_shape)
        init = tf.constant_initializer(value=w, dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=new_shape)
        return var

    def _conv_layer(self, parent, kernel_name, bias_name, name):
        with tf.variable_scope(name) as scope:
            kernel_weights = self._get_kernel(kernel_name)
            init = tf.constant_initializer(value=kernel_weights, dtype=tf.float32)
            kernel = tf.get_variable(name="weights", initializer=init, shape=kernel_weights.shape)
            conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')

            bias = self._get_bias(bias_name)
            init = tf.constant_initializer(value=bias, dtype=tf.float32)
            biases = tf.get_variable(name="biases", initializer=init, shape=bias.shape)

            conv_with_bias = tf.nn.bias_add(conv, biases)
            conv_with_relu = tf.nn.relu(conv_with_bias, name=scope.name)
        return conv_with_relu

    def _max_pool_layer(self, parent, kernel, stride, name, padding='SAME'):
        max_pool = tf.nn.max_pool(parent, ksize=kernel, strides=stride, padding=padding, name=name)
        return max_pool

    def _get_kernel(self, kernel_name):
        kernel = self.vgg_weights[kernel_name]
        return kernel

    def _get_bias(self, name):
        bias_weights = self.vgg_weights[name]
        return bias_weights

    @staticmethod
    def preprocess(images):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='mean')
        return images - mean

    pass


class Runner:

    def __init__(self, train_data, test_data, fcn_classifies, learning_rate, **kw):
        self._train_data = train_data
        self._test_data = test_data
        self._type_number = self._train_data.type_number
        self._image_size = self._train_data.image_size
        self._image_channel = self._train_data.image_channel
        self._batch_size = self._train_data.batch_size
        self._fcn_classifies = fcn_classifies

        input_shape = [self._batch_size, self._image_size, self._image_size, self._image_channel]
        self._images = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self._labels = tf.placeholder(dtype=tf.int32, shape=[self._batch_size, self._image_size, self._image_size])

        self._logits, self._softmax, self._prediction = self._fcn_classifies(self._images, **kw)

        self._entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=self._logits)
        self._loss = tf.reduce_mean(self._entropy)
        self._solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(self._loss)

        self._saver = tf.train.Saver()
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        pass

    # 训练网络
    def train(self, epochs, save_model, min_loss, print_loss, result_path):
        self._sess.run(tf.global_variables_initializer())
        step = 0
        now_epochs = 0
        for step in range(epochs):
            images, labels = self._train_data.next_batch()
            loss, _, softmax = self._sess.run(fetches=[self._loss, self._solver, self._softmax],
                                              feed_dict={self._images: images, self._labels: labels})
            if step % print_loss == 0:
                Tools.print_info("epoch {} {}({}): loss {}".format(self._train_data.now_epochs, step,
                                                                  self._train_data.batch_number, loss))
            if loss < min_loss:
                break

            # 测试、保存模型：每5代
            if self._train_data.now_epochs % 1 == 0 and now_epochs != self._train_data.now_epochs:
                self.test(self._train_data.now_epochs, result_path)
                self._saver.save(self._sess, save_path=save_model)
                now_epochs = self._train_data.now_epochs
                pass
            pass
        Tools.print_info("{}: train end".format(step))
        self.test(self._train_data.now_epochs, result_path)
        self._saver.save(self._sess, save_path=save_model)
        Tools.print_info("test end")
        pass

    # 测试网络
    def test(self, epoch, result_path):
        Tools.print_info("begin to test {} result in {}".format(epoch, result_path))
        now_result_path = Tools.new_dir(os.path.join(result_path, str(epoch)))
        batch_number = self._test_data.batch_number
        for now_batch in range(batch_number):
            images_data, images_name = self._test_data.next_batch_test(now_batch)
            predict = self._sess.run(fetches=self._prediction, feed_dict={self._images: images_data})
            for index, image_name in enumerate(images_name):
                im = Image.fromarray(predict[index].astype(np.uint8) * 60).convert("L")
                im.save(os.path.join(now_result_path, os.path.split(image_name)[1]))
                pass
        Tools.print_info("end to test {} result in {}".format(epoch, result_path))
        pass

    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="fcn_vgg_16_2", help="name")
    parser.add_argument("-epochs", type=int, default=1000000, help="train epoch number")
    parser.add_argument("-batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-type_number", type=int, default=5, help="type number")
    parser.add_argument("-image_size", type=int, default=256, help="image size")
    parser.add_argument("-image_channel", type=int, default=3, help="image channel")
    parser.add_argument("-keep_prob", type=float, default=0.7, help="keep prob")
    args = parser.parse_args()

    output_param = "name={},epochs={},batch_size={},type_num={},size={},channel={},keep_prob={}"
    Tools.print_info(output_param.format(args.name, args.epochs, args.batch_size, args.type_number,
                                         args.image_size, args.image_channel, args.keep_prob))

    train_data_path, train_label_path = PreData.train()
    test_data_path = PreData.test()
    now_train_data = Data(images_path=train_data_path, batch_size=args.batch_size,
                          type_number=args.type_number, image_size=args.image_size, image_channel=args.image_channel)

    now_test_data = Data(images_path=test_data_path, batch_size=args.batch_size,
                         type_number=args.type_number, image_size=args.image_size, image_channel=args.image_channel)

    now_net = FCN_VGGNet16(args.type_number, args.image_size, args.image_channel, args.batch_size)

    runner = Runner(train_data=now_train_data, test_data=now_test_data, fcn_classifies=now_net.fcn_vgg_16,
                    learning_rate=0.0001, keep_prob=args.keep_prob)
    runner.train(epochs=args.epochs, save_model=os.path.join(Tools.new_dir("model/" + args.name), args.name + ".ckpt"),
                 min_loss=1e-4, print_loss=5, result_path=Tools.new_dir("result/" + args.name))

    pass
