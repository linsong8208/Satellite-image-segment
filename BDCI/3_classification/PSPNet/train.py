from __future__ import print_function

import os
import time

import tensorflow as tf
import numpy as np

from pspnet_model import PSPNet
from tools import prepare_label
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


class Train(object):

    def __init__(self, num_classes=5, batch_size=1, ignore_label=255,
                 log_dir="./model_bdci_5", model_name="model.ckpt", save_dir="./output_bdci_5",
                 data_dir="data/bdci/train", train_list="data/bdci/train_list.txt"):

        self.log_dir = Tools.new_dir(log_dir)
        self.model_name = model_name
        self.save_dir = Tools.new_dir(save_dir)
        self.checkpoint_path = os.path.join(self.log_dir, self.model_name)

        self.img_mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        self.input_size = [713, 713]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        self.data_dir = data_dir
        self.data_train_list = train_list

        self.random_seed = 1234
        self.random_scale = True
        self.random_mirror = True
        self.train_beta_gamma = True
        self.weight_decay = 0.0001
        self.learning_rate = 1e-3
        self.num_steps = 60001
        self.power = 0.9
        self.update_mean_var = True
        self.momentum = 0.9

        self.save_pred_freq = 10

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        pass

    def run(self):
        tf.set_random_seed(self.random_seed)
        coord = tf.train.Coordinator()

        # 读取数据
        with tf.name_scope("create_inputs"):
            reader = ImageReader(self.data_dir, self.data_train_list, self.input_size, self.random_scale,
                                 self.random_mirror, self.ignore_label, self.img_mean, coord)
            image_batch, label_batch = reader.dequeue(self.batch_size)

        # 网络
        net = PSPNet({'data': image_batch}, is_training=True, num_classes=self.num_classes)
        raw_output = net.layers['conv6']

        # According from the prototxt in Caffe implement, learning rate must multiply by 10.0 in pyramid module
        fc_list = ['conv5_3_pool1_conv', 'conv5_3_pool2_conv', 'conv5_3_pool3_conv',
                   'conv5_3_pool6_conv', 'conv6', 'conv5_4']
        # 所有的变量
        restore_var = [v for v in tf.global_variables()]
        # 所有可训练变量
        all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or
                         self.train_beta_gamma]
        # fc_list中的全连接层可训练变量和卷积可训练变量
        fc_trainable = [v for v in all_trainable if v.name.split('/')[0] in fc_list]
        conv_trainable = [v for v in all_trainable if v.name.split('/')[0] not in fc_list]  # lr * 1.0
        fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
        fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
        # 验证
        assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(raw_output, [-1, self.num_classes])
        label_process = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]),
                                      num_classes=self.num_classes, one_hot=False)  # [batch_size, h, w]
        raw_gt = tf.reshape(label_process, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        l2_losses = [self.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # Using Poly learning rate policy
        base_lr = tf.constant(self.learning_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / self.num_steps), self.power))

        # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
        update_ops = None if not self.update_mean_var else tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # 对变量以不同的学习率优化：分别求梯度、应用梯度
        with tf.control_dependencies(update_ops):
            opt_conv = tf.train.MomentumOptimizer(learning_rate, self.momentum)
            opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.momentum)
            opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.momentum)

            grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
            grads_conv = grads[:len(conv_trainable)]
            grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
            grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

            train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
            train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
            train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

            train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
            pass

        sess = tf.Session(config=self.config)
        sess.run(tf.global_variables_initializer())

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

        # 加载模型
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver(var_list=restore_var).restore(sess, ckpt.model_checkpoint_path)
            Tools.print_info("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            Tools.print_info('No checkpoint file found.')
            pass

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Iterate over training steps.
        for step in range(self.num_steps):
            start_time = time.time()
            if step % self.save_pred_freq == 0:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict={step_ph: step})
                saver.save(sess, self.checkpoint_path, global_step=step)
                Tools.print_info('The checkpoint has been created.')
            else:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict={step_ph: step})
            duration = time.time() - start_time
            Tools.print_info('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

        coord.request_stop()
        coord.join(threads)
        pass

    pass


if __name__ == '__main__':
    Train().run()
