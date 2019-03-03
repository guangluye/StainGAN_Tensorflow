import tensorflow as tf
import layers as ly

class Discriminator:
    def __init__(self, ndf=64, name="D"):

        self.ndf = ndf
        self.name = name
        self.reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self.reuse):
            dis_conv1 = ly.general_conv2d(input, self.ndf, 4, 4, 2, 2, padding='SAME', do_norm=False, do_relu=False, name=self.name + 'conv1')
            dis_conv1 = tf.nn.leaky_relu(dis_conv1)
            dis_conv2 = ly.general_conv2d(dis_conv1, self.ndf * 2, 4, 4, 2, 2, padding='SAME', do_relu=False, name=self.name + 'conv2')
            dis_conv2 = tf.nn.leaky_relu(dis_conv2)
            dis_conv3 = ly.general_conv2d(dis_conv2, self.ndf * 4, 4, 4, 2, 2, padding='SAME', do_relu=False, name=self.name + 'conv3')
            dis_conv3 = tf.nn.leaky_relu(dis_conv3)
            dis_conv4 = ly.general_conv2d(dis_conv3, self.ndf * 8, 4, 4, 2, 2, padding='SAME', do_relu=False, name=self.name + 'conv4')
            dis_conv4 = tf.nn.leaky_relu(dis_conv4)

            output = ly.general_conv2d(dis_conv4, 4, 4, 1, 1, padding='SAME', do_norm=False, do_relu=False, name=self.name + 'output')
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output