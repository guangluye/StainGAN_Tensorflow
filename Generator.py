import tensorflow as tf
import layers as ly

class Generator:
    def __init__(self, ngf=64, name="G"):

        self.ngf = ngf
        self.name = name
        self.reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self.reuse):

            padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
            en_conv1 = ly.general_conv2d(padded, self.ngf, 7, 7, 1, 1, name="conv1")
            en_conv2 = ly.general_conv2d(en_conv1, 2 * self.ngf, 3, 3, 2, 2, padding="SAME", name="conv2")
            en_conv3 = ly.general_conv2d(en_conv2, 4 * self.ngf, 3, 3, 2, 2, padding="SAME", name="conv3")

            res = en_conv3
            for i in range(1, 10):
                res = ly.resnet_block(res, 4 * self.ngf, name="res{}".format(i))

            de_conv1 = ly.general_deconv2d(res, 2 * self.ngf, 3, 3, 2, 2, padding="SAME", name="deconv1")
            de_conv2 = ly.general_deconv2d(de_conv1, self.ngf, 3, 3, 2, 2, padding="SAME", name="deconv2")

            padded_de_conv2 = tf.pad(de_conv2, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
            output = ly.general_conv2d(padded_de_conv2, 3, 7, 7, 1, 1, do_norm=False, do_relu=False, name="conv")
            output = tf.nn.tanh(output, name="output")
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return output

