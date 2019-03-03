import tensorflow as tf

def general_conv2d(input, nums_output_filters=64, f_h=7, f_w=7, s_w=2, s_h=2,
                  stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True):
    """

    :param input:
    :param nums_output_filters:
    :param f_h:
    :param f_w:
    :param s_w:
    :param s_h:
    :param stddev:
    :param padding:
    :param name:
    :param do_norm:
    :param do_relu:
    :return:
    """
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(input, nums_output_filters, [f_h, f_w], [s_w, s_h], padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.zeros_initializer())
        if do_norm:
            #作者使用的是instance_norm，关于下面函数的各项参数设置还有待考究
            conv = tf.contrib.layers.instance_norm(conv, epsilon=1e-05)

        if do_relu:
            conv = tf.nn.relu(conv, "relu")

        return conv

def general_deconv2d(input, nums_output_filters=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, do_relu=True):

    """

    :param input:
    :param nums_output_filters:
    :param f_h:
    :param f_w:
    :param s_h:
    :param s_w:
    :param stddev:
    :param padding:
    :param name:
    :param do_norm:
    :param do_relu:
    :return:
    """
    with tf.variable_scope(name):
        deconv = tf.contrib.layers.conv2d_transpose(input, nums_output_filters, [f_h, f_w], [s_h, s_w], padding, activation_fn=None,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                    biases_initializer=tf.zeros_initializer())
        if do_norm:
            # 作者使用的是instance_norm，关于下面函数的各项参数设置还有待考究
            deconv = tf.contrib.layers.instance_norm(deconv, epsilon=1e-05)

        if do_relu:
            deconv = tf.nn.relu(deconv, "relu")

        return deconv

def resnet_block(input, dim, name="res_block"):
    """

    :param input:
    :param dim:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        out_res = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, padding="VALID", name="conv1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, padding="VALID", name="conv2", do_relu=False)

        return input + out_res
