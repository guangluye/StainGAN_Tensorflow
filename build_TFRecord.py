import tensorflow as tf
import os
from PIL import Image

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('X_input_dir', 'data/trainX',
                       'X input directory')
tf.flags.DEFINE_string('Y_input_dir', 'data/trainY',
                       'Y input directory')
tf.flags.DEFINE_string('X_output_dir', 'data/trainX_tfrecords',
                       'X output tfrecords files directory')
tf.flags.DEFINE_string('Y_output_dir', 'data/trainY_tfrecords',
                       'Y output tfrecords files directory')

#Tensorflow官方的建议是一个TFRecord中最好图片的数量为1000张左右
nums_images_per_tffile = 1000

def data_paths(input_dir):
    """
    返回input_dir目录下的图片路径集合
    :param input_dir:
    :return:
    """
    file_path = []
    for file in os.scandir(input_dir):
        if file.name.endswith('.jpg') and file.is_file():
            file_path.append(file.path)

    return file_path

def _int64_feature(value):
    """
    生成整数型的属性
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """
    生成字符串型的属性
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(input_dir, output_dir):
    """

    :param input_dir:
    :param output_dir:
    :return:
    """
    file_paths = data_paths(input_dir)
    images_nums = len(file_paths)
    cur_tr_num = 1  #当前tfrecord文件的编号
    cur_img = 1 #当前iamge的编号

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tfrecords_file_name = (output_dir + "/%.3d.tfrecords" % cur_tr_num)
    print(tfrecords_file_name)
    writer = tf.python_io.TFRecordWriter(tfrecords_file_name)

    for i in range(images_nums):
        if(cur_img > nums_images_per_tffile):
            cur_img = 1
            cur_tr_num += 1
            tfrecords_file_name = (output_dir + "/%.3d.tfrecords" % cur_tr_num)
            print(tfrecords_file_name)
            writer = tf.python_io.TFRecordWriter(tfrecords_file_name)

        img = Image.open(file_paths[i])
        size = img.size
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'img_width' : _int64_feature(size[0]),
            # 'img_height' : _int64_feature(size[1]),
            'img_raw' : _bytes_feature(img_raw)
        }))

        writer.write(example.SerializeToString())
        cur_img += 1
    writer.close()

if __name__ == '__main__':
    print('Convert X data to tfrecords...')
    write_tfrecords(FLAGS.X_input_dir, FLAGS.X_output_dir)
    print('Convert Y data to tfrecords...')
    write_tfrecords(FLAGS.Y_input_dir, FLAGS.Y_output_dir)

