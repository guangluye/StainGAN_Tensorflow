import tensorflow as tf
import matplotlib.pyplot as plt


def parser(record):
    features = tf.parse_single_example(record, features={
        'img_width': tf.FixedLenFeature([], tf.int64),
        'img_height': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    decoded_img = tf.decode_raw(features['img_raw'], tf.uint8)
    # decoded_img.set_shape([features['img_width'], features['img_height'], 3])
    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    decoded_img = tf.reshape(decoded_img, [height, width, 3])
    decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)
    return decoded_img


def preprocess(image, width, height):
    image = tf.image.resize_images(image, [height, width])
    print(image.shape)
    return image


def build_dataset(tfrecords_path, width=400, height=300, batch_size=2, num_epochs=100, shuffle_buffer=1000):
    train_files = tf.train.match_filenames_once(tfrecords_path + '*.tfrecords')

    dataset = tf.data.TFRecordDataset(train_files)
    dataset = dataset.map(parser)
    dataset = dataset.map(lambda image : (preprocess(image, width, height)))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    return dataset

if __name__ == '__main__':
    tfrecords_path = './data/test_tfrecords'
    data = build_dataset(tfrecords_path, batch_size=1, num_epochs=1)
    iterator = data.make_initializable_iterator()
    image_batch = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        image = sess.run(image_batch)
        print(image.shape)
        plt.imshow(image[0])
        plt.show()