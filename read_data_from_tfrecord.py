import tensorflow as tf
import utils
import augmentation


def read_data_from_tfrecords(filename_queue, batch_size, is_training):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    # Decode string back to uchar
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.decode_raw(features['mask_raw'], tf.uint8)

    # Reshape data back to image shape
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([height, width, 3])
    label_shape = tf.stack([height, width, 1])
    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, label_shape)

    # For training, all the data should be standardized with [384, 384]
    if is_training is True:
        image, label = augmentation.flip_randomly_left_right_image_with_annotation(image, label)
        image, label = utils.scale_image_with_fixed_size(image, label, [384, 384])
        image, label = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                            capacity=3000, num_threads=2,
                                            min_after_dequeue=2000)
    return image, label