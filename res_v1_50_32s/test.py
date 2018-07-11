import tensorflow as tf
import tensorflow.contrib.slim as slim
import read_data_from_tfrecord
import model, utils


""" define hyperparameters here """
num_classes = 21
num_epochs = 1
# means for r,g,b channels seperately, please refer to VGG-16 data preprocessing
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
filename = '/home/kris/PycharmProjects/eecs545/pascal_val.tfrecords'
res_parameters_dir = '/home/kris/PycharmProjects/eecs545/res_v1_50_32s/res_v1_50_32s.ckpt'
""" end of hyperparameters definition """

# Add them to queue of filenames
filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

# Read training data from tfrecords to tensors
image, label = read_data_from_tfrecord.read_data_from_tfrecords(filename_queue, None, is_training=False)
image = tf.expand_dims(image, 0)
label = tf.expand_dims(label, 0)

# Preprocess the data, move to zero mean & normalization(no need for unsigned images)
image = tf.cast(image, tf.float32)
image = image - [_R_MEAN, _G_MEAN, _B_MEAN]

# If size of image is not multiple of 32
input_size = tf.shape(image)
round_size = tf.round(tf.to_float(input_size[1:3]) / 32) * 32
resized_image = tf.image.resize_images(image, tf.to_int32(round_size))

# Predict the segmentation
logits, _ = model.res_fcn_32s(resized_image, num_classes, is_training=False)

# Rescale back to original shape
preds = tf.arg_max(logits, 3)
preds = tf.expand_dims(preds, 3)
preds = tf.image.resize_nearest_neighbor(preds, input_size[1:3])

# Remove ambiguous pixels
valid_mask = tf.not_equal(label, 255)
valid_idx = tf.where(valid_mask)
valid_label = tf.gather_nd(label, valid_idx)
valid_preds = tf.gather_nd(preds, valid_idx)

# Built-in Mean_IU metric
miou, update_op = slim.metrics.streaming_mean_iou(predictions=valid_preds, labels=valid_label, num_classes=21)


local_vars_init_op = tf.local_variables_initializer()
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)

with tf.Session() as sess:
    idx_img = 0
    sess.run(local_vars_init_op)
    saver.restore(sess, res_parameters_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while True:
            print("idx_img: %d" % (idx_img))
            [original_image, segmented_image, _] = sess.run([image, preds, update_op])
            idx_img += 1
    except tf.errors.OutOfRangeError, e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
    coord.join(threads)
    mean_iu = sess.run(miou)
    print("Mean_IU = %f" % (mean_iu))