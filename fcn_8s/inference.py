import tensorflow as tf
import numpy as np
import cv2
import model

""" define hyperparameters here """
num_classes = 21
# Means for RGB channels seperately from VGG-16 data preprocessing
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
filename = '/home/kris/PycharmProjects/eecs545/img1.jpeg'
fcn8s_parameters_dir = '/home/kris/PycharmProjects/eecs545/fcn_8s/fcn8s.ckpt'
""" end of hyperparameters definition """

image = tf.placeholder(tf.uint8, [None, None, None, 3])

# Preprocess the data, move to zero mean & normalization(no need for unsigned images)
image = tf.cast(image, tf.float32)
image = image - [_R_MEAN, _G_MEAN, _B_MEAN]

# If size of image is not multiple of 32
input_size = tf.shape(image)
round_size = tf.round(tf.to_float(input_size[1:3]) / 32) * 32
resized_image = tf.image.resize_images(image, tf.to_int32(round_size))

# Predict the segmentation
logits, _ = model.fcn_8s(resized_image, num_classes, is_training=False)
logits = tf.arg_max(logits, 3)
logits = tf.expand_dims(logits, axis=3)
logits = tf.image.resize_nearest_neighbor(logits, input_size[1:3])
logits = tf.squeeze(logits, axis=-1)

local_vars_init_op = tf.local_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(local_vars_init_op)
    saver.restore(sess, fcn8s_parameters_dir)

    img = cv2.imread(filename)
    res = sess.run(logits, feed_dict={image:[img]})
    res = np.squeeze(res)
    max_val = np.max(np.max(res, 1))
    res = (255 / max_val) * res
    res = np.uint8(res)
    cv2.imshow('img', img)
    cv2.imshow('res', res)
    cv2.imwrite('/home/kris/PycharmProjects/eecs545/fcn_8s/res.jpg', res)
    cv2.waitKey(0)

    sess.close()