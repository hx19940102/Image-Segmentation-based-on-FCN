import tensorflow as tf
import tensorflow.contrib.slim as slim
import model, utils
import read_data_from_tfrecord


""" Define hyperparameters here """
batch_size = 1
num_classes = 21
num_epochs = None
learning_rate = 0.000001
training_rounds = 50000
# Means for RGB channels seperately from VGG-16 data preprocessing
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
filename = '/home/kris/PycharmProjects/eecs545/pascal_train.tfrecords'
fcn32s_parameters_dir = '/home/kris/PycharmProjects/eecs545/fcn_32s/fcn32s.ckpt'
vgg_16_parameters_dir = '/home/kris/PycharmProjects/eecs545/vgg_16.ckpt'
""" End of hyperparameters definition """

# Add them to queue of filenames
filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
# Read training data from tfrecords to tensors
images, labels = read_data_from_tfrecord.read_data_from_tfrecords(filename_queue, batch_size, is_training=True)
labels = tf.squeeze(labels, axis = -1)
# Preprocess the data, move to zero mean & normalization(no need for unsigned images)
images = tf.cast(images, tf.float32)
images = images - [_R_MEAN, _G_MEAN, _B_MEAN]

logits, vgg_variables = model.fcn_32s(images, num_classes, is_training=True)
valid_labels, valid_logits = utils.remove_ambiguous(labels, logits)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# Most of parameters could be intialized with pre-trained parameters of original vgg-16 network
vgg_variables_without_fc8 = utils.extract_vgg_variables(vgg_variables)
init = slim.assign_from_checkpoint_fn(vgg_16_parameters_dir, vgg_variables_without_fc8)
global_vars_init_op = tf.global_variables_initializer()
local_vars_init_op = tf.local_variables_initializer()
combined_op = tf.group(local_vars_init_op, global_vars_init_op)
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)

with tf.Session() as sess:
    sess.run(combined_op)
    init(sess)
    # If restart training after break
    saver.restore(sess, fcn32s_parameters_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(training_rounds):
        error, _ = sess.run([loss, optimizer])
        print("Round %d, Loss = %f" % (i, error))
        if i % 199 == 0:
            saver.save(sess, fcn32s_parameters_dir)
    coord.request_stop()
    coord.join(threads)

saver.save(sess, fcn32s_parameters_dir)