import tensorflow as tf
import tensorflow.contrib.slim as slim
import read_data_from_tfrecord
import model, utils


""" Define hyperparameters here """
batch_size = 1
num_classes = 21
num_epochs = 30
learning_rate = 0.000001
training_rounds = 200000
# means for r,g,b channels seperately, please refer to VGG-16 data preprocessing
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
filename = '/home/kris/PycharmProjects/eecs545/pascal_train.tfrecords'
res_32s_parameters_dir = '/home/kris/PycharmProjects/eecs545/res_v1_50_32s/res_v1_50_32s.ckpt'
res_parameters_dir = '/home/kris/PycharmProjects/eecs545/resnet_v1_50.ckpt'
""" End of hyperparameters definition """

# Add them to queue of filenames
filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
# Read training data from tfrecords to tensors
images, labels = read_data_from_tfrecord.read_data_from_tfrecords(filename_queue, batch_size, is_training=True)
labels = tf.squeeze(labels, axis = -1)
# Preprocess the data, move to zero mean & normalization(no need for unsigned images)
images = tf.cast(images, tf.float32)
images = images - [_R_MEAN, _G_MEAN, _B_MEAN]

logits, res_variables = model.res_fcn_32s(images, num_classes, is_training=False)
valid_labels, valid_logits = utils.remove_ambiguous(labels, logits)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# Most of parameters could be intialized with pre-trained parameters of original residual network
res_variables_without_logits = utils.extract_res_variables(res_variables)
init = slim.assign_from_checkpoint_fn(res_parameters_dir, res_variables_without_logits)
global_vars_init_op = tf.global_variables_initializer()
local_vars_init_op = tf.local_variables_initializer()
combined_op = tf.group(local_vars_init_op, global_vars_init_op)
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)

with tf.Session() as sess:
    sess.run(combined_op)
    init(sess)
    # If restart training after break
    saver.restore(sess, res_32s_parameters_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(training_rounds):
        error, _ = sess.run([loss, optimizer])
        print("Round %d, Loss = %f" % (i, error))
        if i % 299 == 0:
            saver.save(sess, res_32s_parameters_dir)
    coord.request_stop()
    coord.join(threads)

saver.save(sess, res_32s_parameters_dir)