import tensorflow as tf
import tensorflow.contrib.slim as slim
import upsampling

def fcn_32s(inputs, num_classes, is_training, dropout_keep_prob = 0.5):
    with tf.variable_scope('fcn_32s'):
        # Based on the structure of vgg-16 network
        with tf.variable_scope('vgg_16'):
            # Default settings for conv layers and fc layers
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn = tf.nn.relu,
                                weights_initializer = tf.truncated_normal_initializer(0, 0.01),
                                biases_initializer = tf.zeros_initializer,
                                weights_regularizer = slim.l2_regularizer(0.0005)):
                # Default settings for conv layer
                with slim.arg_scope([slim.conv2d], padding = 'SAME'):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    # Change from fc layer to conv layer
                    net = slim.conv2d(net, 4096, [7, 7], scope='fc6')
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                           scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                           scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='fc8')
                    # Upsampling layer to recover the size of input image
                    # Based on bilinear interpolation upsampling
                    upsample_filter = upsampling.bilinear_upsample_weights(32, num_classes)
                    upsample_filter_tensor = tf.constant(upsample_filter)
                    shape = tf.shape(net)
                    output = tf.nn.conv2d_transpose(net, upsample_filter_tensor,
                                                    output_shape = tf.stack([shape[0], shape[1] * 32,
                                                                    shape[2] * 32, shape[3]]),
                                                    strides=[1, 32, 32, 1])
                    variables = slim.get_variables('fcn_32s')
                    # Extract variables that are the same as original vgg-16
                    # They could be intialized with pre-trained vgg-16 parameters
                    vgg_variables = {}
                    for variable in variables:
                        vgg_variables[variable.name[8:-2]] = variable
                    return output, vgg_variables