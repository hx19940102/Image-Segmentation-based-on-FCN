import tensorflow as tf
import tensorflow.contrib.slim as slim
import upsampling

def fcn_8s(inputs, num_classes, is_training, dropout_keep_prob = 0.5):
    with tf.variable_scope('fcn_8s'):
        # Based on the structure of vgg-16 network
        with tf.variable_scope('vgg_16'):
            end_points_collection = 'vgg_16' + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
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
                        output = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                          normalizer_fn=None, scope='fc8')
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        # Upsampling layer to recover the size of input image
        # Based on bilinear interpolation upsampling
        output_pool4 = end_points['fcn_8s/vgg_16/pool4']
        output_pool4 = slim.conv2d(output_pool4, num_classes, [1, 1], activation_fn=None,
                                    normalizer_fn=None, scope='pool4_fc')
        upsample_filter_2 = upsampling.bilinear_upsample_weights(2, num_classes)
        upsample_filter_tensor_2 = tf.constant(upsample_filter_2)
        shape_output = tf.shape(output)
        output = tf.nn.conv2d_transpose(output, upsample_filter_tensor_2,
                                        output_shape = tf.stack([shape_output[0], shape_output[1] * 2,
                                                                shape_output[2] * 2, shape_output[3]]),
                                        strides=[1, 2, 2, 1])
        output_combined = output + output_pool4
        shape_output_combined = tf.shape(output_combined)
        output_combined = tf.nn.conv2d_transpose(output_combined, upsample_filter_tensor_2,
                                        output_shape=tf.stack([shape_output_combined[0], shape_output_combined[1] * 2,
                                                                shape_output_combined[2] * 2, shape_output_combined[3]]),
                                        strides=[1, 2, 2, 1])
        output_pool3 = end_points['fcn_8s/vgg_16/pool3']
        output_pool3 = slim.conv2d(output_pool3, num_classes, [1, 1], activation_fn=None,
                                    normalizer_fn=None, scope='pool3_fc')
        output_combined = output_combined + output_pool3
        upsample_filter_8 = upsampling.bilinear_upsample_weights(8, num_classes)
        upsample_filter_tensor_8 = tf.constant(upsample_filter_8)
        shape_output_pool3 = tf.shape(output_pool3)
        output_combined = tf.nn.conv2d_transpose(output_combined, upsample_filter_tensor_8,
                                        output_shape=tf.stack([shape_output_pool3[0], shape_output_pool3[1] * 8,
                                                                shape_output_pool3[2] * 8, shape_output_pool3[3]]),
                                        strides=[1, 8, 8, 1])

        variables = slim.get_variables('fcn_8s')
        # Extract variables that are the same as fcn_32s model
        # They could be intialized with pre-trained fcn_32s parameters
        fcn_16s_variables = {}
        for variable in variables:
            if 'pool3_fc' not in variable.name:
                fcn_16s_variables[variable.name[0:4] + '16' + variable.name[5:-2]] = variable
        return output_combined, fcn_16s_variables