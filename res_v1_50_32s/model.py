import tensorflow as tf
import tensorflow.contrib.slim as slim
import upsampling
import resnet_v1

def res_fcn_32s(inputs, num_classes, is_training):
    with tf.variable_scope('res_fcn_32s'):
        # Use the structure of res_v1_50 classification network
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(inputs, num_classes, is_training=is_training,
                                                 global_pool=False, output_stride=32)
        # Deconvolutional layers to recover the size of input image
        # Padding is 'SAME' for conv layers thus conv layers do not change the size
        # There are 5 max-pool layers with size reduced by half
        # Totally size reduced by scale of 2^5 = 32 times
        # That's also the reason why this model is called fcn_32s
        # Use bilinear interpolation for upsampling
        upsample_filter = upsampling.bilinear_upsample_weights(32, num_classes)
        upsample_filter_tensor = tf.constant(upsample_filter)
        shape = tf.shape(net)
        output = tf.nn.conv2d_transpose(net, upsample_filter_tensor,
                                        output_shape = tf.stack([shape[0], shape[1] * 32,
                                                        shape[2] * 32, shape[3]]),
                                        strides=[1, 32, 32, 1])
        variables = slim.get_variables('res_fcn_32s')
        # Extract variables that are the same as original vgg-16, they could be intialized
        # with pre-trained vgg-16 network
        res_variables = {}
        for variable in variables:
            res_variables[variable.name[12:-2]] = variable
        return output, res_variables

# Todo: Train the res_v1_50 based 16 stride fcn
def res_fcn_16s(inputs, num_classes, is_training):
    with tf.variable_scope('res_fcn_16s'):
        # Use the structure of res_v1_50 classification network
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(inputs, num_classes, is_training=is_training,
                                                 global_pool=False, output_stride=16)
        # Deconvolutional layers to recover the size of input image
        # Padding is 'SAME' for conv layers thus conv layers do not change the size
        # There are 5 max-pool layers with size reduced by half
        # Totally size reduced by scale of 2^5 = 32 times
        # That's also the reason why this model is called fcn_32s
        # Use bilinear interpolation for upsampling
        upsample_filter = upsampling.bilinear_upsample_weights(16, num_classes)
        upsample_filter_tensor = tf.constant(upsample_filter)
        shape = tf.shape(net)
        output = tf.nn.conv2d_transpose(net, upsample_filter_tensor,
                                        output_shape = tf.stack([shape[0], shape[1] * 16,
                                                        shape[2] * 16, shape[3]]),
                                        strides=[1, 16, 16, 1])
        variables = slim.get_variables('res_fcn_16s')
        # Extract variables that are the same as original vgg-16, they could be intialized
        # with pre-trained vgg-16 network
        res_variables = {}
        for variable in variables:
            res_variables[variable.name[12:-2]] = variable
        return output, res_variables
