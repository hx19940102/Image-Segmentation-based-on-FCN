import tensorflow as tf

def remove_ambiguous(labels_batch, logits_batch):
    # For pascal voc data, the label of 255 represents 'ambiguous'
    # which is useless for training or testing
    valid_mask = tf.not_equal(labels_batch, 255)
    valid_idx = tf.where(valid_mask)
    labels_batch = tf.one_hot(labels_batch, 21, 1, 0)
    # Remove the 'ambiguous' pixels
    valid_labels_batch = tf.gather_nd(labels_batch, valid_idx)
    valid_logits_batch = tf.gather_nd(logits_batch, valid_idx)
    return valid_labels_batch, valid_logits_batch

def extract_vgg_variables(vgg_variables):
    vgg_keys = vgg_variables.keys()
    vgg_without_fc8_keys = []
    for key in vgg_keys:
        if 'fc8' not in key:
            vgg_without_fc8_keys.append(key)
    updated_mapping = {key: vgg_variables[key] for key in vgg_without_fc8_keys}
    return updated_mapping

def extract_res_variables(res_variables):
    res_keys = res_variables.keys()
    res_without_logits_keys = []
    for key in res_keys:
        if 'logits' not in key:
            res_without_logits_keys.append(key)
    updated_mapping = {key: res_variables[key] for key in res_without_logits_keys}
    return updated_mapping


def scale_image_with_fixed_size(img_tensor, annotation_tensor, output_shape,
                                min_relative_random_scale_change=0.9,
                                max_realtive_random_scale_change=1.1,
                                mask_out_number=255):
    # tf.image.resize_nearest_neighbor needs
    # first dimension to represent the batch number
    img_batched = tf.expand_dims(img_tensor, 0)
    annotation_batched = tf.expand_dims(annotation_tensor, 0)

    annotation_batched = tf.to_int32(annotation_batched)

    # Get height and width tensors
    input_shape = tf.shape(img_batched)[1:3]
    input_shape_float = tf.to_float(input_shape)
    scales = output_shape / input_shape_float

    rand_var = tf.random_uniform(shape=[1],
                                 minval=min_relative_random_scale_change,
                                 maxval=max_realtive_random_scale_change)

    final_scale = tf.reduce_min(scales) * rand_var

    scaled_input_shape = tf.to_int32(tf.round(input_shape_float * final_scale))

    resized_img = tf.image.resize_nearest_neighbor(img_batched, scaled_input_shape)
    resized_annotation = tf.image.resize_nearest_neighbor(annotation_batched, scaled_input_shape)

    resized_img = tf.squeeze(resized_img, axis=0)
    resized_annotation = tf.squeeze(resized_annotation, axis=0)

    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    annotation_shifted_classes = resized_annotation + 1

    cropped_padded_img = tf.image.resize_image_with_crop_or_pad(resized_img, output_shape[0], output_shape[1])
    cropped_padded_annotation = tf.image.resize_image_with_crop_or_pad(annotation_shifted_classes,
                                                                       output_shape[0],
                                                                       output_shape[1])

    annotation_additional_mask_out = tf.to_int32(tf.equal(cropped_padded_annotation, 0)) * (mask_out_number + 1)
    cropped_padded_annotation = cropped_padded_annotation + annotation_additional_mask_out - 1

    return cropped_padded_img, cropped_padded_annotation
