import tensorflow as tf


def continuous_conv2d(input, in_channels, out_channels, batch_length, filter_width=3, filter_height=3, timesteps=10):
    weights = tf.get_variable('w', [filter_width, filter_height, in_channels, out_channels])
    result = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')

    weights = tf.get_variable('w_conv', [filter_width, filter_height, out_channels, out_channels])
    conv_strength = tf.get_variable('conv_strength', [1, 1, 1, out_channels], initializer=tf.constant_initializer(0.1))

    current_shape = result.get_shape()
    conv_strength = tf.tile(conv_strength, [batch_length, current_shape[1], current_shape[2], 1])

    for i in range(timesteps):
        conv_result = tf.nn.conv2d(result, weights, strides=[1, 1, 1, 1], padding='SAME')
        result = result + conv_strength * conv_result

    return result, weights


def continuous_pool(input, in_channels, batch_length, timesteps=10, type='avg', ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='VALID'):
    pool_strength = tf.get_variable('pool_strength', [1, 1, 1, in_channels],
                                    initializer=tf.constant_initializer(0.1))
    current_shape = input.get_shape()

    pool_strength = tf.tile(pool_strength, [batch_length, current_shape[1], current_shape[2], 1])

    for i in range(timesteps):
        pooled = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        diff = pooled - input
        input = input + pool_strength * diff

    if type == 'avg':
        result = tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding=padding)
    elif type == 'max':
        result = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)
    else:
        raise NameError('The available pooling types are \'avg\' and \'max\'!')

    return result


def continuous_conv_layer(input, in_channels, out_channels, filter_width=3, filter_height=3, layer_num=''):
    with tf.variable_scope('cont_conv' + str(layer_num)):
        result = continuous_conv2d(input, in_channels, out_channels, filter_width, filter_height)
        result = continuous_pool(result, out_channels)

        return result
