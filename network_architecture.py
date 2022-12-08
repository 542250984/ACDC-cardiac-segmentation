import tensorflow.compat.v1 as tf
tf.disable_v2_behavior


def RCN(image, segt_class, training, n_filter=[96,96,96,96,96]):
    net = {}
    tmp = tf.expand_dims(image, 1);
    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[0], training=training)
    net['conv1'] = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[0], training=training)

    tmp = conv2d_bn_lrelu_cf_d(net['conv1'], filters=n_filter[1], training=training,dilation_rate=2)
    net['conv2'] = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[1], training=training,dilation_rate=2)


    tmp = conv2d_bn_lrelu_cf_d(net['conv2'], filters=n_filter[2], training=training,dilation_rate=4)
    net['conv3'] = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[2], training=training,dilation_rate=4)


    tmp = conv2d_bn_lrelu_cf_d(net['conv3'], filters=n_filter[3], training=training, dilation_rate=8)
    net['conv4'] = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[3], training=training, dilation_rate=8)


    tmp = conv2d_bn_lrelu_cf_d(net['conv4'], filters=n_filter[4], training=training, dilation_rate=16)
    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[4], training=training, dilation_rate=16)

    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[4], training=training, dilation_rate=8)
    tmp = tf.concat([net['conv4'], tmp], axis=1)
    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[3], training=training,dilation_rate=8)
    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[3], training=training,dilation_rate=8)

    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[3], training=training, dilation_rate=4)
    tmp = tf.concat([net['conv3'], tmp], axis=1)
    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[2], training=training, dilation_rate=4)
    net['conv3_sta'] = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[2], training=training, dilation_rate=4)

    tmp = conv2d_bn_lrelu_cf_d(net['conv3_sta'], filters=n_filter[2], training=training, dilation_rate=2)
    tmp = tf.concat([net['conv2'], tmp], axis=1)
    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[1], training=training, dilation_rate=2)
    net['conv2_sta'] = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[1], training=training, dilation_rate=2)

    tmp = conv2d_bn_lrelu_cf_d(net['conv2_sta'], filters=n_filter[1], training=training)
    tmp = tf.concat([net['conv1'], tmp], axis=1)
    tmp = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[0], training=training)
    net['conv1_sta'] = conv2d_bn_lrelu_cf_d(tmp, filters=n_filter[0], training=training)

    tmp = tf.layers.conv2d(net['conv3_sta'], filters=segt_class, kernel_size=1, strides=1, padding='same',data_format='channels_first')
    tmp = tf.layers.conv2d(net['conv2_sta'], filters=segt_class, kernel_size=1, strides=1, padding='same',data_format='channels_first') + tmp
    tmp = tf.layers.conv2d(net['conv1_sta'], filters=segt_class, kernel_size=1, strides=1, padding='same',data_format='channels_first') + tmp
    logits_segt = tf.reshape(tmp, [tf.shape(tmp)[0], segt_class, tf.shape(tmp)[-2], tf.shape(tmp)[-1]])

    return logits_segt

def conv2d_bn_lrelu_cf_d(x, filters, training,dilation_rate=1, kernel_size=3, strides=1):
    """ Basic Conv + BN + ReLU unit """

    x_conv = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides,padding='same', data_format='channels_first',dilation_rate=dilation_rate,use_bias=False)
    x_bn = tf.layers.batch_normalization(x_conv,axis=1, training=training)
    x_relu = tf.nn.leaky_relu(x_bn)

    return x_relu

def conv2d_bn_lrelu(x, filters, training, kernel_size=3, strides=1):
    """ Basic Conv + BN + ReLU unit """

    x_conv = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides,padding='same', data_format='channels_last',use_bias=False)
    x_bn = tf.layers.batch_normalization(x_conv,axis=-1, training=training)
    x_relu = tf.nn.leaky_relu(x_bn)

    return x_relu

def CSN(image, segt_class, n_slice,training,n_filter=[16,32,64,128,256]):
    net = {}
    # NHWC (Batch_size,Height, Width,channel)
    # 4,288,288,21
    tmp = conv2d_bn_lrelu(image, filters=n_filter[0], training=training)
    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[0], training=training)
    net['conv1'] = conv2d_bn_lrelu(tmp, filters=32, training=training)

    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[1], training=training, strides = 2)
    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[1], training=training)
    net['conv2'] = conv2d_bn_lrelu(tmp, filters=32, training=training)

    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[2], training=training, strides = 2)
    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[2], training=training)
    net['conv3'] = conv2d_bn_lrelu(tmp, filters=32, training=training)

    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[3], training=training, strides = 2)
    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[3], training=training)
    net['conv4'] = conv2d_bn_lrelu(tmp, filters=32, training=training)

    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[4], training=training, strides = 2)
    tmp = conv2d_bn_lrelu(tmp, filters=n_filter[4], training=training)
    tmp = conv2d_bn_lrelu(tmp, filters=32, training=training)

    tmp = tf.layers.conv2d_transpose(tmp, filters=32, kernel_size=31, strides=16, padding='same')
    net['conv4'] = tf.layers.conv2d_transpose(net['conv4'], filters=32, kernel_size=15, strides=8, padding='same')
    net['conv3'] = tf.layers.conv2d_transpose(net['conv3'], filters=32, kernel_size=7, strides=4, padding='same')
    net['conv2'] = tf.layers.conv2d_transpose(net['conv2'], filters=32, kernel_size=3, strides=2, padding='same')
    tmp = tf.concat([net['conv1'], net['conv2'], net['conv3'], net['conv4'], tmp], axis=-1)

    tmp = conv2d_bn_lrelu(tmp, filters=64, training=training)
    tmp = conv2d_bn_lrelu(tmp, filters=64, training=training)
    logits_segt  = tf.layers.conv2d(tmp, filters=segt_class*n_slice,kernel_size = 1, padding = 'same')
    logits_segt = tf.reshape(logits_segt , [tf.shape(logits_segt )[0],tf.shape(logits_segt )[1], tf.shape(logits_segt )[2],n_slice,segt_class])

    return logits_segt

