import tensorflow as tf

tf.__version__

_BATCH_NORM_DECAY=0.9
_BATCH_NORM_EPSILON=1e-05
_LEAKY_RELU=0.1
_ANCHORS=[(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]


def batch_normalization(inputs, training, data_format):
    '''
    Perform the batch normalization of the tensor input

    Argument:
    input -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')
    training -- bool: whether trainning () or not the parameters of this layer (mean and varance) 

    data_format -- str: the format of the tensor input ('channels_last' or 'channels_first')


    return: -- A tensor after normalizing the input
    '''

    return tf.layers.BatchNormalization(inputs=inputs, axis=1 if data_format=='channels_first' else 3,
                                        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                        scale=True, training=training)
    

def fixed_paddind(inputs, filter_size,data_format, mode='CONSTANT'):
    '''
    Padding the inputs independently from the size of filter

    Argument:
    input -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')
    filter_size -- integer: the size of filters (kernels)

    data_format -- str: the format of the tensor input ('channels_last' or 'NCHW')

    mode -- str: the mode for tf.pad

    return: -- A tensor with the same format as input after padding the height and width
    '''

    pad_total=filter_size-1
    pad_begin=pad // 2
    padd_end=pad_total-pad_begin

    if data_format =='channels_last':
        return tf.pad(inputs, [[0,0],[pad_begin, padd_end],
                              [pad_begin, padd_end],
                              [0,0]], mode)
    else:
        return tf.pad(inputs, [[0,0],[0,0],
                              [pad_begin, padd_end],
                              [pad_begin, padd_end]], mode)

    
def conv2d_fixed_padding(inputs,filters, filter_size, data_format, mode='CONSTANT', strides=1):
     fixed_paddind(inputs, filter_size,data_format, mode='CONSTANT'):
    '''
    Stride 2D conv layers with the fixed padding


    '''
    if strides>1:
        inputs=fixed_paddind(inputs, filter_size,data_format)

    return tf.nn.conv2d(inputs,filters=filters, kernel_size=filter_size, strides=strides,
            padding='SAME' if strides==1 else 'VALID', data_format=data_format)

def darknet53_residual_block(inputs,filters, trainning, data_format):
    '''
    Create the block of the Darknet53 model

    Argument:
    input -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')
    filters -- integer: the number of filters (kernels)
    training -- bool: for batch_normalization layers

    data_format -- str: the format of the tensor input ('channels_last' or 'channels_first')


    return: -- A residual block of Darknet53 netework 
    '''

    short_cut=inputs

    inputs=conv2d_fixed_padding(inputs,filters=filters,filter_size=1 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs=conv2d_fixed_padding(inputs,filters=filters * 2,filter_size=3, data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs +=short_cut
    return inputs

def darknet53_model(inputs, trainning, data_format):
    '''
    Create the Darknet53 model

    Argument:
    input -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')
    training -- bool: for batch_normalization layers

    data_format -- string: the format of the tensor input ('channels_last' or 'channels_first')


    return: -- the Darknet 53 model
    '''
    inputs=conv2d_fixed_padding(inputs,filters=32,filter_size=1 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs=conv2d_fixed_padding(inputs,filters=64,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    inputs=conv2d_fixed_padding(inputs,filters=32,filter_size=1 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs=conv2d_fixed_padding(inputs,filters=64,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs=darknet53_residual_block(inputs, 32, trainning, data_format)


    inputs=conv2d_fixed_padding(inputs,filters=128,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    for _ in range(2):
        inputs=darknet53_residual_block(inputs, 64, trainning, data_format)


    inputs=conv2d_fixed_padding(inputs,filters=256,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    for _ in range(8):
        inputs=darknet53_residual_block(inputs, 128, trainning, data_format)

    routine1=inputs

    inputs=conv2d_fixed_padding(inputs,filters=512,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _in range(8):
        inputs=darknet53_residual_block(inputs, 256, trainning, data_format)

    routine2=inputs

    inputs=conv2d_fixed_padding(inputs,filters=1024,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs=darknet53_residual_block(inputs, 512, trainning, data_format)


    return routine1, routine2, inputs

    



