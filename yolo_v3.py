import tensorflow as tf



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
    

def fixed_padding(inputs, filter_size,data_format, mode='CONSTANT'):
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
    pad_begin=pad_total // 2
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
    '''
    Stride 2D conv layers with the fixed padding


    '''
    if strides>1:
        inputs=fixed_padding(inputs, filter_size,data_format)

    return tf.layers.conv2d(inputs,filters=filters, kernel_size=filter_size, strides=strides,
            padding='SAME' if strides==1 else 'VALID', data_format=data_format)

def darknet53_residual_block(inputs,filters, training, data_format):
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

def darknet53_model(inputs, training, data_format):
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

    inputs=darknet53_residual_block(inputs, 32, training, data_format)


    inputs=conv2d_fixed_padding(inputs,filters=128,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    for _ in range(2):
        inputs=darknet53_residual_block(inputs, 64, training, data_format)


    inputs=conv2d_fixed_padding(inputs,filters=256,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    for _ in range(8):
        inputs=darknet53_residual_block(inputs, 128, training, data_format)

    routine1=inputs

    inputs=conv2d_fixed_padding(inputs,filters=512,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs=darknet53_residual_block(inputs, 256, training, data_format)

    routine2=inputs

    inputs=conv2d_fixed_padding(inputs,filters=1024,filter_size=3, data_format=data_format, strides=2)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs=darknet53_residual_block(inputs, 512, training, data_format)


    return routine1, routine2, inputs


def yolo_conv_block(inputs, filters, training, data_format):
    '''
    Create the Yolo convolution block

    Argument:
    inputs -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')
    filters -- integer: the number of filters (kernels)
    training -- bool: for batch_normalization layers

    data_format -- string: the format of the tensor input ('channels_last' or 'channels_first')


    return: -- The orperation layer to use after darknet 53
    '''

    inputs=conv2d_fixed_padding(inputs,filters=filters,filter_size=1 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    inputs=conv2d_fixed_padding(inputs,filters=filters*2,filter_size=3 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs=conv2d_fixed_padding(inputs,filters=filters,filter_size=1 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    inputs=conv2d_fixed_padding(inputs,filters=filters*2,filter_size=3 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)


    inputs=conv2d_fixed_padding(inputs,filters=filters,filter_size=1 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    route=inputs

    inputs=conv2d_fixed_padding(inputs,filters=filters*2,filter_size=3 , data_format=data_format)
    inputs=batch_normalization(inputs, training=training, data_format=data_format)
    inputs=tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    return route, inputs

def detection_layer_yolo(inputs, n_classes, anchors, img_size, data_format):
    '''
    Create the detection layer

    Argument:
    inputs -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')
    n_classes -- integer: the number of classes
    anchors -- list: list of the anchor boxes
    img_size-- list: the size of the image
    data_format -- string: the format of the tensor input ('channels_last' or 'channels_first')


    return: -- the tensor contains : box_centers, box_shapes, confidenece, classes
    '''

    n_anchors=len(anchors)
    inputs=tf.layers.conv2d(inputs, filters=n_anchors*(5+n_classes),
                            kernel_size=1, strides=1, use_bias=True, data_format=data_format)


    shape=inputs.get_shape().as_list()

    gird_shape= shape[2:4] if data_format=='channels_first' else shape[1:3]

    if data_format=='channels_first':
        inputs=tf.transpose(inputs, [0, 2, 3, 1])
    
    inputs=tf.reshape(inputs, shape=[-1, n_anchors*gird_shape[0]*gird_shape[1], 5+n_classes])

    scales=(img_size[0]// gird_shape[0], img_size[1]// gird_shape[1])

    box_centers, box_shapes, confidenece, classes= tf.split(inputs, [2,2,1,n_classes], axis=-1)

    x=tf.range(gird_shape[0], dtype=tf.float32)
    y=tf.range(gird_shape[1], dtype=tf.float32)

    x_offset, y_offset=tf.meshgrid(x, y)
    x_offset=tf.reshape(x_offset, (-1, 1))
    y_offset=tf.reshape(y_offset, (-1, 1))

    x_y_offset=tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset=tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset=tf.reshape(x_y_offset, [1, -1, 2])

    box_centers=tf.nn.sigmoid(box_centers)
    box_centers=(box_centers+x_y_offset)*scales

    box_shapes=tf.exp(box_shapes) * tf.to_float(anchors)

    confidenece=tf.nn.sigmoid(confidenece)

    classes=tf.nn.sigmoid(classes)

    inputs=tf.concat([box_centers, box_shapes, confidenece, classes], axis=-1)

    return inputs

def upsample(inputs, out_shape, data_format):
    '''
    Upsamplle to 'out shape'

    Argument:
    inputs -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')
    out_shape -- shape of output
    data_format -- string: the format of the tensor input ('channels_last' or 'channels_first')


    return: -- the tensor 
    '''

    if data_format=='channels_first':
        inputs=tf.reshape(inputs, shape=[0,2,3,1])
        new_height=out_shape[3]
        new_weight=out_shape[2]
    else:
        new_height=out_shape[2]
        new_weight=out_shape[1]

    inputs=tf.image.resize_nearest_neighbor(inputs, (new_height, new_weight))

    if data_format=='channels_first':
        inputs=tf.reshape(inputs, shape=[0,3,1,2])

    return inputs


def build_boxes(inputs):
    '''
    Compute thr left and bottom right points of the boxes

    Argument:
    inputs -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')


    return: -- the tensor 
    '''

    center_x, center_y, width, height, confidenece, classes= tf.split(inputs, [1,1,1,1,1,-1], axis=-1)

    top_left_x=center_x- width/2
    top_left_y=center_y-height/2

    bottom_right_x= center_x + width/2
    bottom_right_y= center_x + height/2

    return tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidenece, classes], axis=-1)


def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_thresold):
    '''
    Perform non-max suppression

    Argument:
    inputs -- tensor: tensor input of size [bath, height, width, channels] ('channels_last')
            or [batch, channels, height, width] ('channels_first')


    return: -- the tensor 
    '''

    batch=tf.unstack(inputs)
    boxes_dicts=[]
    for boxes in batch:
        boxes= tf.boolean_mask(boxes, boxes[:, 4] > confidence_thresold)
        classes= tf.argmax(boxes[:, 5:], axis=-1)
        classes=tf.expand_dims(tf.to_float(classes), axis=-1)
        boxes=tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict=dict()

        for cls in range(n_classes):
            mask=tf.equal(boxes[:, 5:], cls)
            mask_shape=mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes= tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_score, _ =tf.split(class_boxes, [4, 1, -1], axis=-1)
                boxes_conf_score=tf.reshape(boxes_conf_score, [-1])
                indices=tf.image.non_max_suppression(boxes_coords, boxes_conf_score, max_output_size, iou_threshold)

                class_boxes=tf.gather(class_boxes, indices)
                boxes_dict[cls]=class_boxes[:, :5]
        boxes_dicts.append(boxes_dict)

    return boxes_dicts
        
