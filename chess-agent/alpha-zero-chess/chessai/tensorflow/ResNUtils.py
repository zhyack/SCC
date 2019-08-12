import tensorflow as tf

def residual_block(inputLayer, filters,kernel_size,stage,block, isTraining):
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    shortcut = inputLayer

    residual_layer = tf.layers.conv2d(inputLayer, filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),name=conv_name+'2a',padding='same',use_bias=False)
    residual_layer = tf.layers.batch_normalization(residual_layer, axis=3, name=bn_name+'2a', training=isTraining)
    residual_layer = tf.nn.relu(residual_layer)
    residual_layer = tf.layers.conv2d(residual_layer, filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),name=conv_name+'2b',padding='same',use_bias=False)
    residual_layer = tf.layers.batch_normalization(residual_layer, axis=3, name=bn_name+'2b', training=isTraining)
    add_shortcut = tf.add(residual_layer, shortcut)
    residual_result = tf.nn.relu(add_shortcut)

    return residual_result

def buildGraph(x_image, board_x, board_y, action_size, valid_mask, isTraining, dropout, num_channels):
    x_image = tf.layers.conv2d(x_image, num_channels, kernel_size=(3, 3), strides=(1, 1),name='conv',padding='same',use_bias=False)
    x_image = tf.layers.batch_normalization(x_image, axis=1, name='conv_bn', training=isTraining)
    x_image = tf.nn.relu(x_image)
    residual_tower = residual_block(inputLayer=x_image, kernel_size=3, filters=num_channels, stage=1, block='a', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=2, block='b', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=3, block='c', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=4, block='d', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=5, block='e', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=6, block='g', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=7, block='h', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=8, block='i', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=9, block='j', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=10, block='k', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=11, block='m', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=12, block='n', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=13, block='o', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=14, block='p', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=15, block='q', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=16, block='r', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=17, block='s', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=18, block='t', isTraining=isTraining)
    residual_tower = residual_block(inputLayer=residual_tower, kernel_size=3, filters=num_channels, stage=19, block='u', isTraining=isTraining)

    board_state = residual_tower

    policy = tf.layers.conv2d(residual_tower, 2,kernel_size=(1, 1), strides=(1, 1),name='pi',padding='same',use_bias=False)
    policy = tf.layers.batch_normalization(policy, axis=3, name='bn_pi', training=isTraining)
    policy = tf.nn.relu(policy)
    policy = tf.layers.flatten(policy, name='p_flatten')
    pi = tf.layers.dense(policy, action_size)
    pi = valid_mask * pi
    prob = tf.nn.softmax(pi)

    value = tf.layers.conv2d(residual_tower, 1,kernel_size=(1, 1), strides=(1, 1),name='v',padding='same',use_bias=False)
    value = tf.layers.batch_normalization(value, axis=3, name='bn_v', training=isTraining)
    value = tf.nn.relu(value)
    value = tf.layers.flatten(value, name='v_flatten')
    value = tf.layers.dense(value, units=256)
    value = tf.nn.relu(value)
    value = tf.layers.dense(value, 1)
    v = tf.nn.tanh(value)

    return board_state, prob, v   