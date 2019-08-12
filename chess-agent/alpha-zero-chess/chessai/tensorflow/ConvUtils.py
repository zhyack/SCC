import tensorflow as tf
Relu = tf.nn.relu
Tanh = tf.nn.tanh
BatchNormalization = tf.layers.batch_normalization
Dropout = tf.layers.dropout
Dense = tf.layers.dense

def conv2d(x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding, use_bias=False)

def buildGraph(x_image, board_x, board_y, action_size, isTraining, dropout, num_channels):
    print('()@#$(*!@#', isTraining)
    h_conv1 = Relu(BatchNormalization(conv2d(x_image, num_channels, 'same'), axis=3, training=isTraining))     # batch_size  x board_x x board_y x num_channels
    h_conv2 = Relu(BatchNormalization(conv2d(h_conv1, num_channels, 'same'), axis=3, training=isTraining))     # batch_size  x board_x x board_y x num_channels
    h_conv3 = Relu(BatchNormalization(conv2d(h_conv2, num_channels, 'valid'), axis=3, training=isTraining))    # batch_size  x (board_x-2) x (board_y-2) x num_channels
    h_conv4 = Relu(BatchNormalization(conv2d(h_conv3, num_channels, 'valid'), axis=3, training=isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
    h_conv4_flat = tf.reshape(h_conv4, [-1, num_channels*(board_x-4)*(board_y-4)])
    s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, num_channels*2, use_bias=False), axis=1, training=isTraining)), rate=dropout) # batch_size x 1024
    s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, num_channels, use_bias=False), axis=1, training=isTraining)), rate=dropout)         # batch_size x 512
    board_state = s_fc2
  
    return board_state                                       

def calcLoss(pi, v, target_pis, target_vs, lr):
    loss_pi =  tf.losses.softmax_cross_entropy(target_pis, pi)
    loss_v = tf.losses.mean_squared_error(target_vs, tf.reshape(v, shape=[-1,]))
    total_loss = loss_pi + loss_v*2.0
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss)
    return loss_pi, loss_v, train_step