#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import copy
from tensorflow.python.layers import core as layers_core
import math



def modelInitWordEmbedding(dict_size, embedding_size, name='word_embedding_matrix'):
    sqrt3 = math.sqrt(3)
    initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

    word_embedding_matrix = tf.get_variable(name=name, shape=[dict_size, embedding_size], initializer=initializer, dtype=tf.float32)
    return word_embedding_matrix

def modelGetWordEmbedding(word_embedding_matrix, inputs):
    return tf.nn.embedding_lookup(word_embedding_matrix, inputs)

def modelInitBidirectionalEncoder(size, layers, cell_type='LSTM', input_dropout=1.0, output_dropout=1.0):
    cells = []
    for _ in range(layers):
        config_cell = None
        if cell_type in ['LSTM', 'lstm']:
            config_cell = tf.contrib.rnn.BasicLSTMCell(size)
        elif cell_type in ['GRU', 'gru']:
            config_cell = tf.contrib.rnn.GRUCell(size)
        else:
            config_cell = tf.contrib.rnn.BasicRNNCell(size)
        config_cell = tf.contrib.rnn.DropoutWrapper(config_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
        cells.append(config_cell)
    encoder_fw_cell = copy.deepcopy(cells)
    encoder_bw_cell = copy.deepcopy(cells)
    
    return (encoder_fw_cell, encoder_bw_cell)

def modelRunBidirectionalEncoder(cells, encoder_inputs, inputs_lengths):
    # input: batch_size * len * embedding_size
    # length: batch_size
    # outputs: batch_size * len * embedding_size
    encoder_fw_cell, encoder_bw_cell = cells
    encoder_outputs, encoder_fw_states, encoder_bw_states = None, None, None
    
    (encoder_outputs, encoder_fw_states, encoder_bw_states) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=encoder_fw_cell, cells_bw=encoder_bw_cell, inputs=encoder_inputs, sequence_length=inputs_lengths, dtype=tf.float32)

    encoder_outputs_fw, encoder_outputs_bw = tf.split(encoder_outputs, 2, -1)
    encoder_outputs = (encoder_outputs_fw+encoder_outputs_bw)/2.0

    encoder_states = None
    encoder_state_c = (encoder_fw_states[-1].c+encoder_bw_states[-1].c)/2.0
    encoder_state_h = (encoder_fw_states[-1].h+encoder_bw_states[-1].h)/2.0
    encoder_states = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

    return encoder_outputs, encoder_states

def modelInitAttentionDecoderCell(size, layers, cell_type, input_dropout, output_dropout, encoder_outputs, encoder_outputs_lengths, att_type='LUONG'):
    cells = []
    for _ in range(layers):
        config_cell = None
        if cell_type in ['LSTM', 'lstm']:
            config_cell = tf.contrib.rnn.BasicLSTMCell(size)
        elif cell_type in ['GRU', 'gru']:
            config_cell = tf.contrib.rnn.GRUCell(size)
        else:
            config_cell = tf.contrib.rnn.BasicRNNCell(size)
        config_cell = tf.contrib.rnn.DropoutWrapper(config_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
        cells.append(config_cell)
    op_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    attention_mechanism = None
    if att_type=='LUONG':
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(size, encoder_outputs, memory_sequence_length=encoder_outputs_lengths, scale=True)
    elif att_type=='BAHDANAU':
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(size, encoder_outputs, memory_sequence_length=encoder_outputs_lengths, normalize=True)
    else:
        raise Exception('Unknown attention type.')

    op_cell = tf.contrib.seq2seq.AttentionWrapper(op_cell, attention_mechanism, attention_layer_size=size, output_attention=True)

    return op_cell

def modelRunDecoderForTrain(cell, inputs, inputs_lengths, initial_state,  output_projection_layer):
    # input: batch_size * len * hidden_size
    # length: batch_size
    # output: batch_size * len * output_size
    train_helper = tf.contrib.seq2seq.TrainingHelper(inputs, inputs_lengths, time_major=False)
    decoder=tf.contrib.seq2seq.BasicDecoder(cell, helper=train_helper, initial_state=initial_state, output_layer=output_projection_layer)
    outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=None)
    return outputs.rnn_output

def modelRunDecoderForGreedyInfer(cell, inputs, word_embedding_matrix, id_end, max_len, initial_state, output_projection_layer):
    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embedding_matrix, inputs, id_end)
    decoder=tf.contrib.seq2seq.BasicDecoder(cell, helper=infer_helper, initial_state=initial_state, output_layer=output_projection_layer)
    outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False, maximum_iterations=max_len*2)
    return outputs.sample_id

def modelRunDecoderForBSInfer(cell, inputs, word_embedding_matrix, beam_width, id_end, max_len, initial_state, output_projection_layer):
    decoder=tf.contrib.seq2seq.BeamSearchDecoder(cell, word_embedding_matrix, inputs, id_end, initial_state=initial_state, beam_width=beam_width, output_layer=output_projection_layer, length_penalty_weight=1.2)
    outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False, maximum_iterations=max_len*2)
    return tf.transpose(outputs.predicted_ids, [2,0,1])[0]

def Dense(size, use_bias=False):
    return layers_core.Dense(size, use_bias=use_bias)

def updateBP(loss, lr, var_list, optimizer, norm=None):
    gradients = [tf.gradients(loss, var_list[i]) for i in range(len(lr))]
    if norm!=None:
        gradients = [tf.clip_by_global_norm(gradients[i], norm)[0] for i in range(len(lr))]
    optimizers = [optimizer(lr[i]) for i in range(len(lr))]
    return [optimizers[i].apply_gradients(zip(gradients[i], var_list[i])) for i in range(len(lr))]