import sys
sys.path.append('../../')
from utils import *

from ..ChessLogic import fsize
from .ConvUtils import buildGraph as ConvGraph
from .ConvUtils import calcLoss as chessCalcLoss
from .ResNUtils import buildGraph as ResGraph
from . import ComUtils


import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

class ChessNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():

######################Chess AI Module##############################

            # global common placeholder
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y, fsize])
            self.valid_mask = tf.placeholder(tf.float32, shape=[None, self.action_size])
            # chess special placeholder
            self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
            self.target_vs = tf.placeholder(tf.float32, shape=[None])
            if 'chess' not in args['models']:
                raise Exception('Please define your chess model in main.py: args[\'models\'][\'chess\']')
            
            # define chess operations
            if args['models']['chess']=='Conv':
                self.chess_graph = ConvGraph
            elif args['models']['chess']=='Res':
                self.chess_graph = ResGraph
            self.chess_calc_loss = chessCalcLoss


            # calc chess state, p, v
            print("!?#!#!@# ", self.args['is_train'])
            self.board_state = self.chess_graph(self.input_boards, self.board_x, self.board_y, self.action_size, self.args['is_train'], self.args['dropout'], self.args['num_channels'])

            self.action_pi = tf.layers.dense(self.board_state, self.action_size) * self.valid_mask
            self.state_value = tf.nn.tanh(tf.layers.dense(self.board_state, 1))
            self.action_prob = tf.nn.softmax(self.action_pi)

            # calc chess loss, update
            self.chess_loss_pi, self.chess_loss_v, self.train_op_chess = self.chess_calc_loss(self.action_pi, self.state_value, self.target_pis, self.target_vs, self.args['lr'])

            self.chess_ai_variables = tf.trainable_variables()
            print('ChessAI: ', self.chess_ai_variables)

#################Common Module For Text Generation#########################
            if "comment" in self.args['models']:
                # text generation placeholder
                self.input_action_descs = tf.placeholder(tf.int32, shape=[None, self.args['action_desc_length']])
                self.decoder_outputs = tf.placeholder(tf.int32, shape=[None, None])
                self.decoder_targets = tf.placeholder(tf.int32, shape=[None, None])
                self.decoder_targets_length = tf.placeholder(tf.int32, shape=[None])
                self.decoder_targets_mask = tf.placeholder(tf.float32, shape=[None, None])
                
                # text generation parameters
                self.input_size = self.args['input_size']
                self.embedding_size = self.args['num_channels']

                # input embeddings            
                with tf.variable_scope("ActionWordEmbedding") as scope:
                    self.action_word_embedding_matrix = ComUtils.modelInitWordEmbedding(self.input_size, self.embedding_size, name='we_input')

                    self.action_word_embedding_variables = scope.trainable_variables()
                    print('ActionWordEmbedding: ', self.action_word_embedding_variables)

                # for input hierarchical embeddings
                with tf.variable_scope("ActionEncoder") as scope:
                    self.action_encoder_cell = ComUtils.modelInitBidirectionalEncoder(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'])

                    encoder_inputs_example = ComUtils.modelGetWordEmbedding(self.action_word_embedding_matrix, self.input_action_descs)
                    encoder_inputs_len_example = None
                    ComUtils.modelRunBidirectionalEncoder(self.action_encoder_cell, encoder_inputs_example, encoder_inputs_len_example)
                    self.action_encoder_variables = scope.trainable_variables()
                    print('ActionEncoder: ', self.action_encoder_variables)
                    scope.reuse_variables()

                # for encoding comparative states
                with tf.variable_scope("StateWithDiff") as scope:
                    self.state_diff_projection_layer = ComUtils.Dense(self.embedding_size, use_bias=False)
                    state_1, state_2 = tf.split(self.board_state, 2)
                    value_1, value_2 = tf.split(self.state_value, 2)
                    state_3 = tf.concat([state_1, state_2, (-value_2-value_1)/2.0], 1)
                    vec_3 = self.state_diff_projection_layer(state_3)
                    scope.reuse_variables()
                    self.state_diff_variables = scope.trainable_variables()
                    print('StateWithDiff: ', self.state_diff_variables)

                # for encoding state with value
                with tf.variable_scope("StateWithValue") as scope:
                    self.state_value_projection_layer = ComUtils.Dense(self.embedding_size, use_bias=False)
                    state = tf.concat([self.board_state, self.state_value], 1)
                    vec = self.state_value_projection_layer(state)
                    scope.reuse_variables()
                    self.state_value_variables = scope.trainable_variables()
                    print('StateWithValue: ', self.state_value_variables)

                

########################Comment Model *A*################################
                if "A" in self.args['models']['comment']:
                    # for Direct Move Description generation
                    with tf.variable_scope("CommentModelA") as scope:
                        # encode a_t
                        with tf.variable_scope("ActionEncoder") as sub_scope:
                            encoder_inputs = ComUtils.modelGetWordEmbedding(self.action_word_embedding_matrix, self.input_action_descs)
                            encoder_inputs_len = tf.convert_to_tensor([self.args['action_desc_length']]*self.args['batch_size'], dtype=tf.int32)
                            encoder_outputs, encoder_states = ComUtils.modelRunBidirectionalEncoder(self.action_encoder_cell, encoder_inputs, encoder_inputs_len) # [batch_size, len, hidden_size]
                        
                        self.state_projection_layer_a = ComUtils.Dense(self.embedding_size, use_bias=False)
                        # add s_t
                        board_vec = self.state_projection_layer_a(self.board_state)
                        board_vec = tf.reshape(board_vec, [-1,1,self.embedding_size]) #[batch_size, 1, hidden_size]
                        if 'test_with_no_ai' in self.args and self.args['test_with_no_ai']:
                            decoder_contexts = encoder_outputs #[batch_size, len, hidden_size]
                            encoder_inputs_len_for_att = encoder_inputs_len # [batch_size]
                        else:
                            decoder_contexts = tf.concat([encoder_outputs, board_vec], 1) #[batch_size, len+1, hidden_size]
                            encoder_inputs_len_for_att = encoder_inputs_len+1 # [batch_size]
                        

                        # init decoder
                        self.output_size_a = self.args['output_size_A']
                        self.output_word_embedding_matrix_a = ComUtils.modelInitWordEmbedding(self.output_size_a, self.embedding_size, name='we_output_a')

                        if not self.args['is_train'] and self.args['use_bs']:
                            encoder_states = seq2seq.tile_batch(encoder_states, self.args['beam_width'])
                            encoder_outputs = seq2seq.tile_batch(encoder_outputs, self.args['beam_width'])
                            encoder_inputs_len_for_att = seq2seq.tile_batch(encoder_inputs_len_for_att, self.args['beam_width'])
                        
                        self.decoder_a_cell = ComUtils.modelInitAttentionDecoderCell(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'], encoder_outputs, encoder_inputs_len_for_att, 'LUONG')

                        if not self.args['is_train'] and self.args['use_bs']:
                            initial_state = self.decoder_a_cell.zero_state(batch_size=self.args['batch_size']*self.args['beam_width'], dtype=tf.float32)
                        else:
                            initial_state = self.decoder_a_cell.zero_state(batch_size=self.args['batch_size'], dtype=tf.float32)
                        cat_state = tuple([encoder_states] + list(initial_state.cell_state)[:-1])
                        initial_state.clone(cell_state=cat_state)

                        self.output_projection_layer_a = ComUtils.Dense(self.output_size_a, use_bias=False)


                        decoder_inputs_embedded = ComUtils.modelGetWordEmbedding(self.output_word_embedding_matrix_a, self.decoder_outputs) #[batch_size, len, hidden_size]

                        
                        # run decoder
                        if args['is_train']:
                            self.train_outputs_a = ComUtils.modelRunDecoderForTrain(self.decoder_a_cell, decoder_inputs_embedded, self.decoder_targets_length, initial_state, self.output_projection_layer_a) # [batch_size, len, output_size]

                            self.loss_a = seq2seq.sequence_loss(logits=self.train_outputs_a, targets=self.decoder_targets, weights=self.decoder_targets_mask)
                        
                        if not self.args['is_train'] and self.args['use_bs']:
                            self.infer_outputs_a = ComUtils.modelRunDecoderForBSInfer(self.decoder_a_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_a, self.args['beam_width'], self.args['id_eos_A'], self.args['max_text_length'], initial_state, self.output_projection_layer_a)
                        else:
                            self.infer_outputs_a = ComUtils.modelRunDecoderForGreedyInfer(self.decoder_a_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_a, self.args['id_eos_A'], self.args['max_text_length'], initial_state, self.output_projection_layer_a)

                        self.model_a_variables = scope.trainable_variables()
                        print('CommentModelA: ', self.model_a_variables)

                        if args['is_train']:
                            optimizer = tf.train.AdamOptimizer
                            if self.args['text_op_A']=='GD':
                                optimizer = tf.train.GradientDescentOptimizer
                            self.train_op_a = ComUtils.updateBP(self.loss_a, [self.args['text_lr_A']], [self.model_a_variables + self.action_encoder_variables + self.action_word_embedding_variables], optimizer, norm=self.args['text_norm_A'])

########################Comment Model *B*################################
                if "B" in self.args['models']['comment']:
                    with tf.variable_scope("CommentModelB") as scope:
                        # encode a_t
                        with tf.variable_scope("ActionEncoder") as sub_scope:
                            encoder_inputs = ComUtils.modelGetWordEmbedding(self.action_word_embedding_matrix, self.input_action_descs)
                            encoder_inputs_len = tf.convert_to_tensor([self.args['action_desc_length']]*self.args['batch_size'], dtype=tf.int32)
                            encoder_outputs, encoder_states = ComUtils.modelRunBidirectionalEncoder(self.action_encoder_cell, encoder_inputs, encoder_inputs_len) # [batch_size, len, hidden_size]
                        
                        
                        # add s_t
                        board_state_1, board_state_2 = tf.split(self.board_state, 2) # [batch_size, hidden_size]
                        state_value_1, state_value_2 = tf.split(self.state_value, 2) #[batch_size, 1]
                        board_state_3 = tf.concat([board_state_1, board_state_2, (-state_value_2-state_value_1)/2.0], 1) # [batch_size, hidden_size*2+1]
                        board_state_1 = tf.concat([board_state_1, state_value_1], 1) # [batch_size, hidden_size+1]
                        board_state_2 = tf.concat([board_state_2, state_value_2], 1) # [batch_size, hidden_size+1]
                        with tf.variable_scope("StateWithDiff") as sub_scope:
                            board_vec_3 = self.state_diff_projection_layer(board_state_3) # [batch_size, hidden_size]
                            board_vec_3 = tf.reshape(board_vec_3, [-1,1,self.embedding_size]) # [batch_size, 1, hidden_size]
                        with tf.variable_scope("StateWithValue") as sub_scope:
                            board_vec_1 = self.state_value_projection_layer(board_state_1) # [batch_size, hidden_size]
                            board_vec_2 = self.state_value_projection_layer(board_state_2) # [batch_size, hidden_size]
                            board_vec_1 = tf.reshape(board_vec_1, [-1,1,self.embedding_size]) # [batch_size, 1, hidden_size]
                            board_vec_2 = tf.reshape(board_vec_2, [-1,1,self.embedding_size]) # [batch_size, 1, hidden_size]

                        # decoder_contexts = tf.concat([encoder_outputs, board_vec_1, board_vec_2, board_vec_3], 1) #[batch_size, len+3, hidden_size]
                        # encoder_inputs_len_for_att = encoder_inputs_len+3 # [batch_size]
                        decoder_contexts = tf.concat([encoder_outputs, board_vec_3], 1) #[batch_size, len+3, hidden_size]
                        encoder_inputs_len_for_att = encoder_inputs_len+1 # [batch_size]

                        # init decoder
                        self.output_size_b = self.args['output_size_B']
                        self.output_word_embedding_matrix_b = ComUtils.modelInitWordEmbedding(self.output_size_b, self.embedding_size, name='we_output_b')
                        if not self.args['is_train'] and self.args['use_bs']:
                            encoder_states = seq2seq.tile_batch(encoder_states, self.args['beam_width'])
                            encoder_outputs = seq2seq.tile_batch(encoder_outputs, self.args['beam_width'])
                            encoder_inputs_len_for_att = seq2seq.tile_batch(encoder_inputs_len_for_att, self.args['beam_width'])
                        
                        self.decoder_b_cell = ComUtils.modelInitAttentionDecoderCell(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'], encoder_outputs, encoder_inputs_len_for_att, 'LUONG')

                        if not self.args['is_train'] and self.args['use_bs']:
                            initial_state = self.decoder_b_cell.zero_state(batch_size=self.args['batch_size']*self.args['beam_width'], dtype=tf.float32)
                        else:
                            initial_state = self.decoder_b_cell.zero_state(batch_size=self.args['batch_size'], dtype=tf.float32)
                        cat_state = tuple([encoder_states] + list(initial_state.cell_state)[:-1])
                        # initial_state.clone(cell_state=cat_state)

                        self.output_projection_layer_b = ComUtils.Dense(self.output_size_b, use_bias=False)


                        decoder_inputs_embedded = ComUtils.modelGetWordEmbedding(self.output_word_embedding_matrix_b, self.decoder_outputs) #[batch_size, len, hidden_size]

                        
                        # run decoder
                        if args['is_train']:
                            self.train_outputs_b = ComUtils.modelRunDecoderForTrain(self.decoder_b_cell, decoder_inputs_embedded, self.decoder_targets_length, initial_state, self.output_projection_layer_b) # [batch_size, len, output_size]

                            self.loss_b = seq2seq.sequence_loss(logits=self.train_outputs_b, targets=self.decoder_targets, weights=self.decoder_targets_mask)
                        
                        if not self.args['is_train'] and self.args['use_bs']:
                            self.infer_outputs_b = ComUtils.modelRunDecoderForBSInfer(self.decoder_b_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_b, self.args['beam_width'], self.args['id_eos_B'], self.args['max_text_length'], initial_state, self.output_projection_layer_b)
                        else:
                            self.infer_outputs_b = ComUtils.modelRunDecoderForGreedyInfer(self.decoder_b_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_b, self.args['id_eos_B'], self.args['max_text_length'], initial_state, self.output_projection_layer_b)

                        self.model_b_variables = scope.trainable_variables()
                        print('CommentModelB: ', self.model_b_variables)

                        if args['is_train']:
                            optimizer = tf.train.AdamOptimizer
                            if self.args['text_op_B']=='GD':
                                optimizer = tf.train.GradientDescentOptimizer
                            self.train_op_b = ComUtils.updateBP(self.loss_b, [self.args['text_lr_B']], [self.model_b_variables, self.state_diff_variables], optimizer, norm=self.args['text_norm_B'])

########################Comment Model *C*################################
                if "C" in self.args['models']['comment']:
                    with tf.variable_scope("CommentModelC") as scope:
                        # encode a_t, s_t
                        decoder_contexts_list = []
                        decoder_vecs_list = []
                        encoder_inputs_len_for_att_list = []

                        self.state_projection_layer_c = ComUtils.Dense(self.embedding_size, use_bias=False)

                        for each_input_action_descs, each_board_state, each_state_value in zip(tf.split(self.input_action_descs, 2), tf.split(self.board_state, 2), tf.split(self.state_value, 2)):

                            with tf.variable_scope("ActionEncoder") as sub_scope:
                                encoder_inputs = ComUtils.modelGetWordEmbedding(self.action_word_embedding_matrix, each_input_action_descs)
                                encoder_inputs_len = tf.convert_to_tensor([self.args['action_desc_length']]*self.args['batch_size'], dtype=tf.int32)
                                encoder_outputs, encoder_states = ComUtils.modelRunBidirectionalEncoder(self.action_encoder_cell, encoder_inputs, encoder_inputs_len) # [batch_size, len, hidden_size]
                            with tf.variable_scope("StateWithValue") as sub_scope:
                                board_state = tf.concat([each_board_state, each_state_value], 1)
                                board_vec = self.state_projection_layer_c(board_state)
                                decoder_vecs_list.append(board_vec) #[batch_size,hidden_size]
                                board_vec = tf.reshape(board_vec, [-1,1,self.embedding_size]) #[batch_size, 1, hidden_size]
                            decoder_contexts_list.append(tf.concat([encoder_outputs, board_vec], 1)) #[batch_size, len+1, hidden_size]
                            encoder_inputs_len_for_att_list.append(encoder_inputs_len+1) # [batch_size]
                        decoder_contexts = tf.concat(decoder_contexts_list, 0) # [batch_size*high_layer_num, len+1, hidden_size]
                        encoder_inputs_len_for_att = tf.concat(encoder_inputs_len_for_att_list, 0) # [batch_size*high_layer_num]
                        decoder_vecs = tf.convert_to_tensor(decoder_vecs_list) # [high_layer_num, batch_size, hidden_size]
                        self.attention_W_c = tf.get_variable('attention_W', [self.embedding_size], dtype=tf.float32)
                        attention_scores = tf.einsum('ijk,k->ji', decoder_vecs, self.attention_W_c) #[high_layer_num, batch_size, hidden_size] * [hidden_size] = [batch_size, high_layer_num]
                        
                        
                        # init decoder
                        self.output_size_c = self.args['output_size_C']
                        self.output_word_embedding_matrix_c = ComUtils.modelInitWordEmbedding(self.output_size_c, self.embedding_size, name='we_output')
                        if not self.args['is_train'] and self.args['use_bs']:
                            encoder_states = seq2seq.tile_batch(encoder_states, self.args['beam_width'])
                            decoder_contexts = seq2seq.tile_batch(decoder_contexts, self.args['beam_width'])
                            encoder_inputs_len_for_att = seq2seq.tile_batch(encoder_inputs_len_for_att, self.args['beam_width'])
                            attention_scores = seq2seq.tile_batch(attention_scores, self.args['beam_width'])
                        
                        self.decoder_c_cell = ComUtils.modelInitHierarchicalAttentionDecoderCell(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'], decoder_contexts, encoder_inputs_len_for_att, 2, attention_scores, 'LUONG')

                        if not self.args['is_train'] and self.args['use_bs']:
                            initial_state = self.decoder_c_cell.zero_state(batch_size=self.args['batch_size']*self.args['beam_width'], dtype=tf.float32)
                        else:
                            initial_state = self.decoder_c_cell.zero_state(batch_size=self.args['batch_size'], dtype=tf.float32)
                        cat_state = tuple([encoder_states] + list(initial_state.cell_state)[:-1])
                        initial_state.clone(cell_state=cat_state)

                        self.output_projection_layer_c = ComUtils.Dense(self.output_size_c, use_bias=False)


                        decoder_inputs_embedded = ComUtils.modelGetWordEmbedding(self.output_word_embedding_matrix_c, self.decoder_outputs) #[batch_size, len, hidden_size]

                        
                        # run decoder
                        if args['is_train']:
                            self.train_outputs_c = ComUtils.modelRunDecoderForTrain(self.decoder_c_cell, decoder_inputs_embedded, self.decoder_targets_length, initial_state, self.output_projection_layer_c) # [batch_size, len, output_size]

                            self.loss_c = seq2seq.sequence_loss(logits=self.train_outputs_c, targets=self.decoder_targets, weights=self.decoder_targets_mask)
                        
                        if not self.args['is_train'] and self.args['use_bs']:
                            self.infer_outputs_c = ComUtils.modelRunDecoderForBSInfer(self.decoder_c_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_c, self.args['beam_width'], self.args['id_eos_C'], self.args['max_text_length'], initial_state, self.output_projection_layer_c)
                        else:
                            self.infer_outputs_c = ComUtils.modelRunDecoderForGreedyInfer(self.decoder_c_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_c, self.args['id_eos_C'], self.args['max_text_length'], initial_state, self.output_projection_layer_c)

                        self.model_c_variables = scope.trainable_variables()
                        print('CommentModelC: ', self.model_c_variables)

                        if args['is_train']:
                            optimizer = tf.train.AdamOptimizer
                            if self.args['text_op_C']=='GD':
                                optimizer = tf.train.GradientDescentOptimizer
                            self.train_op_c = ComUtils.updateBP(self.loss_c, [self.args['text_lr_C']], [self.model_c_variables ], optimizer, norm=self.args['text_norm_C'])

########################Comment Model *D*################################
                if "D" in self.args['models']['comment']:
                    with tf.variable_scope("CommentModelD") as scope:
                        # encode a_t, s_t
                        decoder_contexts_list = []
                        decoder_vecs_list = []
                        encoder_inputs_len_for_att_list = []

                        self.state_projection_layer_d = ComUtils.Dense(self.embedding_size, use_bias=False)

                        for each_input_action_descs, each_board_state, each_state_value in zip(tf.split(self.input_action_descs, 3), tf.split(self.board_state, 3), tf.split(self.state_value, 3)):

                            with tf.variable_scope("ActionEncoder") as sub_scope:
                                encoder_inputs = ComUtils.modelGetWordEmbedding(self.action_word_embedding_matrix, each_input_action_descs)
                                encoder_inputs_len = tf.convert_to_tensor([self.args['action_desc_length']]*self.args['batch_size'], dtype=tf.int32)
                                encoder_outputs, encoder_states = ComUtils.modelRunBidirectionalEncoder(self.action_encoder_cell, encoder_inputs, encoder_inputs_len) # [batch_size, len, hidden_size]
                            with tf.variable_scope("StateWithValue") as sub_scope:
                                board_state = tf.concat([each_board_state, each_state_value], 1)
                                board_vec = self.state_projection_layer_d(board_state)
                                decoder_vecs_list.append(board_vec) #[batch_size,hidden_size]
                                board_vec = tf.reshape(board_vec, [-1,1,self.embedding_size]) #[batch_size, 1, hidden_size]
                            decoder_contexts_list.append(tf.concat([encoder_outputs, board_vec], 1)) #[batch_size, len+1, hidden_size]
                            encoder_inputs_len_for_att_list.append(encoder_inputs_len+1) # [batch_size]
                        decoder_contexts = tf.concat(decoder_contexts_list, 0) # [batch_size*high_layer_num, len+1, hidden_size]
                        encoder_inputs_len_for_att = tf.concat(encoder_inputs_len_for_att_list, 0) # [batch_size*high_layer_num]
                        decoder_vecs = tf.convert_to_tensor(decoder_vecs_list) # [high_layer_num, batch_size, hidden_size]
                        self.attention_W_d = tf.get_variable('attention_W', [self.embedding_size], dtype=tf.float32)
                        attention_scores = tf.einsum('ijk,k->ji', decoder_vecs, self.attention_W_d) #[high_layer_num, batch_size, hidden_size] * [hidden_size] = [batch_size, high_layer_num]
                        
                        
                        # init decoder
                        self.output_size_d = self.args['output_size_D']
                        self.output_word_embedding_matrix_d = ComUtils.modelInitWordEmbedding(self.output_size_d, self.embedding_size, name='we_output')
                        if not self.args['is_train'] and self.args['use_bs']:
                            encoder_states = seq2seq.tile_batch(encoder_states, self.args['beam_width'])
                            decoder_contexts = seq2seq.tile_batch(decoder_contexts, self.args['beam_width'])
                            encoder_inputs_len_for_att = seq2seq.tile_batch(encoder_inputs_len_for_att, self.args['beam_width'])
                            attention_scores = seq2seq.tile_batch(attention_scores, self.args['beam_width'])
                        
                        self.decoder_d_cell = ComUtils.modelInitHierarchicalAttentionDecoderCell(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'], decoder_contexts, encoder_inputs_len_for_att, 3, attention_scores, 'LUONG')

                        if not self.args['is_train'] and self.args['use_bs']:
                            initial_state = self.decoder_d_cell.zero_state(batch_size=self.args['batch_size']*self.args['beam_width'], dtype=tf.float32)
                        else:
                            initial_state = self.decoder_d_cell.zero_state(batch_size=self.args['batch_size'], dtype=tf.float32)
                        cat_state = tuple([encoder_states] + list(initial_state.cell_state)[:-1])
                        initial_state.clone(cell_state=cat_state)

                        self.output_projection_layer_d = ComUtils.Dense(self.output_size_d, use_bias=False)


                        decoder_inputs_embedded = ComUtils.modelGetWordEmbedding(self.output_word_embedding_matrix_d, self.decoder_outputs) #[batch_size, len, hidden_size]

                        
                        # run decoder
                        if args['is_train']:
                            self.train_outputs_d = ComUtils.modelRunDecoderForTrain(self.decoder_d_cell, decoder_inputs_embedded, self.decoder_targets_length, initial_state, self.output_projection_layer_d) # [batch_size, len, output_size]

                            self.loss_d = seq2seq.sequence_loss(logits=self.train_outputs_d, targets=self.decoder_targets, weights=self.decoder_targets_mask)
                        
                        if not self.args['is_train'] and self.args['use_bs']:
                            self.infer_outputs_d = ComUtils.modelRunDecoderForBSInfer(self.decoder_d_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_d, self.args['beam_width'], self.args['id_eos_D'], self.args['max_text_length'], initial_state, self.output_projection_layer_d)
                        else:
                            self.infer_outputs_d = ComUtils.modelRunDecoderForGreedyInfer(self.decoder_d_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_d, self.args['id_eos_D'], self.args['max_text_length'], initial_state, self.output_projection_layer_d)

                        self.model_d_variables = scope.trainable_variables()
                        print('CommentModelD: ', self.model_d_variables)

                        if args['is_train']:
                            optimizer = tf.train.AdamOptimizer
                            if self.args['text_op_D']=='GD':
                                optimizer = tf.train.GradientDescentOptimizer
                            self.train_op_d = ComUtils.updateBP(self.loss_d, [self.args['text_lr_D']], [self.model_d_variables ], optimizer, norm=self.args['text_norm_D'])

########################Comment Model *E* (older version)################################
                # if "E" in self.args['models']['comment']:
                    # with tf.variable_scope("CommentModelE") as scope:
                    #     # encode a_t, s_t
                    #     encoder_vecs_list = []

                    #     self.state_projection_layer_e = ComUtils.Dense(self.embedding_size, use_bias=False)

                    #     for each_board_state, each_state_value in zip(tf.split(self.board_state, 5), tf.split(self.state_value, 5)):

                    #         with tf.variable_scope("StateWithValue") as sub_scope:
                    #             board_state = tf.concat([each_board_state, each_state_value], 1)
                    #             board_vec = self.state_projection_layer_e(board_state)
                    #             encoder_vecs_list.append(board_vec) #[batch_size,hidden_size]

                    #     encoder_vecs = tf.convert_to_tensor(encoder_vecs_list) # [high_layer_num, batch_size, hidden_size]
                    #     encoder_vecs = tf.reshape(encoder_vecs, [-1, 5, self.embedding_size])

                    #     # for sequential board states embeddings
                    #     self.state_encoder_cell_e = ComUtils.modelInitBidirectionalEncoder(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'])

                    #     encoder_inputs = encoder_vecs
                    #     encoder_inputs_len = tf.convert_to_tensor([5]*self.args['batch_size'], dtype=tf.int32)
                    #     encoder_outputs, encoder_states = ComUtils.modelRunBidirectionalEncoder(self.state_encoder_cell_e, encoder_inputs, encoder_inputs_len)     
                    #     encoder_inputs_len_for_att = encoder_inputs_len                   
                        
                    #     # init decoder
                    #     self.output_size_e = self.args['output_size_E']
                    #     self.output_word_embedding_matrix_e = ComUtils.modelInitWordEmbedding(self.output_size_e, self.embedding_size, name='we_output_e')

                    #     if not self.args['is_train'] and self.args['use_bs']:
                    #         encoder_states = seq2seq.tile_batch(encoder_states, self.args['beam_width'])
                    #         encoder_outputs = seq2seq.tile_batch(encoder_outputs, self.args['beam_width'])
                    #         encoder_inputs_len_for_att = seq2seq.tile_batch(encoder_inputs_len_for_att, self.args['beam_width'])
                        
                    #     self.decoder_e_cell = ComUtils.modelInitAttentionDecoderCell(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'], encoder_outputs, encoder_inputs_len_for_att, 'LUONG')

                    #     if not self.args['is_train'] and self.args['use_bs']:
                    #         initial_state = self.decoder_e_cell.zero_state(batch_size=self.args['batch_size']*self.args['beam_width'], dtype=tf.float32)
                    #     else:
                    #         initial_state = self.decoder_e_cell.zero_state(batch_size=self.args['batch_size'], dtype=tf.float32)
                    #     cat_state = tuple([encoder_states] + list(initial_state.cell_state)[:-1])
                    #     initial_state.clone(cell_state=cat_state)

                    #     self.output_projection_layer_e = ComUtils.Dense(self.output_size_e, use_bias=False)


                    #     decoder_inputs_embedded = ComUtils.modelGetWordEmbedding(self.output_word_embedding_matrix_e, self.decoder_outputs) #[batch_size, len, hidden_size]

                        
                    #     # run decoder
                    #     if args['is_train']:
                    #         self.train_outputs_e = ComUtils.modelRunDecoderForTrain(self.decoder_e_cell, decoder_inputs_embedded, self.decoder_targets_length, initial_state, self.output_projection_layer_e) # [batch_size, len, output_size]

                    #         self.loss_e = seq2seq.sequence_loss(logits=self.train_outputs_e, targets=self.decoder_targets, weights=self.decoder_targets_mask)
                        
                    #     if not self.args['is_train'] and self.args['use_bs']:
                    #         self.infer_outputs_e = ComUtils.modelRunDecoderForBSInfer(self.decoder_e_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_e, self.args['beam_width'], self.args['id_eos_E'], self.args['max_text_length'], initial_state, self.output_projection_layer_e)
                    #     else:
                    #         self.infer_outputs_e = ComUtils.modelRunDecoderForGreedyInfer(self.decoder_e_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_e, self.args['id_eos_E'], self.args['max_text_length'], initial_state, self.output_projection_layer_e)

                    #     self.model_e_variables = scope.trainable_variables()
                    #     print('CommentModelE: ', self.model_e_variables)

                    #     if args['is_train']:
                    #         optimizer = tf.train.AdamOptimizer
                    #         if self.args['text_op_E']=='GD':
                    #             optimizer = tf.train.GradientDescentOptimizer
                    #         self.train_op_e = ComUtils.updateBP(self.loss_e, [self.args['text_lr_E']], [self.model_e_variables + self.state_value_variables], optimizer, norm=self.args['text_norm_E'])


########################Comment Model *E*################################
                if "E" in self.args['models']['comment']:
                    with tf.variable_scope("CommentModelE") as scope:
                        # encode a_t, s_t
                        decoder_contexts_list = []
                        decoder_vecs_list = []
                        encoder_inputs_len_for_att_list = []

                        self.state_projection_layer_e = ComUtils.Dense(self.embedding_size, use_bias=False)

                        for each_input_action_descs, each_board_state, each_state_value in zip(tf.split(self.input_action_descs, 5), tf.split(self.board_state, 5), tf.split(self.state_value, 5)):

                            with tf.variable_scope("ActionEncoder") as sub_scope:
                                encoder_inputs = ComUtils.modelGetWordEmbedding(self.action_word_embedding_matrix, each_input_action_descs)
                                encoder_inputs_len = tf.convert_to_tensor([self.args['action_desc_length']]*self.args['batch_size'], dtype=tf.int32)
                                encoder_outputs, encoder_states = ComUtils.modelRunBidirectionalEncoder(self.action_encoder_cell, encoder_inputs, encoder_inputs_len) # [batch_size, len, hidden_size]
                            with tf.variable_scope("StateWithValue") as sub_scope:
                                board_state = tf.concat([each_board_state, each_state_value], 1)
                                board_vec = self.state_projection_layer_e(board_state)
                                decoder_vecs_list.append(board_vec) #[batch_size,hidden_size]
                                board_vec = tf.reshape(board_vec, [-1,1,self.embedding_size]) #[batch_size, 1, hidden_size]
                            decoder_contexts_list.append(tf.concat([encoder_outputs, board_vec], 1)) #[batch_size, len+1, hidden_size]
                            encoder_inputs_len_for_att_list.append(encoder_inputs_len+1) # [batch_size]
                        decoder_contexts = tf.concat(decoder_contexts_list, 0) # [batch_size*high_layer_num, len+1, hidden_size]
                        encoder_inputs_len_for_att = tf.concat(encoder_inputs_len_for_att_list, 0) # [batch_size*high_layer_num]
                        decoder_vecs = tf.convert_to_tensor(decoder_vecs_list) # [high_layer_num, batch_size, hidden_size]
                        self.attention_W_e = tf.get_variable('attention_W', [self.embedding_size], dtype=tf.float32)
                        attention_scores = tf.einsum('ijk,k->ji', decoder_vecs, self.attention_W_e) #[high_layer_num, batch_size, hidden_size] * [hidden_size] = [batch_size, high_layer_num]
                        
                        
                        # init decoder
                        self.output_size_e = self.args['output_size_E']
                        self.output_word_embedding_matrix_e = ComUtils.modelInitWordEmbedding(self.output_size_e, self.embedding_size, name='we_output')
                        if not self.args['is_train'] and self.args['use_bs']:
                            encoder_states = seq2seq.tile_batch(encoder_states, self.args['beam_width'])
                            decoder_contexts = seq2seq.tile_batch(decoder_contexts, self.args['beam_width'])
                            encoder_inputs_len_for_att = seq2seq.tile_batch(encoder_inputs_len_for_att, self.args['beam_width'])
                            attention_scores = seq2seq.tile_batch(attention_scores, self.args['beam_width'])
                        
                        self.decoder_e_cell = ComUtils.modelInitHierarchicalAttentionDecoderCell(self.embedding_size, self.args['text_layers'], 'LSTM', self.args['text_dropout'], self.args['text_dropout'], decoder_contexts, encoder_inputs_len_for_att, 5, attention_scores, 'LUONG')

                        if not self.args['is_train'] and self.args['use_bs']:
                            initial_state = self.decoder_e_cell.zero_state(batch_size=self.args['batch_size']*self.args['beam_width'], dtype=tf.float32)
                        else:
                            initial_state = self.decoder_e_cell.zero_state(batch_size=self.args['batch_size'], dtype=tf.float32)
                        cat_state = tuple([encoder_states] + list(initial_state.cell_state)[:-1])
                        initial_state.clone(cell_state=cat_state)

                        self.output_projection_layer_e = ComUtils.Dense(self.output_size_e, use_bias=False)


                        decoder_inputs_embedded = ComUtils.modelGetWordEmbedding(self.output_word_embedding_matrix_e, self.decoder_outputs) #[batch_size, len, hidden_size]

                        
                        # run decoder
                        if args['is_train']:
                            self.train_outputs_e = ComUtils.modelRunDecoderForTrain(self.decoder_e_cell, decoder_inputs_embedded, self.decoder_targets_length, initial_state, self.output_projection_layer_e) # [batch_size, len, output_size]

                            self.loss_e = seq2seq.sequence_loss(logits=self.train_outputs_e, targets=self.decoder_targets, weights=self.decoder_targets_mask)
                        
                        if not self.args['is_train'] and self.args['use_bs']:
                            self.infer_outputs_e = ComUtils.modelRunDecoderForBSInfer(self.decoder_e_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_e, self.args['beam_width'], self.args['id_eos_E'], self.args['max_text_length'], initial_state, self.output_projection_layer_e)
                        else:
                            self.infer_outputs_e = ComUtils.modelRunDecoderForGreedyInfer(self.decoder_e_cell, tf.transpose(self.decoder_outputs, [1,0])[0], self.output_word_embedding_matrix_e, self.args['id_eos_E'], self.args['max_text_length'], initial_state, self.output_projection_layer_e)

                        self.model_e_variables = scope.trainable_variables()
                        print('CommentModelD: ', self.model_e_variables)

                        if args['is_train']:
                            optimizer = tf.train.AdamOptimizer
                            if self.args['text_op_E']=='GD':
                                optimizer = tf.train.GradientDescentOptimizer
                            self.train_op_e = ComUtils.updateBP(self.loss_e, [self.args['text_lr_E']], [self.model_e_variables ], optimizer, norm=self.args['text_norm_E'])