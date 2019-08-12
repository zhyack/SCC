import os
import shutil
import time
import random
import numpy as np
import math
import sys


sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import tensorflow as tf
from .ChessNNet import ChessNNet as onnet
from ..ChessLogic import getBoardFeatures, getActionFeatures, extend2Vector, awsize
from . import data_utils

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, spe_args=None):
        if spe_args and 'nn_args' in spe_args:
            args.update(spe_args['nn_args'])
        self.args = args
        self.text_list = {}

        if 'text_dict_A' in self.args and 'A' in self.args['models']['comment']:
            self.text_dict_a, self.text_list_a = data_utils.loadDict(self.args['text_dict_A'])
            self.args['output_size_A'] = len(self.text_list_a)
            self.args['id_eos_A'] = self.text_dict_a['<EOS>']
            self.text_list[1] = self.text_list_a

        if 'text_dict_B' in self.args and 'B' in self.args['models']['comment']:
            self.text_dict_b, self.text_list_b = data_utils.loadDict(self.args['text_dict_B'])
            self.args['output_size_B'] = len(self.text_list_b)
            self.args['id_eos_B'] = self.text_dict_b['<EOS>']
            self.text_list[2] = self.text_list_b

        if 'text_dict_C' in self.args and 'C' in self.args['models']['comment']:
            self.text_dict_c, self.text_list_c = data_utils.loadDict(self.args['text_dict_C'])
            self.args['output_size_C'] = len(self.text_list_c)
            self.args['id_eos_C'] = self.text_dict_c['<EOS>']
            self.text_list[3] = self.text_list_c

        if 'text_dict_D' in self.args and 'D' in self.args['models']['comment']:
            self.text_dict_d, self.text_list_d = data_utils.loadDict(self.args['text_dict_D'])
            self.args['output_size_D'] = len(self.text_list_d)
            self.args['id_eos_D'] = self.text_dict_d['<EOS>']
            self.text_list[4] = self.text_list_d

        if 'text_dict_E' in self.args and 'E' in self.args['models']['comment']:
            self.text_dict_e, self.text_list_e = data_utils.loadDict(self.args['text_dict_E'])
            self.args['output_size_E'] = len(self.text_list_e)
            self.args['id_eos_E'] = self.text_dict_e['<EOS>']
            self.text_list[5] = self.text_list_e

        self.args['input_size'] = awsize

        self.game = game
        self.game.getInitBoard()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.train_functions = [self.train_chess_ai, self.train_comment_a, self.train_comment_b, self.train_comment_c, self.train_comment_d, self.train_comment_e]
        self.predict_functions = [self.predict_chess_ai, self.predict_comment_a, self.predict_comment_b, self.predict_comment_c, self.predict_comment_d, self.predict_comment_e]
        
        self.nnet = onnet(game, args)
        tfconfig = tf.ConfigProto(gpu_options=tf.GPUOptions())
        tfconfig.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.nnet.graph, config=tfconfig)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train_chess_ai(self, examples, transform):
        boards, pis, vs, valids, texts = list(zip(*examples))
        if transform:
            boards = [getBoardFeatures(b) for b in boards]
            pis = [extend2Vector(p, self.action_size) for p in pis]
            valids = [extend2Vector(v, self.action_size) for v in valids]
        # predict and compute gradient and do SGD step
        input_dict = {self.nnet.input_boards: boards, self.nnet.valid_mask: valids, self.nnet.target_pis: pis, self.nnet.target_vs: vs}

        # record loss
        [_, pi_loss, v_loss] = self.sess.run([self.nnet.train_op_chess, self.nnet.chess_loss_pi, self.nnet.chess_loss_v], feed_dict=input_dict)
        
        return (pi_loss, v_loss)
    
    def predict_chess_ai(self, board, others):
        if isinstance(board, list):
            board = getBoardFeatures(board)
        board = board[np.newaxis, :, :, :]
        valids = others[np.newaxis, :]
        self.nnet.args['dropout']=0
        [prob, v] = self.sess.run([self.nnet.action_prob, self.nnet.state_value], feed_dict={self.nnet.input_boards: board, self.nnet.valid_mask: valids})
        return (prob[0], v[0])

    def train_comment_a(self, examples, transform):
        boards, moves, vs, valids, texts = list(zip(*examples))
        action_descs = [getActionFeatures(self.game, board, m[0]) for board, m in zip(boards, moves)]
        decoder_outputs, decoder_targets, decoder_targets_length, decoder_targets_mask = data_utils.getDecoderData(texts, self.text_dict_a, self.args['max_text_length'])
        boards = [getBoardFeatures(b) for b in boards]

        input_dict = {self.nnet.input_boards: boards, self.nnet.input_action_descs: action_descs, self.nnet.decoder_outputs:decoder_outputs, self.nnet.decoder_targets:decoder_targets, self.nnet.decoder_targets_length:decoder_targets_length, self.nnet.decoder_targets_mask:decoder_targets_mask}

        _, loss = self.sess.run([self.nnet.train_op_a, self.nnet.loss_a], feed_dict=input_dict)

        return (loss)
    
    def predict_comment_a(self, board, others):
        boards = board
        moves, valids = others
        if not isinstance(boards, tuple):
            boards = [boards]
            moves = [moves]
            valids = [valids]
        action_descs = [getActionFeatures(self.game, b, m[0]) for b,m in zip(boards, moves)]
        boards = [getBoardFeatures(b) for b in boards]
        decoder_outputs = np.array([ [self.text_dict_a['<BOS>']] for _ in range(self.nnet.args['batch_size'])], np.int32)

        input_dict = {self.nnet.input_boards: boards, self.nnet.input_action_descs: action_descs, self.nnet.decoder_outputs:decoder_outputs}

        [infer_outputs] = self.sess.run([self.nnet.infer_outputs_a], feed_dict=input_dict)

        return (infer_outputs)

      
    def train_comment_b(self, examples, transform):
        boards, moves, vs, valids, texts = list(zip(*examples))
        action_descs = [getActionFeatures(self.game, board, m[0]) for board, m in zip(boards, moves)]
        decoder_outputs, decoder_targets, decoder_targets_length, decoder_targets_mask = data_utils.getDecoderData(texts, self.text_dict_b, self.args['max_text_length'])
        next_boards = []
        for b, m in zip(boards, moves):
            nb, npl = self.game.getNextState(b, b[-1], m[0])
            next_boards.append(getBoardFeatures(self.game.getCanonicalForm(nb, npl)))
        
        boards = [getBoardFeatures(b) for b in boards] + next_boards

        input_dict = {self.nnet.input_boards: boards, self.nnet.input_action_descs: action_descs, self.nnet.decoder_outputs:decoder_outputs, self.nnet.decoder_targets:decoder_targets, self.nnet.decoder_targets_length:decoder_targets_length, self.nnet.decoder_targets_mask:decoder_targets_mask}

        _, loss = self.sess.run([self.nnet.train_op_b, self.nnet.loss_b], feed_dict=input_dict)

        return (loss)
    
    def predict_comment_b(self, board, others):
        boards = board
        moves, valids = others
        if not isinstance(boards, tuple):
            boards = [boards]
            moves = [moves]
            valids = [valids]
        action_descs = [getActionFeatures(self.game, b, m[0]) for b,m in zip(boards, moves)]
        
        next_boards = []
        for b, m in zip(boards, moves):
            nb, npl = self.game.getNextState(b, b[-1], m[0])
            next_boards.append(getBoardFeatures(self.game.getCanonicalForm(nb, npl)))
        boards = [getBoardFeatures(b) for b in boards] + next_boards

        valids = [extend2Vector(v, self.action_size) for v in valids]
        decoder_outputs = np.array([ [self.text_dict_b['<BOS>']] for _ in range(self.nnet.args['batch_size'])], np.int32)

        input_dict = {self.nnet.input_boards: boards, self.nnet.input_action_descs: action_descs, self.nnet.decoder_outputs:decoder_outputs}

        [infer_outputs] = self.sess.run([self.nnet.infer_outputs_b], feed_dict=input_dict)

        return (infer_outputs)

    def train_comment_c(self, examples, transform):
        boards, moves, vs, valids, texts = list(zip(*examples))

        feat_boards = [getBoardFeatures(b) for b in boards]
        feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
        [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
        best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)


        action_descs = [getActionFeatures(self.game, board, m[0]) for board, m in zip(boards, moves)]
        action_descs_best =  [getActionFeatures(self.game, board, m) for board, m in zip(boards, best_moves)]
        action_descs += action_descs_best

        boards = [getBoardFeatures(b) for b in boards] + feat_boards
        
        decoder_outputs, decoder_targets, decoder_targets_length, decoder_targets_mask = data_utils.getDecoderData(texts, self.text_dict_c, self.args['max_text_length'])

        input_dict = {self.nnet.input_boards: boards, self.nnet.input_action_descs: action_descs, self.nnet.decoder_outputs:decoder_outputs, self.nnet.decoder_targets:decoder_targets, self.nnet.decoder_targets_length:decoder_targets_length, self.nnet.decoder_targets_mask:decoder_targets_mask}

        _, loss = self.sess.run([self.nnet.train_op_c, self.nnet.loss_c], feed_dict=input_dict)

        return (loss)

    def predict_comment_c(self, board, others):
        boards = board
        moves, valids = others
        if not isinstance(boards, tuple):
            boards = [boards]
            moves = [moves]
            valids = [valids]
        
        feat_boards = [getBoardFeatures(b) for b in boards]
        feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
        [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
        best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)
        
        action_descs = [getActionFeatures(self.game, board, m[0]) for board, m in zip(boards, moves)]
        action_descs_best =  [getActionFeatures(self.game, board, m) for board, m in zip(boards, best_moves)]
        action_descs += action_descs_best

        boards = [getBoardFeatures(b) for b in boards] + feat_boards

        decoder_outputs = np.array([ [self.text_dict_c['<BOS>']] for _ in range(self.nnet.args['batch_size'])], np.int32)

        input_dict = {self.nnet.input_boards: boards, self.nnet.input_action_descs: action_descs, self.nnet.decoder_outputs:decoder_outputs}

        [infer_outputs] = self.sess.run([self.nnet.infer_outputs_c], feed_dict=input_dict)

        return (infer_outputs)
    
    def train_comment_d(self, examples, transform):
        boards, moves, vs, valids, texts = list(zip(*examples))

        batch_size = len(boards)
        boards_all = []
        action_descs_all = []
        current_player = 1
        boards = list(boards)
        moves = list(moves)
        valids = list(valids)
        is_end = [False]*batch_size

        for _ in range(6):
            board_display = self.game.restoreOriginBoard(boards[0])
            if current_player == 1:
                for i, b, m in zip(range(batch_size), boards, moves):
                    if is_end[i]:
                        boards_all.append(boards_all[-batch_size])
                        action_descs_all.append(action_descs_all[-batch_size])
                    else:
                        boards_all.append(getBoardFeatures(b))
                        action_descs_all.append(getActionFeatures(self.game, b, m[0]))
            if len(boards_all)==batch_size*3:
                    break
            for i in range(batch_size):
                try:
                    sub_player = boards[i][-1]
                    boards[i], _ = self.game.getNextState(boards[i], sub_player, moves[i][0])
                    valids[i] = self.game.getValidMoves(boards[i], -sub_player, mode='list')
                    boards[i] = self.game.getCanonicalForm(boards[i], -sub_player)
                except Exception:
                    is_end[i]=True
            current_player = -current_player

            feat_boards = [getBoardFeatures(b) for b in boards]
            feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
            [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
            best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)
            for i, m in enumerate(best_moves):
                moves[i] = [m]
                if m not in valids[i]:
                    is_end[i] = True
        
        decoder_outputs, decoder_targets, decoder_targets_length, decoder_targets_mask = data_utils.getDecoderData(texts, self.text_dict_d, self.args['max_text_length'])

        input_dict = {self.nnet.input_boards: boards_all, self.nnet.input_action_descs: action_descs_all, self.nnet.decoder_outputs:decoder_outputs, self.nnet.decoder_targets:decoder_targets, self.nnet.decoder_targets_length:decoder_targets_length, self.nnet.decoder_targets_mask:decoder_targets_mask}

        _, loss = self.sess.run([self.nnet.train_op_d, self.nnet.loss_d], feed_dict=input_dict)

        return (loss)

    def predict_comment_d(self, board, others):
        boards = board
        moves, valids = others
        if not isinstance(boards, tuple):
            boards = [boards]
            moves = [moves]
            valids = [valids]
        
        batch_size = len(boards)
        boards_all = []
        action_descs_all = []
        current_player = 1
        boards = list(boards)
        moves = list(moves)
        valids = list(valids)
        is_end = [False]*batch_size

        for _ in range(6):
            board_display = self.game.restoreOriginBoard(boards[0])
            if current_player == 1:
                for i, b, m in zip(range(batch_size), boards, moves):
                    if is_end[i]:
                        boards_all.append(boards_all[-batch_size])
                        action_descs_all.append(action_descs_all[-batch_size])
                    else:
                        boards_all.append(getBoardFeatures(b))
                        action_descs_all.append(getActionFeatures(self.game, b, m[0]))
            if len(boards_all)==batch_size*3:
                    break
            for i in range(batch_size):
                try:
                    sub_player = boards[i][-1]
                    boards[i], _ = self.game.getNextState(boards[i], sub_player, moves[i][0])
                    valids[i] = self.game.getValidMoves(boards[i], -sub_player, mode='list')
                    boards[i] = self.game.getCanonicalForm(boards[i], -sub_player)
                except Exception:
                    is_end[i]=True
            current_player = -current_player

            feat_boards = [getBoardFeatures(b) for b in boards]
            feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
            [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
            best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)
            for i, m in enumerate(best_moves):
                moves[i] = [m]
                if m not in valids[i]:
                    is_end[i] = True

        decoder_outputs = np.array([ [self.text_dict_d['<BOS>']] for _ in range(self.nnet.args['batch_size'])], np.int32)

        input_dict = {self.nnet.input_boards: boards_all, self.nnet.input_action_descs: action_descs_all, self.nnet.decoder_outputs:decoder_outputs}

        [infer_outputs] = self.sess.run([self.nnet.infer_outputs_d], feed_dict=input_dict)

        return (infer_outputs)

    def train_comment_e(self, examples, transform):
        boards, moves, vs, valids, texts = list(zip(*examples))

        batch_size = len(boards)
        boards_all = []
        action_descs_all = []
        current_player = 1
        boards = list(boards)
        moves = list(moves)
        valids = list(valids)
        is_end = [False]*batch_size

        for _ in range(10):
            board_display = self.game.restoreOriginBoard(boards[0])
            if current_player == 1:
                for i, b, m in zip(range(batch_size), boards, moves):
                    if is_end[i]:
                        boards_all.append(boards_all[-batch_size])
                        action_descs_all.append(action_descs_all[-batch_size])
                    else:
                        boards_all.append(getBoardFeatures(b))
                        action_descs_all.append(getActionFeatures(self.game, b, m[0]))
            if len(boards_all)==batch_size*5:
                    break
            for i in range(batch_size):
                try:
                    sub_player = boards[i][-1]
                    boards[i], _ = self.game.getNextState(boards[i], sub_player, moves[i][0])
                    valids[i] = self.game.getValidMoves(boards[i], -sub_player, mode='list')
                    boards[i] = self.game.getCanonicalForm(boards[i], -sub_player)
                except Exception:
                    is_end[i]=True
            current_player = -current_player

            feat_boards = [getBoardFeatures(b) for b in boards]
            feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
            [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
            best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)
            for i, m in enumerate(best_moves):
                moves[i] = [m]
                if m not in valids[i]:
                    is_end[i] = True
        
        decoder_outputs, decoder_targets, decoder_targets_length, decoder_targets_mask = data_utils.getDecoderData(texts, self.text_dict_e, self.args['max_text_length'])

        input_dict = {self.nnet.input_boards: boards_all, self.nnet.input_action_descs: action_descs_all, self.nnet.decoder_outputs:decoder_outputs, self.nnet.decoder_targets:decoder_targets, self.nnet.decoder_targets_length:decoder_targets_length, self.nnet.decoder_targets_mask:decoder_targets_mask}

        _, loss = self.sess.run([self.nnet.train_op_e, self.nnet.loss_e], feed_dict=input_dict)

        return (loss)

    def predict_comment_e(self, board, others):
        boards = board
        moves, valids = others
        if not isinstance(boards, tuple):
            boards = [boards]
            moves = [moves]
            valids = [valids]
        
        batch_size = len(boards)
        boards_all = []
        action_descs_all = []
        current_player = 1
        boards = list(boards)
        moves = list(moves)
        valids = list(valids)
        is_end = [False]*batch_size

        for _ in range(10):
            board_display = self.game.restoreOriginBoard(boards[0])
            if current_player == 1:
                for i, b, m in zip(range(batch_size), boards, moves):
                    if is_end[i]:
                        boards_all.append(boards_all[-batch_size])
                        action_descs_all.append(action_descs_all[-batch_size])
                    else:
                        boards_all.append(getBoardFeatures(b))
                        action_descs_all.append(getActionFeatures(self.game, b, m[0]))
            if len(boards_all)==batch_size*5:
                    break
            for i in range(batch_size):
                try:
                    sub_player = boards[i][-1]
                    boards[i], _ = self.game.getNextState(boards[i], sub_player, moves[i][0])
                    valids[i] = self.game.getValidMoves(boards[i], -sub_player, mode='list')
                    boards[i] = self.game.getCanonicalForm(boards[i], -sub_player)
                except Exception:
                    is_end[i]=True
            current_player = -current_player

            feat_boards = [getBoardFeatures(b) for b in boards]
            feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
            [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
            best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)
            for i, m in enumerate(best_moves):
                moves[i] = [m]
                if m not in valids[i]:
                    is_end[i] = True

        decoder_outputs = np.array([ [self.text_dict_e['<BOS>']] for _ in range(self.nnet.args['batch_size'])], np.int32)

        input_dict = {self.nnet.input_boards: boards_all, self.nnet.input_action_descs: action_descs_all, self.nnet.decoder_outputs:decoder_outputs}

        [infer_outputs] = self.sess.run([self.nnet.infer_outputs_e], feed_dict=input_dict)

        return (infer_outputs)


# # Older version of model e.
    # def train_comment_e(self, examples, transform):
    #     boards, moves, vs, valids, texts = list(zip(*examples))

    #     batch_size = len(boards)
    #     boards_all = []
    #     current_player = 1
    #     boards = list(boards)
    #     moves = list(moves)
    #     valids = list(valids)
    #     is_end = [False]*batch_size

    #     for _ in range(10):
    #         board_display = self.game.restoreOriginBoard(boards[0])
    #         if current_player == 1:
    #             for i, b, m in zip(range(batch_size), boards, moves):
    #                 if is_end[i]:
    #                     boards_all.append(boards_all[-batch_size])
    #                 else:
    #                     boards_all.append(getBoardFeatures(b))
    #         if len(boards_all)==batch_size*5:
    #                 break
    #         for i in range(batch_size):
    #             try:
    #                 sub_player = boards[i][-1]
    #                 boards[i], _ = self.game.getNextState(boards[i], sub_player, moves[i][0])
    #                 valids[i] = self.game.getValidMoves(boards[i], -sub_player, mode='list')
    #                 boards[i] = self.game.getCanonicalForm(boards[i], -sub_player)
    #             except Exception:
    #                 is_end[i]=True
    #         current_player = -current_player

    #         feat_boards = [getBoardFeatures(b) for b in boards]
    #         feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
    #         [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
    #         best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)
    #         for i, m in enumerate(best_moves):
    #             moves[i] = [m]
    #             if m not in valids[i]:
    #                 is_end[i] = True
        
    #     decoder_outputs, decoder_targets, decoder_targets_length, decoder_targets_mask = data_utils.getDecoderData(texts, self.text_dict_e, self.args['max_text_length'])

    #     input_dict = {self.nnet.input_boards: boards_all, self.nnet.decoder_outputs:decoder_outputs, self.nnet.decoder_targets:decoder_targets, self.nnet.decoder_targets_length:decoder_targets_length, self.nnet.decoder_targets_mask:decoder_targets_mask}

    #     _, loss = self.sess.run([self.nnet.train_op_e, self.nnet.loss_e], feed_dict=input_dict)

    #     return (loss)
    # def predict_comment_e(self, board, others):
        boards = board
        moves, valids = others
        if not isinstance(boards, tuple):
            boards = [boards]
            moves = [moves]
            valids = [valids]
        
        batch_size = len(boards)
        boards_all = []
        current_player = 1
        boards = list(boards)
        moves = list(moves)
        valids = list(valids)
        is_end = [False]*batch_size

        for _ in range(10):
            board_display = self.game.restoreOriginBoard(boards[0])
            if current_player == 1:
                for i, b, m in zip(range(batch_size), boards, moves):
                    if is_end[i]:
                        boards_all.append(boards_all[-batch_size])
                    else:
                        boards_all.append(getBoardFeatures(b))
            if len(boards_all)==batch_size*5:
                    break
            for i in range(batch_size):
                try:
                    sub_player = boards[i][-1]
                    boards[i], _ = self.game.getNextState(boards[i], sub_player, moves[i][0])
                    valids[i] = self.game.getValidMoves(boards[i], -sub_player, mode='list')
                    boards[i] = self.game.getCanonicalForm(boards[i], -sub_player)
                except Exception:
                    is_end[i]=True
            current_player = -current_player

            feat_boards = [getBoardFeatures(b) for b in boards]
            feat_valids = np.array([extend2Vector(v, self.action_size) for v in valids])
            [prob] = self.sess.run([self.nnet.action_prob], feed_dict={self.nnet.input_boards: feat_boards, self.nnet.valid_mask: feat_valids})
            best_moves = np.argmax((prob+(np.random.randn(self.action_size)+1.0)/20000.0)*feat_valids, axis=1)
            for i, m in enumerate(best_moves):
                moves[i] = [m]
                if m not in valids[i]:
                    is_end[i] = True

        decoder_outputs = np.array([ [self.text_dict_e['<BOS>']] for _ in range(self.nnet.args['batch_size'])], np.int32)

        input_dict = {self.nnet.input_boards: boards_all, self.nnet.decoder_outputs:decoder_outputs}

        [infer_outputs] = self.sess.run([self.nnet.infer_outputs_e], feed_dict=input_dict)

        return (infer_outputs)

    def train(self, examples, transform=False, models=[0]):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            batch_time = AverageMeter()
            losses = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                sample_examples = [examples[i] for i in sample_ids]
                for m in models:
                    loss = self.train_functions[m](sample_examples, transform)
                    if m == 0:
                        pi_losses.update(loss[0], len(sample_examples))
                        v_losses.update(loss[1], len(sample_examples))
                    else:
                        losses.update(loss, len(sample_examples))
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1
                bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.3f} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            total=bar.elapsed_td,
                            loss=losses.avg,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()

    def predict(self, board, valids=None, models=[0], transform=False):
        ret = []
        for m in models:
            results = self.predict_functions[m](board, valids)
            if transform:
                if isinstance(results, tuple):
                    results = results[0]
                n_res = len(results)
                new_results = []
                for k in range(n_res):
                    res = ''
                    for ind in results[k]:
                        if self.text_list[m][ind]=='<EOS>':
                            break
                        res+=self.text_list[m][ind] +' '
                    new_results.append(res)
                results = new_results
            ret.extend(list(results))

        return tuple(ret)


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'), max_to_keep=100)
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'.meta'):
            print("No model in path {}".format(filepath))
            return
        with self.nnet.graph.as_default():
            print("Find model in path {}".format(filepath))
            print('Restoring Models...')

            self.chess_ai_saver = tf.train.Saver(self.nnet.chess_ai_variables)
            try:
                self.chess_ai_saver.restore(self.sess, filepath)
                print('Chess AI Model Loaded!')
            except Exception as e:
                print(e)
                print('Chess AI Model Not Found...')

            if 'comment' in self.nnet.args['models']:
                self.text_shared_saver = tf.train.Saver(self.nnet.action_word_embedding_variables + self.nnet.action_encoder_variables)
                try:
                    self.text_shared_saver.restore(self.sess, filepath)
                    print('Action Embeddings & Encoders Loaded!')
                except Exception:
                    print('Action Embeddings & Encoders Not Found...')
                self.text_shared_saver = tf.train.Saver(self.nnet.state_diff_variables)
                try:
                    self.text_shared_saver.restore(self.sess, filepath)
                    print('State Diff Encoder Loaded!')
                except Exception:
                    print('State Diff Encoder Not Found...')
                self.text_shared_saver = tf.train.Saver( self.nnet.state_value_variables)
                try:
                    self.text_shared_saver.restore(self.sess, filepath)
                    print('State Value Encoder Loaded!')
                except Exception:
                    print('State Value Encoder Not Found...')
                # self.text_shared_saver = tf.train.Saver(self.nnet.action_word_embedding_variables + self.nnet.action_encoder_variables + self.nnet.state_diff_variables + self.nnet.state_value_variables)
                # try:
                #     self.text_shared_saver.restore(self.sess, filepath)
                #     print('Shared Embeddings & Encoders Loaded!')
                # except Exception:
                #     print('Shared Embeddings & Encoders Not Found...')

                if 'A' in self.nnet.args['models']['comment']:
                    self.model_a_saver = tf.train.Saver(self.nnet.model_a_variables)
                    try:
                        self.model_a_saver.restore(self.sess, filepath)
                        print('Comment Model A Loaded!')
                    except Exception:
                        print('Comment Model A Not Found...')
                if 'B' in self.nnet.args['models']['comment']:
                    self.model_b_saver = tf.train.Saver(self.nnet.model_b_variables)
                    try:
                        self.model_b_saver.restore(self.sess, filepath)
                        print('Comment Model B Loaded!')
                    except Exception:
                        print('Comment Model B Not Found...')
                if 'C' in self.nnet.args['models']['comment']:
                    self.model_c_saver = tf.train.Saver(self.nnet.model_c_variables)
                    try:
                        self.model_c_saver.restore(self.sess, filepath)
                        print('Comment Model C Loaded!')
                    except Exception:
                        print('Comment Model C Not Found...')
                if 'D' in self.nnet.args['models']['comment']:
                    self.model_d_saver = tf.train.Saver(self.nnet.model_d_variables)
                    try:
                        self.model_d_saver.restore(self.sess, filepath)
                        print('Comment Model D Loaded!')
                    except Exception:
                        print('Comment Model D Not Found...')
                if 'E' in self.nnet.args['models']['comment']:
                    self.model_e_saver = tf.train.Saver(self.nnet.model_e_variables)
                    try:
                        self.model_e_saver.restore(self.sess, filepath)
                        print('Comment Model E Loaded!')
                    except Exception:
                        print('Comment Model E Not Found...')
                print('Done')

            
