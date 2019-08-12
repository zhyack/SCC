from collections import deque
import sys
sys.path.append('alpha-zero-chess/')
sys.path.append('tools/')
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle, randint
from parse_game import loadTrainExamples
import bleu
import meteor
import diversity
import chess
import random
import nltk


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        if self.args.use_pitting:
            self.pnet = self.nnet.__class__(self.game, nnet.args)  # the competitor network
        if self.args.use_self_play:
            self.selfplaynum = 1
            self.trainExampleSelfPlay = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
            self.mcts = MCTS(self.game, self.nnet, self.args)

        if self.args.comment_training:
            self.validExamples = []
        if self.args.comment_training or self.args.chess_training or self.args.use_self_play:
            self.trainExamples = []    # history of examples from 
        

    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            # print(board.unicode().replace(u'·', u'.'))
            # print(self.curPlayer)
            # print([m.uci() for m in board.pseudo_legal_moves])
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            valids = self.game.getValidMoves(board, 1)

            bs, ps = zip(*self.game.getSymmetries(canonicalBoard, pi))
            _, valids_sym = zip(*self.game.getSymmetries(canonicalBoard, valids))
            sym = zip(bs,ps,valids_sym)
            for b,p,valid in sym:
                trainExamples.append([b, self.curPlayer, p, valid])

            # action = np.random.choice(len(pi), p=pi)
            action = np.argmax(pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)


            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer)),x[3], None) for x in trainExamples]

    def getDataSetDistribution(self, type_str='chess'):
        query_str = 'train_%s_files'%(type_str)
        n = self.args['nn_args']['sample_batches'] * self.args['nn_args']['batch_size']
        datafile_indexes = list(range(len(self.args[query_str][1])//2))
        shuffle(datafile_indexes)
        ns = []
        train_data_files = []
        nns = min(self.args['sample_files'], len(datafile_indexes))
        for i in range(nns):
            k = datafile_indexes[i]
            ns.append(self.args[query_str][1][k*2+1])
            train_data_files.append(self.args[query_str][1][k*2])
        sum_ns = sum(ns)
        ns_left = n
        for i in range(nns):
            ns[i] = min(ns_left, int(ns[i]/sum_ns*n))
            ns_left -= ns[i]
        if ns_left>0:
            ns[randint(0, nns-1)] += ns_left
        return train_data_files, ns
    
    def learn_chess_iter(self):
        self.trainExamples = []

        train_chess_files, ns = self.getDataSetDistribution('chess')

        for j, file_name in enumerate(train_chess_files):
            trainExamples = loadTrainExamples(self.args['train_chess_files'][0]+file_name)
            shuffle(trainExamples)
            self.trainExamples.extend(trainExamples[:ns[j]])

        shuffle(self.trainExamples)

        self.nnet.train(self.trainExamples, transform=True)

        self.trainExamples = []
    
    def learn_self_play_iter(self):
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        eps_time = AverageMeter()
        bar = Bar('Self Play', max=self.selfplaynum)
        end = time.time()

        for eps in range(self.selfplaynum):
            iterationTrainExamples += self.executeEpisode()

            # bookkeeping + plot progress
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.selfplaynum, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()
        self.trainExampleSelfPlay.extend(iterationTrainExamples)
        shuffle(self.trainExampleSelfPlay)
        print('Got %d replays through self-play.'%(len(self.trainExampleSelfPlay)))
        self.nnet.train(self.trainExampleSelfPlay, transform=True)
    
    def learn_comment_iter(self):
        self.trainExamples = []
        self.validExamples = []

        if 'train_text_files' in self.args:

            train_text_files, ns = self.getDataSetDistribution('text')

            for j, file_name in enumerate(train_text_files):
                trainExamples = loadTrainExamples(self.args['train_text_files'][0]+file_name)
                shuffle(trainExamples)
                self.trainExamples.extend(trainExamples[:ns[j]])
        
        if 'valid_text_files' in self.args:
            for file_name in self.args['valid_text_files'][1]:
                validExamples = loadTrainExamples(self.args['valid_text_files'][0]+file_name)
                self.validExamples.extend(validExamples)
        
        # define train classification
        shuffle(self.trainExamples)
        n_e = len(self.trainExamples)
        if n_e:
            class_examples = [[], [], [], [], [], []]
            for i in range(n_e):
                e = self.trainExamples[i]
                if e==None:
                    continue
                class_examples[0].append(e)
                if e[-1]!=None:
                    e = list(e)
                    if isinstance(e[-1], tuple):
                        cs = set(e[-1][1])
                        e[-1]=e[-1][0]
                        for c in cs:
                            class_examples[c].append(e)
                        if 'test_with_no_ai' in self.nnet.args and self.nnet.args['test_with_no_ai'] and 1 not in cs:
                            class_examples[1].append(e)
                    else:
                        le = e[-1].split('@\t@')
                        assert(len(le)==2)
                        e[-1]=le[1]
                        cs = set([int(s) for s in le[0].split()])
                        for c in cs:
                            class_examples[c].append(e)
                        if 'test_with_no_ai' in self.nnet.args and self.nnet.args['test_with_no_ai'] and 1 not in cs:
                            class_examples[1].append(e)
                self.trainExamples[i] = None
            print('TrainExamples Distribution: ', [len(e) for e in class_examples])
            if self.args['comment_chess_training']:
                print('Training Chess: ')
                self.nnet.train(class_examples[0], transform=True, models=[0])
            for m in self.nnet.args['models']['comment']:
                i = ord(m)-ord('A')+1
                if len(class_examples[i]):
                    print('Training Class %s:\n'%(m))
                    self.nnet.train(class_examples[i], transform=True, models=[i])

        # define dev
        n_e = len(self.validExamples)
        if n_e:
            class_examples = [[], [], [], [], [], []]
            for i in range(n_e):
                e = self.validExamples[i]
                if e==None:
                    continue
                class_examples[0].append(e)
                if e[-1]!=None:
                    e = list(e)
                    if isinstance(e[-1], tuple):
                        cs = set(e[-1][1])
                        e[-1]=e[-1][0]
                        for c in cs:
                            class_examples[c].append(e)
                        if 'test_with_no_ai' in self.nnet.args and self.nnet.args['test_with_no_ai'] and 1 not in cs:
                            class_examples[1].append(e)
                    else:
                        le = e[-1].split('@\t@')
                        assert(len(le)==2)
                        e[-1]=le[1]
                        cs = set([int(s) for s in le[0].split()])
                        for c in cs:
                            class_examples[c].append(e)
                        if 'test_with_no_ai' in self.nnet.args and self.nnet.args['test_with_no_ai'] and 1 not in cs:
                            class_examples[1].append(e)
                self.validExamples[i] = None
            print('ValidExamples Distribution: ', [len(e) for e in class_examples])
            for m in self.nnet.args['models']['comment']:
                i = ord(m)-ord('A')+1
                if len(class_examples[i]):
                    print('Evaluating Class %s:'%(m))
                    n = len(class_examples[i])
                    bsize = self.nnet.args['batch_size']
                    predict_texts = []
                    gold_texts = []
                    for b in range((n+bsize-1)//bsize):
                        batch = class_examples[i][b*bsize:min((b+1)*bsize,n)]
                        n_batch = len(batch)
                        if n_batch<bsize:
                            batch += class_examples[i][:bsize-n_batch]
                        boards, pis, vs, valids, texts = list(zip(*batch))
                        rets = list(self.nnet.predict(boards, [pis, valids], models=[i], transform=True))
                        for tb in range(n_batch):
                            if boards[tb][-1]==-1 and len(rets[tb])>0:
                                rets[tb] = self.postProcess(rets[tb], player="black")
                            elif len(rets[tb])>0:
                                rets[tb] = self.postProcess(rets[tb], player="white")
                        predict_texts.extend(rets[:n_batch])
                        gold_texts.extend(texts[:n_batch])
                        # for k in range(n_batch):
                        #     if random.random()<0.0005 or len(self.trainExamples)==0:
                        #         print('Board: ', boards[k][-1])
                        #         print(chess.Board(boards[k][0]).unicode().replace(u'·', u'.'))
                        #         print('Move: ', self.game.action_list[pis[k][0]])
                        #         print('Expected: ', texts[k].strip())
                        #         print('Predicted: ', rets[k])
                    
                    result = bleu.corpus_bleu(predict_texts, [[t.strip()] for t in gold_texts])[0][0]
                    # refs = []
                    # hyps = []
                    # for p, t in zip(predict_texts, gold_texts):
                    #     refs.append([t.split()])
                    #     hyps.append(p.split())
                    # result = nltk.translate.bleu_score.corpus_bleu(refs, hyps, auto_reweigh=True)
                    print('BLEU-4 for Class %s: %.2f'%(m, result*100))
                    result = bleu.corpus_bleu(predict_texts, [[t.strip()] for t in gold_texts], max_n=2)[0][0]
                    print('BLEU-2 for Class %s: %.2f'%(m, result*100))
                    save2text(gold_texts, self.args.checkpoint+'_gold%d-%d.txt'%(i, self.iter))
                    save2text(predict_texts, self.args.checkpoint+'_predicted%d-%d.txt'%(i, self.iter))
                    print('METEOR for Class %s: '%(m), meteor.evaluate(self.args.checkpoint+'_predicted%d-%d.txt'%(i, self.iter), self.args.checkpoint+'_gold%d-%d.txt'%(i, self.iter)))
                    print('Dist-2 for Class %s: '%(m), diversity.corpus_diversity(predict_texts))
        self.trainExamples = []
        self.validExamples = []



    def learn(self):
        for i in range(1, self.args.numIters+1):
            self.iter = i
            # bookkeeping
            print('------ITER ' + str(i) + '------')

            if self.args.use_pitting:
                # training new network, keeping a copy of the old one
                if os.path.exists(os.path.join(self.args.checkpoint,'best.pth.tar.index')):
                    self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                else:
                    self.pnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                pmcts_game = self.game.__class__()
                pmcts_game.getInitBoard()
                pmcts = MCTS(pmcts_game, self.pnet, self.args)
            

            if self.args.comment_training:
                self.learn_comment_iter()
            
            if not self.args.nn_args['is_train'] and not self.args.use_pitting:
                break

            if self.args.chess_training:
                self.learn_chess_iter()
           
            if self.args.use_self_play:
                self.learn_self_play_iter()
                self.mcts = MCTS(self.game, self.nnet, self.args)

            if self.args.save_model:
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                if self.args['comment_training']:
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='new.pth.tar')
            
            if self.args.use_pitting:
                nmcts_game = self.game.__class__()
                nmcts_game.getInitBoard()
                nmcts = MCTS(nmcts_game, self.nnet, self.args)

                print('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                            lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

                print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            
                if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                    print('REJECTING NEW MODEL')
                else:
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                    if self.args.use_self_play: 
                        self.trainExampleSelfPlay = []
                        self.selfplaynum+=1
                nmcts = None
                pmcts = None
            

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'
        
    def postProcess(self, s, player="black"):
        opposite_player = {'black':'white', 'white':'black'}
        def getOppositePlayer(player):
            return opposite_player[player]
        s = s.replace("...","")
        s = s.strip()
        s = s.replace("'m","am")
        words = s.split()
        if len(words)<3:
            return s
        if words[0]=="so" or words[0]=="and":
            words = words[1:]
        if words[0].lower()=="i":
            if words[1].lower()=="think":
                #words[1] = "It"
                words = words[2:]
                #pass
            else:
                words[0] = str.upper(player[0]) + player[1:]
                opposite_player = getOppositePlayer(player)
                for j in range(len(words)):
                    if words[j]=="his":
                        words[j] = opposite_player #+ "'s"
                for j in range(len(words)):
                    if words[j]=="my":
                        words[j] = "his"
            s = " ".join(words)
        return s

import json
import codecs
def save2json(d, pf):
    f = codecs.open(pf,'w','utf-8')
    f.write(json.dumps(d, ensure_ascii=False, indent=4))
    f.close()
def save2text(l, pf):
    f = codecs.open(pf,'w','utf-8')
    l = [line.replace('\n', '').replace('\r', '') for line in l]
    f.write('\n'.join(l))
    f.close()
def json2load(pf):
    f = codecs.open(pf,'r','utf-8')
    s = ''.join(f.readlines())
    d = json.loads(s)
    f.close()
    return d

