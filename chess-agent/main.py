from Coach import Coach
import sys
sys.path.append('alpha-zero-chess/')
from chessai.ChessGame import ChessGame as Game
from chessai.tensorflow.NNet import NNetWrapper as nn
from utils import *
import os
import json
import codecs
def save2json(d, pf):
    f = codecs.open(pf,'w','utf-8')
    f.write(json.dumps(d, ensure_ascii=False, indent=4))
    f.close()
def json2load(pf):
    f = codecs.open(pf,'r','utf-8')
    s = ''.join(f.readlines())
    d = json.loads(s)
    f.close()
    return d
import argparse

parser = argparse.ArgumentParser(
    description="Specify the config files!")

parser.add_argument(
    "-c",
    dest="config_name",
    type=str,
    default='default',
    help="The preset config file name (under configs/). ")

parser.add_argument(
    "-l",
    dest="load_id",
    type=int,
    default=0,
    help="load_id ")

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = codecs.open(filename, 'a', 'UTF-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__=="__main__":
    given_args = parser.parse_args()
    args = dotdict(json2load('configs/%s.json'%(given_args.config_name)))

    g = Game(buildDict=True)
    nnet = nn(g, args)
    

    if args.load_model:
        ln = len(args.load_folder_file)
        for i in range(ln//2):
            if given_args.load_id!=0:
                nnet.load_checkpoint(args.load_folder_file[i*2], args.load_folder_file[i*2+1].replace('??', str(given_args.load_id)))
            else:
                nnet.load_checkpoint(args.load_folder_file[i*2], args.load_folder_file[i*2+1])
    
    if args.save_model:
        if not os.path.exists(args.checkpoint):
            print("Checkpoint Directory does not exist! Making directory {}".format(args.checkpoint))
            os.mkdir(args.checkpoint)
        if not os.path.exists(os.path.join(args.checkpoint, 'log.txt')):
            f = codecs.open(os.path.join(args.checkpoint, 'log.txt'), 'w', 'UTF-8')
            f.close()
        save2json(args, os.path.join(args.checkpoint, 'config.json'))
    
        sys.stdout = Logger(os.path.join(args.checkpoint, 'log.txt'), sys.stdout)
        sys.stderr = Logger(os.path.join(args.checkpoint, 'log.txt'), sys.stderr)
    
    c = Coach(g, nnet, args)
    c.learn()