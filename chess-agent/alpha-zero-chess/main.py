from Coach import Coach
from chessai.ChessGame import ChessGame as Game
from chessai.tensorflow.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 10,
    'tempThreshold': 10,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 2000,
    'numMCTSSims': 50,
    'arenaCompare': 20,
    'cpuct': 2.0,
    'checkpoint': 'try/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 3,

})

if __name__=="__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
