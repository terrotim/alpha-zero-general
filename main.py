"""
from Coach import Coach
from connect4.Connect4Game import Connect4Game
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict
"""

from Coach import Coach
from sotf.SotfGame import SotfGame
from sotf.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict


args = dotdict({
    'numIters': 100,
    'numEps': 2000,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 2000000,
    'numMCTSSims': 25,
    'arenaCompare': 100,
    'cpuct': 1,
    'dirichletAlpha': 1.5,
    
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = SotfGame()
    #g = Connect4Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
