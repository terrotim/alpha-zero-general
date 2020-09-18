import logging
import coloredlogs

"""
from Coach import Coach
from connect4.Connect4Game import Connect4Game as Game
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict
"""

from Coach import Coach
from sotf.SotfGame import SotfGame as Game
from sotf.tensorflow.NNet import NNetWrapper as nn
from utils import dotdict


log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100,
    'numEps': 500,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 2000000,
    'numMCTSSims': 25,
    'arenaCompare': 100,
    'cpuct': 1,
    'dirichletAlpha': 1.0,
    
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 50,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()
    #g = SotfGame()
    #g = Connect4Game()

    log.info('Loading %s...', nn.__name__)

    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
