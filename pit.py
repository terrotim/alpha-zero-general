import Arena
from MCTS import MCTS
from sotf.SotfGame import SotfGame, display
from sotf.SotfPlayers import *
from sotf.tensorflow.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = SotfGame()

# all players
rp = RandomPlayer(g).play
rp2 = RandomPlayer(g).play
hp = HumanPlayer(g).play
hp2 = HumanPlayer(g).play
# nnet players


n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('./temp/','temp.pth.tar')
args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))


#arena = Arena.Arena(rp, rp2, g, display=display)
arena = Arena.Arena(n1p, n2p, g, display=display)
print(arena.playGames(100, verbose=True))
