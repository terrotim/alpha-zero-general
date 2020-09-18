import Arena
from MCTS import MCTS


from sotf.SotfGame import SotfGame, display
from sotf.SotfPlayers import *
from sotf.tensorflow.NNet import NNetWrapper as NNet


"""
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.tensorflow.NNet import NNetWrapper as NNet
"""

"""
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
"""


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


g = SotfGame()
#g = Connect4Game()

# all players
rp = RandomPlayer(g).play
rp2 = RandomPlayer(g).play
#hp = HumanPlayer(g).play
#hp2 = HumanPlayer(g).play
"""
mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play
"""

# nnet players

"""
n1 = NNet(g)
n1.load_checkpoint('./temp/','temp.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
"""
"""
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
"""

"""
if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.
"""

n2 = NNet(g)
n2.load_checkpoint('./temp/','best.pth.tar')
args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))


n3 = NNet(g)
n3.load_checkpoint('./pretrained_models/sotf/fixed','best.pth.tar')
args3 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts3 = MCTS(g, n3, args3)
n3p = lambda x: np.argmax(mcts3.getActionProb(x, temp=0))


#arena = Arena.Arena(rp, rp2, g, display=display)
arena = Arena.Arena(n2p, rp, g, display=display)
print(arena.playGames(100, verbose=True))

"""
arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)
print(arena.playGames(2, verbose=True))
"""