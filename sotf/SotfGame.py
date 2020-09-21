import sys
import numpy as np
import colorama
from itertools import permutations,product

sys.path.append('..')
from Game import Game
from .SotfLogic import Board
from termcolor import colored, cprint

colorama.init()

class SotfGame(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=4, width=12, tiles=None):
        self.height = height
        self.width = width
        self.board = Board(self.height,self.width)

    def getInitBoard(self):
        # main 4x12 board, 2 pboards size 51 (both with a 3 reserve board and a 48 tile board), 1 first_claim, and 1 action_num
        np.random.shuffle(self.board.layout[:self.height*self.width])
        return self.board.layout

    def getBoardSize(self):
        #48 + 51 + 51  + 1 + 1
        return self.board.layout.size,1

    def getActionSize(self):
        # user can use three spirits in any way on the 4x12 board, and they can pass
        return (self.board.height*self.board.width * 3) + 1

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = Board(self.height,self.width)
        b.layout = np.copy(board)
        
        #move = np.unravel(action,[self.height,self.width])
        #b.layout = b.execute_move(move,player)
        if action < self.getActionSize()-1:
            b.layout = b.execute_move(action,player)
            #print('action',action,player)
            #print('b.layout',b.layout)
        if b.layout[-1] < 3:
            b.layout[-1] += 1
            return b.layout, player
        else:
            b.layout[-1] = 1
            return b.layout, -player
        
        

    def getValidMoves(self, board, player):
        b = Board(self.height,self.width)
        b.layout = np.copy(board)
        
        validMoves = [0]*self.getActionSize()
        
        for i in range(self.getActionSize() - 1):
            validMoves[i] = b.is_legal_move(i,player)
            #spirit_used,tile_row,tile_column = np.unravel_index(i,[self.board.height,self.board.width,3])
            #move = (spirit_used,tile_row,tile_column
            
            #validMoves[i] = b.is_legal_move(move,player)
        
        #if not valid moves, user can pass
        if not 1 in validMoves:
            validMoves[-1] = 1
        
        #if its the second or third move of a user's turn, they may pass
        if b.layout[-1] > 1:
            validMoves[-1] = 1
            
            
        #print('validmoves', validMoves)
        return validMoves

    def getGameEnded(self, board, player):
        b = Board(self.height,self.width)
        b.layout = np.copy(board)
        
        allTilesTaken = b.all_tiles_taken()
        
        if allTilesTaken:
            p1score, p2score = b.get_scores(player)
            if p1score>p2score:
                return 1
            if p2score>p1score:
                return -1
            return 1e-4
        else:
            return 0
        
    def getCanonicalForm(self, board, player):
        # main 4x12 board, 2 pboards size 51 (both with a 3 reserve board and a 48 tile board), and 1 action_num
        b = np.copy(board)
        tiles = b[:self.height*self.width]
        p_tiles = b[self.height*self.width:self.height*self.width*3]
        p_spirits = b[self.height*self.width*3:-2]
        p_tiles = p_tiles.reshape(2,-1)
        p_spirits = p_spirits.reshape(2,-1)
        
        first_claim = b[-2]
        action_num = b[-1]

        return np.concatenate((tiles,p_tiles[::player].flatten(),p_spirits[::player].flatten(),[first_claim],[action_num]))

    def getSymmetries(self, board, pi):
        
        b = np.copy(board)
        tiles = b[:self.height*self.width]
        p_tiles = b[self.height*self.width:self.height*self.width*3]
        p_spirits = b[self.height*self.width*3:-2]
        first_claim = b[-2]
        action_num = b[-1]
        
        tile_board = tiles.reshape(self.height,self.width)
        pi_board = np.asarray(pi[:-1]).reshape(self.height,self.width,3)
        
        #t_perms = list(permutations(tile_board))
        #p_perms = list(permutations(pi_board))
        t_perms = [tile_board]
        p_perms = [pi_board]
        
        assert len(t_perms) == len(p_perms)

        syms = []
        combs = list(product(range(-1,2,2),repeat=4))

        for p in range(len(t_perms)):
            t_perm = t_perms[p]
            p_perm = p_perms[p]
            for c in combs:
                new_t = np.concatenate([t_perm[i][::c[i]] for i in range(len(c))])
                new_p = np.concatenate([p_perm[i][::c[i]] for i in range(len(c))])
                sym = (np.concatenate((new_t,p_tiles,p_spirits,[first_claim],[action_num])).reshape(self.getBoardSize()),np.concatenate((np.concatenate(new_p),[pi[-1]])))
                syms.append(sym)
        
        return syms
        
        #return [(board.reshape(self.getBoardSize()),pi)]
        
    def stringRepresentation(self, board):
        return board.tostring()


def display(board):
    height = 4
    width = 12
    b = Board(height,width)
    tile_data = b.tile_data
    
    
    tiles = board[:height*width]
    p_tiles = board[height*width:height*width*3]
    p_spirits = board[height*width*3:-2]
    taken_tiles = p_tiles.reshape(2,height*width)
    spirit_board = p_spirits.reshape(2,3)
    print(p_spirits)
    for ind,t in enumerate(tiles):
        tile = tile_data[t]
        if len(tile)>1:
            space = '\t'
        else:
            space = '\t\t'
        if t in spirit_board[0]:
            bcolor = 'on_red'
        elif t in spirit_board[1]:
            bcolor = 'on_cyan'
        else:
            bcolor = 'on_grey'
        if t in taken_tiles[0]:
            cprint(colored(tile,'red',bcolor),end=space)
        elif t in taken_tiles[1]:
            cprint(colored(tile,'cyan',bcolor),end=space)
        else:
            cprint(colored(tile,'white',bcolor),end=space)
        if (ind+1) % 12 == 0:
            print()
            
            