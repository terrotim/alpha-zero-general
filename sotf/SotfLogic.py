from collections import namedtuple
import numpy as np

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4

WinState = namedtuple('WinState', 'is_ended winner')


class Board():
    """
    Connect4 Board.
    """

    def __init__(self, height=None, width=None, tiles=None):
        self.height = height
        self.width = width
        
        if tiles is None:
            self.tiles = np.arange(height*width)
            #Randomize tiles
            #np.random.shuffle(self.tiles)
            
            self.p_tiles = np.full(height*width*2,-1)
            self.p_spirits = np.full(6,-1)

        else:
            self.tiles = tiles
            
        self.initiate_tiles()
        
        self.layout = np.concatenate((self.tiles,self.p_tiles,self.p_spirits,[-1],[1]))
        
    def execute_move(self, action, player):
        tiles = self.layout[:self.height*self.width]
        p_tiles = self.layout[self.height*self.width:self.height*self.width*3]
        p_spirits = self.layout[self.height*self.width*3:-2]
        first_claim = self.layout[-2]
        action_num = self.layout[-1]
        
        spirit_used,tile_index  = np.unravel_index(action,[3,self.height*self.width])
        taken_tiles = p_tiles.reshape(2,self.height*self.width)
        spirit_board = p_spirits.reshape(2,3)
        own_tiles_list = taken_tiles[0 if player == 1 else 1]
        own_spirits = spirit_board[0 if player == 1 else 1]
        other_spirits = spirit_board[1 if player == 1 else 0]
            
        if action_num < 3:
            #if it is action 1 or 2, its a claim action
            own_tiles = [t for t in own_tiles_list if not t == -1]
            own_tiles.append(tile_index)
            own_tiles.sort()
            while len(own_tiles) < len(own_tiles_list):
                own_tiles.append(-1)
            if player == 1:
                taken_tiles[0] = own_tiles
            else:
                taken_tiles[1] = own_tiles
            p_tiles = taken_tiles.flatten()
            
            #if the tiles[tile_index] was reserved in p_spirits
            if tile_index in p_spirits:
                #print('p_spirits',p_spirits)
                if tile_index in own_spirits:
                    s_used = np.where(own_spirits == tile_index)[0]
                    own_spirits[s_used] = -1
                else:
                    own_spirits[spirit_used] = -2
                    s_used = np.where(other_spirits == tile_index)[0]
                    other_spirits[s_used] = -1
                #print('p_spirits2',p_spirits)
                """
                if player == 1:
                    spirit_board[0] = own_spirits
                    spirit_board[1] = other_spirits
                else:
                    spirit_board[1] = own_spirits
                    spirit_board[0] = other_spirits
                p_spirits = spirit_board.flatten()
                print('p_spirits3',p_spirits)
                """
            if action_num == 1:
                first_claim = tile_index
            else:
                first_claim = -1
            return np.concatenate((tiles,p_tiles,p_spirits,[first_claim],[action_num]))

            
        else:
            #its a reserve action
            first_claim = -1
            own_spirits[spirit_used] = tile_index
            if player == 1:
                spirit_board[0] = own_spirits
            else:
                spirit_board[1] = own_spirits
            p_spirits = spirit_board.flatten()
            return np.concatenate((tiles,p_tiles,p_spirits,[first_claim],[action_num]))
            
    def is_legal_move(self,action,player):
        tiles = self.layout[:self.height*self.width]
        p_tiles = self.layout[self.height*self.width:self.height*self.width*3]
        p_spirits = self.layout[self.height*self.width*3:-2]
        first_claim = self.layout[-2]
        action_num = self.layout[-1]
        
        spirit_used,tile_index = np.unravel_index(action,[3,self.height*self.width])
        taken_tiles = p_tiles.reshape(2,self.height*self.width)
        spirit_board = p_spirits.reshape(2,3)
        own_tiles_list = taken_tiles[0 if player == 1 else 1]
        own_spirits = spirit_board[0 if player == 1 else 1]
        other_spirits = spirit_board[1 if player == 1 else 0]
        
        if action_num < 3:
            if tile_index in p_tiles:
                return 0
            if tile_index in other_spirits and own_spirits[spirit_used] == -2:
                return 0
                
            if action_num == 2:
                #print(action)
                if first_claim == -1:
                    return 0
                spirits1 = self.tile_data[first_claim]
                spirits2 = self.tile_data[tile_index]
                if first_claim == tile_index:
                    return 0
                if not spirits1[0] == spirits2[0]:
                    return 0
                if len(spirits1) > 1 and spirits1[0] == spirits1[1]: 
                    return 0
                if len(spirits2) > 1 and spirits2[0] == spirits2[1]:
                    return 0
            
            tile_board = tiles.reshape(self.height,self.width)
            edge_tiles = []
            for row in tile_board:
                fr = [t for t in row if not t in p_tiles]
                if len(fr) > 0:
                    if fr[0] not in edge_tiles:
                        edge_tiles.append(fr[0])
                    if fr[-1] not in edge_tiles:
                        edge_tiles.append(fr[-1])
            if not tile_index in edge_tiles:
                return 0
            return 1

        else:
            if tile_index in p_tiles or tile_index in p_spirits:
                return 0
            if own_spirits[spirit_used] == -2:
                return 0
            return 1
                
    def all_tiles_taken(self):
        p_tiles = self.layout[self.height*self.width:self.height*self.width*3]
        
        all_taken = [t for t in p_tiles if not t == -1]
    
        return len(all_taken) == self.height*self.width
        
    def get_scores(self,player):
        p1score = 0
        p2score = 0
        p1spirits = []
        p2spirits = []
        
        p_tiles = self.layout[self.height*self.width:self.height*self.width*3]
        taken_tiles = p_tiles.reshape(2,self.height*self.width)
        p1_tiles = taken_tiles[0 if player == 1 else 1]
        p2_tiles = taken_tiles[1 if player == 1 else 0]
    
        p1_spirits = np.concatenate([self.tile_data[t] for t in p1_tiles if not t == -1])
        p2_spirits = np.concatenate([self.tile_data[t] for t in p2_tiles if not t == -1])
        
        unique1, counts1 = np.unique(p1_spirits, return_counts=True)
        unique2, counts2 = np.unique(p2_spirits, return_counts=True)
        
        p1dict = dict(zip(unique1, counts1))
        p2dict = dict(zip(unique2, counts2))
        """
        print('self.layout',self.layout)
        print('p1dict',p1dict)
        print('p2dict',p2dict)
        """
        for tile in ['a','b','c','d','e','f','g','h','i','x','y','z']:
            p1count = p1dict.get(tile,0)
            p2count = p2dict.get(tile,0)
            
            if p1count == 0:
                p1score -= 3
            if p2count == 0:
                p2score -= 3
            if p1count >= p2count:
                p1score += p1count
            if p2count >= p1count:
                p2score += p2count
        """
        print('p1score:',p1score)
        print('p2score:',p2score)
        input("Press any button to continue...")
        """
        return p1score, p2score
        
    def initiate_tiles(self):
        # initiates each tile's symbols
        # 5 = green, 4 tiles, a
        # 6 = red, 5 tiles, b
        # 6 = black, 4 tiles, c
        # 7 = blue, 5 tiles, d
        # 7 = brown, 5 tiles, e
        # 8 = tan, 6 tiles, f
        # 8 = maroon, 6 tiles, g 
        # 8 = orange, 6 tiles, h 
        # 10 = purple, 7 tiles, i
        # sun = x,
        # moon = y,
        # fire = z,
        data = {}
        
        data[0] = ['a','a']
        data[1] = ['a','x']
        data[2] = ['a','y']
        data[3] = ['a','z']
        data[4] = ['b','b']
        data[5] = ['b','x']
        data[6] = ['b','y']
        data[7] = ['b','z']
        data[8] = ['b','z']
        data[9] = ['c','c']
        data[10] = ['c','c']
        data[11] = ['c','x']
        data[12] = ['c','y']
        data[13] = ['d','d']
        data[14] = ['d','d']
        data[15] = ['d','x']
        data[16] = ['d','y']
        data[17] = ['d','z']
        data[18] = ['e','e']
        data[19] = ['e','e']
        data[20] = ['e','x']
        data[21] = ['e','y']
        data[22] = ['e','z']
        data[23] = ['f']
        data[24] = ['f','f']
        data[25] = ['f','f']
        data[26] = ['f','x']
        data[27] = ['f','y']
        data[28] = ['f','z']
        data[29] = ['g']
        data[30] = ['g','g']
        data[31] = ['g','g']
        data[32] = ['g','x']
        data[33] = ['g','y']
        data[34] = ['g','z']
        data[35] = ['h']
        data[36] = ['h','h']
        data[37] = ['h','h']
        data[38] = ['h','x']
        data[39] = ['h','y']
        data[40] = ['h','z']
        data[41] = ['i']
        data[42] = ['i','i']
        data[43] = ['i','i']
        data[44] = ['i','i']
        data[45] = ['i','x']
        data[46] = ['i','y']
        data[47] = ['i','z']
        self.tile_data = data
