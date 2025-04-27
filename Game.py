 
from Player import Player
from utils import *
from Board import Board
import os
from Gaddag import *
import numpy as np

HORIZ = 0
VERT = 1

class Game():
    def __init__(self):
        self.players = [Player(), Player()]
        self.bag = BAG
        self.board = Board()
        self.currentPlayer = 0
        self.numMoves = 0
        self.gameOver = False
        dictionary_path = "dictionary.txt"
        saved_dictionary_path = "dictionary.p"

        if os.path.exists(saved_dictionary_path):
            self.dictionary = Dictionary.load_from_pickle(saved_dictionary_path)
        else:
            self.dictionary = Dictionary.construct_with_text_file(dictionary_path)
            self.dictionary.store(saved_dictionary_path)

        self.initGame()

    def exchangeSeven(self, player):
        np.random.shuffle(self.bag)
        for i in range(len(self.players[player].rack)):
            self.bag = np.append(self.bag, self.players[player].rack[i])
            self.players[player].rack.remove(self.players[player].rack[i])
            self.players[player].rack.append(self.bag[0])

    def initGame(self):
        for _ in range(np.random.randint(5, 10)):
            np.random.shuffle(self.bag)
        for i in range(7):
            self.players[0].rack.append(self.bag[0])
            self.bag = np.delete(self.bag, 0)
            np.random.shuffle(self.bag)
        for i in range(7):
            self.players[1].rack.append(self.bag[0])
            self.bag = np.delete(self.bag, 0)
            np.random.shuffle(self.bag)

    def find_best_moves(self, rack, num=5):
        rack = list(rack)
        if self.numMoves == 0:
            moves = self.board.generate_moves((7, 7), HORIZ, rack, self.dictionary, self.bag, {})
        else:
            across_moves = self.board.find_best_moves(rack, HORIZ, self.dictionary, self.bag)
            down_moves = self.board.find_best_moves(rack, VERT, self.dictionary, self.bag)
            moves = across_moves + down_moves
        moves.sort(key=lambda move_: move_[1], reverse=True)
        return moves[:num]

    def play(self, start_square, word, direction):
        lettersUsed = self.board.place_word(start_square, word, direction)

        # ✅ Add these two lines
        score = self.board.scorePlay(start_square, word, direction)
        self.players[self.currentPlayer].score += score

        for _ in range(np.random.randint(1, 10)):
            np.random.shuffle(self.bag)

        tilesDrawn = self.players[self.currentPlayer].drawTiles(self.bag, lettersUsed)
        for _ in range(tilesDrawn):
            self.bag = np.delete(self.bag, 0)

        if len(self.players[self.currentPlayer].rack) == 0:
            self.endGame()
            return

        self.board.update_cross_set(start_square, direction, self.dictionary)
        other_direction = HORIZ if direction == VERT else VERT
        coordinate = start_square
        for _ in word:
            self.board.update_cross_set(coordinate, other_direction, self.dictionary)
            coordinate = self.board.offset(coordinate, direction, 1)

        self.currentPlayer = not self.currentPlayer
        self.numMoves += 1

    def clear(self):
        self.players = [Player(), Player()]
        self.bag = BAG
        self.board = Board()
        self.currentPlayer = 0
        self.numMoves = 0
        self.initGame()

    def endGame(self):
        self.numMoves = -1
