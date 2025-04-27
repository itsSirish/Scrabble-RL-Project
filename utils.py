from Tile import Tile
import numpy as np

TRIPLE_WORD = "TW"
TRIPLE_LETTER = "TL"
DOUBLE_WORD = "DW"
DOUBLE_LETTER = "DL"

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

points = {
    'A': 1,
    'B': 3,
    'C': 3,
    'D': 2,
    'E': 1,
    'F': 4,
    'G': 2,
    'H': 4,
    'I': 1,
    'J': 8,
    'K': 5,
    'L': 1,
    'M': 3,
    'N': 1,
    'O': 1,
    'P': 3,
    'Q': 10,
    'R': 1,
    'S': 1,
    'T': 1,
    'U': 1,
    'V': 4,
    'W': 4,
    'X': 8,
    'Y': 4,
    'Z': 10,
    'a': 0,
    'b': 0,
    'c': 0,
    'd': 0,
    'e': 0,
    'f': 0,
    'g': 0,
    'h': 0,
    'i': 0,
    'j': 0,
    'k': 0,
    'l': 0,
    'm': 0,
    'n': 0,
    'o': 0,
    'p': 0,
    'q': 0,
    'r': 0,
    's': 0,
    't': 0,
    'u': 0,
    'v': 0,
    'w': 0,
    'x': 0,
    'y': 0,
    'z': 0
}


BAG = list("AAAAAAAAABBCCDDDDEEEEEEEEEEEEFFGGGHHIIIIIIIIIJKLLLLMMNNNNNNOOOOOOOOPPQRRRRRRSSSSTTTTTTUUUUVVWWXYYZ??")
def bag():
    return np.array(BAG)


def letters():
    return LETTERS

class IllegalMoveError(Exception):
    pass


def b():
    BOARD = [[Tile(TRIPLE_WORD), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(TRIPLE_WORD), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(TRIPLE_WORD)],
         [Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(
         ), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile()],
         [Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(
         ), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile()],
         [Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile(
             DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(DOUBLE_LETTER)],
         [Tile(), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(
         ), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile()],
         [Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(
         ), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(), Tile(), Tile(TRIPLE_LETTER), Tile()],
         [Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(
         ), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile()],
         [Tile(TRIPLE_WORD), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(
             DOUBLE_WORD), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(TRIPLE_WORD)],
         [Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(
         ), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile()],
         [Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(
         ), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(), Tile(), Tile(TRIPLE_LETTER), Tile()],
         [Tile(), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(
         ), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile()],
         [Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile(
             DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(DOUBLE_LETTER)],
         [Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(
         ), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile(), Tile()],
         [Tile(), Tile(DOUBLE_WORD), Tile(), Tile(), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(
         ), Tile(), Tile(TRIPLE_LETTER), Tile(), Tile(), Tile(), Tile(DOUBLE_WORD), Tile()],
         [Tile(TRIPLE_WORD), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(), Tile(
             TRIPLE_WORD), Tile(), Tile(), Tile(), Tile(DOUBLE_LETTER), Tile(), Tile(), Tile(TRIPLE_WORD)]
         ]
    return BOARD


def mapBoardToState(game, leave):
    state = {}
    bag = [0] * 27
    rack = [0] * 27

    for letter in game.bag:
        if letter == '?':
            bag[0] += 1
        else:
            bag[ord(letter) - 64] += 1

    for letter in game.players[game.currentPlayer].rack:
        if letter == '?':
            bag[0] += 1
        else:
            bag[ord(letter) - 64] += 1

    state['bag'] = bag

    for letter in leave:
        if letter == '?':
            rack[0] += 1
        else:
            rack[ord(letter) - 64] += 1

    state['rack'] = rack
    state['score'] = [
        game.players[not game.currentPlayer].score,
        game.players[not game.currentPlayer].score - game.players[game.currentPlayer].score
    ]

    board = [[[0, 0] for _ in range(15)] for _ in range(15)]
    for i in range(len(game.board.board)):
        for j in range(len(game.board.board)):
            tile = game.board.board[i][j]
            board[i][j][0] = 0 if tile.letter is None else ord(tile.letter.upper()) - 64
            board[i][j][1] = multiplierToNumber(tile.multiplier)

    state['board'] = board
    return state

def multiplierToNumber(ml):
    if ml is None:
        return 0
    return {"DL": 1, "TL": 2, "DW": 3, "TW": 4}.get(ml, 0)
