from copy import deepcopy
from Gaddag import Dictionary, DELIMITER, Arc
from utils import *
import Tile
import string

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class Board():
    def __init__(self):
        super().__init__()
        self.board = b()

    def __str__(self):
        board_string = ""
        for i in range(len(self.board)):
            row = (self.square((i, j)).letter for j in range(len(self.board)))
            row_string = "  ".join(tile if tile else "-" for tile in row)
            board_string += row_string + "\n"
        return board_string

    def square(self, coord):
        row, col = coord
        if row < 0 or row >= len(self.board):
            return None
        if col < 0 or col >= len(self.board):
            return None
        return self.board[row][col]

    def scorePlay(self, coord, word, direction):
        letters_used = 0
        main_word_score = 0
        word_multiplier = 1
        other_direction = 0 if direction == 1 else 1
        cur = coord

        for letter in word:
            sq = self.square(cur)
            if not sq:
                break

            if sq.letter:  # Existing tile
                main_word_score += points[letter]
            else:  # New tile placed
                letters_used += 1

                # Add cross-word score if formed
                cross_score = self._score_cross_word(cur, letter, other_direction)
                main_word_score += cross_score

                # Apply tile multipliers for main word
                letter_score = points[letter]
                if sq.multiplier == "DL":
                    letter_score *= 2
                elif sq.multiplier == "TL":
                    letter_score *= 3
                elif sq.multiplier == "DW":
                    word_multiplier *= 2
                elif sq.multiplier == "TW":
                    word_multiplier *= 3

                main_word_score += letter_score

            cur = self.offset(cur, direction, 1)

        total_score = main_word_score * word_multiplier
        if letters_used == 7:
            total_score += 50  # Bingo bonus

        return total_score

    def _score_cross_word(self, coord, letter, direction):
        """Score any perpendicular cross-word formed by a new tile placed at coord."""
        left = self.offset(coord, direction, -1)
        right = self.offset(coord, direction, 1)
        has_cross = (self.square(left) and self.square(left).letter) or (self.square(right) and self.square(right).letter)
        if not has_cross:
            return 0

        cross_score = points[letter]
        sq = self.square(coord)
        if sq.multiplier == "DL":
            cross_score *= 2
        elif sq.multiplier == "TL":
            cross_score *= 3

        # Add letters to the left
        l = left
        while self.square(l) and self.square(l).letter:
            cross_score += points[self.square(l).letter]
            l = self.offset(l, direction, -1)

        # Add letters to the right
        r = right
        while self.square(r) and self.square(r).letter:
            cross_score += points[self.square(r).letter]
            r = self.offset(r, direction, 1)

        return cross_score

    def generate_moves(self, anchor, direction, rack, dictionary, tile_set, anchors_used):
        plays = []

        def gen(pos_, word_, rack_, arc_, new_tiles_, wild_cards_):
            rack_ = deepcopy(rack_)
            coord = self.offset(anchor, direction, pos_)
            tile = self.square(coord).letter
            if tile:
                new_tiles_ = deepcopy(new_tiles_)
                go_on(pos_, tile, word_, rack_, arc_.get_next(tile), arc_, new_tiles_, wild_cards_)
            elif rack_:
                other_direction = 1 if direction == 0 else 0
                for letter_ in (x for x in set(rack_) if x in self.square(coord).playable[other_direction]):
                    tmp_rack_ = deepcopy(rack_)
                    tmp_rack_.remove(letter_)
                    tmp_new_tiles_ = deepcopy(new_tiles_)
                    tmp_new_tiles_.append(pos_)
                    go_on(pos_, letter_, word_, tmp_rack_, arc_.get_next(letter_), arc_, tmp_new_tiles_, wild_cards_)
                if "?" in rack_:
                    for letter_ in (x for x in LETTERS if x in self.square(coord).playable[other_direction]):
                        tmp_rack_ = deepcopy(rack_)
                        tmp_rack_.remove("?")
                        tmp_new_tiles_ = deepcopy(new_tiles_)
                        tmp_new_tiles_.append(pos_)
                        tmp_wild_cards_ = deepcopy(wild_cards_)
                        tmp_wild_cards_.append(pos_)
                        next_arc = arc_.get_next(letter_)
                        go_on(pos_, letter_, word_, tmp_rack_, next_arc, arc_, tmp_new_tiles_, tmp_wild_cards_)

        def go_on(pos_, char_, word_, rack_, new_arc_, old_arc_, new_tiles_, wild_cards_):
            directly_left = self.offset(anchor, direction, pos_ - 1)
            directly_left_square = self.square(directly_left)
            directly_right = self.offset(anchor, direction, pos_ + 1)
            directly_right_square = self.square(directly_right)
            right_side = self.offset(anchor, direction, 1)
            right_side_square = self.square(right_side)

            if pos_ <= 0:
                word_ = char_ + word_
                left_good = not directly_left_square or not directly_left_square.letter
                right_good = not right_side_square or not right_side_square.letter
                if char_ in old_arc_.letter_set and left_good and right_good and new_tiles_:
                    temp_word = word_[:]
                    for i in wild_cards_:
                        i = i - pos_
                        temp_word = temp_word[:i] + temp_word[i].lower() + temp_word[i+1:]
                    plays.append((temp_word, self.scorePlay(self.offset(anchor, direction, pos_), temp_word, direction), self.offset(anchor, direction, pos_), direction))
                if new_arc_:
                    if directly_left_square and directly_left not in anchors_used:
                        gen(pos_ - 1, word_, rack_, new_arc_, new_tiles_, wild_cards_)
                    new_arc_ = new_arc_.get_next(DELIMITER)
                    if new_arc_ and left_good and right_side_square:
                        gen(1, word_, rack_, new_arc_, new_tiles_, wild_cards_)
            else:
                word_ = word_ + char_
                right_good = not directly_right_square or not directly_right_square.letter
                if char_ in old_arc_.letter_set and right_good and new_tiles_:
                    left_most = pos_ - len(word_) + 1
                    temp_word = word_[:]
                    for i in wild_cards_:
                        index = i - left_most
                        temp_word = temp_word[:index] + temp_word[index].lower() + temp_word[index+1:]
                    plays.append((temp_word, self.scorePlay(self.offset(anchor, direction, left_most), temp_word, direction), self.offset(anchor, direction, left_most), direction))
                if new_arc_ and directly_right_square:
                    gen(pos_ + 1, word_, rack_, new_arc_, new_tiles_, wild_cards_)

        initial_arc = Arc("", dictionary.root)
        gen(0, "", deepcopy(rack), initial_arc, [], [])
        return plays

    def update_cross_set(self, start_coordinate, direction, dictionary):
        """update cross sets affected by this coordinate"""
        # Code same as before (no issues in your original)

        def __clear_cross_sets(start_coordinate_, direction_):
            right_most_square = self.fast_forward(start_coordinate_, direction_, 1)
            right_square_ = self.offset(right_most_square, direction_, 1)
            if self.square(right_square_):
                self.square(right_square_).set_cross_set(direction_, {})
            left_most_square = self.fast_forward(start_coordinate_, direction_, -1)
            left_square_ = self.offset(left_most_square, direction_, -1)
            if self.square(left_square_):
                self.square(left_square_).set_cross_set(direction_, {})

        def __check_candidate(coordinate_, candidate_, direction_, step):
            last_arc_ = candidate_
            state_ = candidate_.destination
            next_square_ = self.offset(coordinate_, direction_, step)
            while self.square(next_square_) and self.square(next_square_).letter:
                coordinate_ = next_square_
                tile_ = self.square(coordinate_).letter.upper()
                last_arc_ = state_.arcs[tile_] if tile_ in state_.arcs else None
                if not last_arc_:
                    return False
                state_ = last_arc_.destination
                next_square_ = self.offset(coordinate_, direction_, step)
            return self.square(coordinate_).letter.upper() in last_arc_.letter_set

        if not self.square(start_coordinate) or not self.square(start_coordinate).letter:
            return

        end_coordinate = self.fast_forward(start_coordinate, direction, 1)
        coordinate = end_coordinate
        last_state = dictionary.root
        state = last_state.get_next(self.square(coordinate).letter.upper())
        next_square = self.offset(coordinate, direction, -1)
        while self.square(next_square) and self.square(next_square).letter:
            coordinate = next_square
            last_state = state
            state = state.get_next(self.square(coordinate).letter)
            if not state:
                __clear_cross_sets(start_coordinate, direction)
                return
            next_square = self.offset(coordinate, direction, -1)

        right_square = self.offset(end_coordinate, direction, 1)
        left_square = self.offset(coordinate, direction, -1)

        if self.square(self.offset(left_square, direction, -1)) and self.square(self.offset(left_square, direction, -1)).letter:
            candidates = (arc for arc in state if arc.char != "#")
            cross_set = set(candidate.char for candidate in candidates if __check_candidate(left_square, candidate, direction, -1))
            self.square(left_square).set_cross_set(direction, cross_set)
        elif self.square(left_square):
            cross_set = last_state.get_arc(self.square(coordinate).letter.upper()).letter_set
            self.square(left_square).set_cross_set(direction, cross_set)

        if self.square(self.offset(right_square, direction, 1)) and self.square(self.offset(right_square, direction, 1)).letter:
            end_state = state.get_next(DELIMITER)
            candidates = (arc for arc in end_state if arc != "#") if end_state else {}
            cross_set = set(candidate.char for candidate in candidates if __check_candidate(right_square, candidate, direction, 1))
            self.square(right_square).set_cross_set(direction, cross_set)
        elif self.square(right_square):
            end_arc = state.get_arc(DELIMITER)
            cross_set = end_arc.letter_set if end_arc else {}
            self.square(right_square).set_cross_set(direction, cross_set)

    @staticmethod
    def offset(coord, direction, step):
        if direction == 0:
            return coord[0], coord[1] + step
        else:
            return coord[0] + step, coord[1]

    def fast_forward(self, start_coordinate, direction, step):
        coord = start_coordinate
        next_coord = self.offset(coord, direction, step)
        while self.square(next_coord) and self.square(next_coord).letter:
            coord = next_coord
            next_coord = self.offset(coord, direction, step)
        return coord

    def place_word(self, start_coordinate, word, direction):
        end_coordinate = self.offset(start_coordinate, direction, len(word))
        if any(idx > len(self.board) for idx in end_coordinate):
            raise IllegalMoveError("Word out of bounds.")

        coordinate = start_coordinate
        offset = 0
        letters_used = []
        for char in word:
            sq = self.square(coordinate)
            if not sq.letter:
                letters_used.append(char)
                sq.letter = char
            offset += 1
            coordinate = self.offset(start_coordinate, direction, offset)
        return letters_used

    def get_letters_used(self, start_coordinate, word, direction):
        coordinate = start_coordinate
        offset = 0
        letters_used = []
        for char in word:
            sq = self.square(coordinate)
            if not sq.letter:
                letters_used.append(char)
            offset += 1
            coordinate = self.offset(start_coordinate, direction, offset)
        return letters_used

    def find_best_moves(self, rack, direction, dictionary, tile_set):
        has_any_letters = any(square.letter for row in self.board for square in row)
        anchors_used = []
        moves = []
        other_direction = 0 if direction == 1 else 1

        def is_anchor(coord):
            if self.square(coord).letter:
                return False
            if not has_any_letters:
                return coord == (7, 7)
            for delta in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (coord[0] + delta[0], coord[1] + delta[1])
                sq = self.square(neighbor)
                if sq and sq.letter:
                    return True
            return False

        corner = (0, 0)
        for i in range(len(self.board)):
            left_most = self.offset(corner, other_direction, i)
            for j in range(len(self.board)):
                current = self.offset(left_most, direction, j)
                if is_anchor(current):
                    moves.extend(self.generate_moves(current, direction, rack, dictionary, tile_set, anchors_used))
                    anchors_used.append(current)
        return moves
