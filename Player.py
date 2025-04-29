import numpy as np

class Player:
    def __init__(self):
        self.rack = []
        self.score = 0

    def drawTiles(self, bag, lettersUsed):
        drawn = 0
        for letter in lettersUsed:
            try:
                self.rack.remove(letter)
            except ValueError:
                # 🔥 Letter already not present, continue safely
                pass
        while len(self.rack) < 7 and len(bag) > 0:
            self.rack.append(bag[0])
            bag = np.delete(bag, 0)
            drawn += 1
        return drawn