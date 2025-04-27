import pickle
import gzip

DELIMITER = "#"

class Dictionary:
    def __init__(self, root):
        self.root = root

    def store(self, filename):
        with gzip.open(filename, "wb") as f:
            f.write(pickle.dumps(self.root))

    @classmethod
    def construct_with_text_file(cls, filename):
        with open(filename) as f:
            words = f.readlines()
        word_list = set(x.strip().upper() for x in words)
        root = cls.__construct_lexicon_with_list_of_words(word_list)
        return cls(root)

    @classmethod
    def load_from_pickle(cls, filename):
        with gzip.open(filename, "rb") as f:
            root = pickle.loads(f.read())
        return cls(root)

    @staticmethod
    def __construct_lexicon_with_list_of_words(word_list):
        root = State()
        for word in word_list:
            word = word.upper()
            Dictionary.__add_word(root, word)
        return root

    @staticmethod
    def __add_word(root, word):
        state = root
        for char in word[len(word):1:-1]:
            state = state.add_arc(char)
        state.add_final_arc(word[1], word[0])

        state = root
        for char in word[len(word) - 2::-1]:
            state = state.add_arc(char)
        state = state.add_final_arc(DELIMITER, word[-1])

        for m in range(len(word) - 2, 0, -1):
            destination = state
            state = root
            for char in word[m - 1::-1]:
                state = state.add_arc(char)
            state = state.add_arc(DELIMITER)
            state.add_arc(word[m], destination)

class State:
    def __init__(self):
        self.arcs = dict()
        self.letter_set = set()

    def __iter__(self):
        for char in self.arcs:
            yield self.arcs[char]

    def __contains__(self, char):
        return char in self.arcs

    def get_arc(self, char):
        return self.arcs[char] if char in self.arcs else None

    def add_arc(self, char, destination=None):
        if char not in self.arcs:
            self.arcs[char] = Arc(char, destination)
        return self.get_next(char)

    def add_final_arc(self, char, final):
        if char not in self.arcs:
            self.arcs[char] = Arc(char)
        self.get_next(char).add_letter(final)
        return self.get_next(char)

    def add_letter(self, char):
        self.letter_set.add(char)

    def get_next(self, char):
        return self.arcs[char].destination if char in self.arcs else None

class Arc:
    def __init__(self, char, destination=None):
        self.char = char
        if not destination:
            destination = State()
        self.destination = destination

    def __contains__(self, char):
        return char in self.destination.letter_set

    def __eq__(self, other):
        return other == self.char

    @property
    def letter_set(self):
        return self.destination.letter_set if self.destination else set()

    def get_next(self, char):
        return self.destination.arcs[char] if char in self.destination.arcs else None
