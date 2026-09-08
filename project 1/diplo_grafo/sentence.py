class Sentence:
    def __init__(self, text, pos):
        self.text = text
        self.pos = pos
        self.tokens = []
        self.denotations = []

    def __str__(self):
        return self.text