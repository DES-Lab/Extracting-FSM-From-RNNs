from aalpy.base import SUL


class RnnBinarySUL(SUL):
    def __init__(self, nn):
        super().__init__()
        self.word = ""
        self.rnn = nn

    def pre(self):
        self.rnn.state = self.rnn.rnn.initial_state()

    def post(self):
        self.rnn.renew()

    def step(self, letter):
        if letter is None:
            return self.rnn.step(None)
        return self.rnn.step(letter)


class RnnMealySUL(SUL):
    def __init__(self, nn, int_2_output_dict):
        super().__init__()
        self.rnn = nn
        self.seq = []
        self.int_2_output_dict = int_2_output_dict

    def pre(self):
        self.rnn.state = self.rnn.rnn.initial_state()

    def post(self):
        self.rnn.renew()

    def step(self, letter):
        out = self.rnn.step(letter)
        return self.int_2_output_dict[out]


class RNN_BinarySUL_for_Weiss_Framework(SUL):
    def __init__(self, nn):
        super().__init__()
        self.word = ""
        self.rnn = nn

    def query(self, input_word: tuple) -> list:
        self.pre()
        out = []
        # Empty string for DFA
        if len(input_word) == 0:
            return self.step(None)
        out = [self.predict(input_word)]
        self.post()
        return out

    def pre(self):
        self.word = ""

    def post(self):
        self.rnn.renew()

    def step(self, letter):
        if letter is not None:
            self.word += letter
        return self.rnn.classify_word(self.word)

    def predict(self, seq):
        prediction = self.rnn.classify_word(seq)
        return prediction
