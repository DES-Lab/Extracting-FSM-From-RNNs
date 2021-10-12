from collections import defaultdict
from random import choice

from aalpy.base import SUL


class RnnBinarySUL(SUL):
    """
    SUL used to learn DFA from RNN Binary Classifiers.
    """

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
    """
    SUL used to learn Mealy Machines from RNNs.
    """

    def __init__(self, nn, int_2_output_dict: dict):
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
    """
    SUL conforming to the behaviour and methods found in the refinement-based learning framework by Weiss et al.
    https://github.com/tech-srl/lstar_extraction
    """

    def __init__(self, nn):
        super().__init__()
        self.word = ""
        self.rnn = nn

    def pre(self):
        self.word = ""

    def post(self):
        self.rnn.renew()

    def step(self, letter):
        if letter is not None:
            self.word += letter
        return self.rnn.classify_word(self.word)


class Abstract_Mapper_MQTT_RNN_SUL(SUL):
    """
    SUL implementing the MAPPER component.
    """

    def __init__(self, nn, concrete_input_al, concrete_output_al):
        super().__init__()
        self.rnn = nn
        self.seq = []

        self.abstract_inputs = ['connect', 'disconnect', 'subscribe', 'unsubscribe', 'publish', 'invalid']
        self.abstract_outputs = ['CONNACK', 'CONCLOSED', 'SUBACK', 'UNSUBACK', 'PUBACK__PUBLISH', 'PUBACK']

        self.concrete_inputs = concrete_input_al
        # self.concrete_inputs.sort()

        self.concrete_outputs = concrete_output_al
        self.topics = []
        self.current_topic = None

        self.abstract_2_concrete_inputs_map = defaultdict(list)

        self.int_2_output_dict = {i: o for i, o in enumerate(concrete_output_al)}
        self.create_abstract_alphabets()

    def create_abstract_alphabets(self):
        import re
        quoted = re.compile('"([^"]*)"')

        for i in self.concrete_inputs:
            for ai in self.abstract_inputs:
                if i.startswith(ai):
                    if 'topic' in i:
                        for value in quoted.findall(i):
                            if value not in self.topics:
                                self.topics.append(value)
                    self.abstract_2_concrete_inputs_map[ai].append(i)
                    break

    def pre(self):
        self.rnn.state = self.rnn.rnn.initial_state()
        self.current_topic = choice(self.topics)

    def post(self):
        self.seq.clear()
        self.rnn.renew()

    def step(self, letter):
        concrete_inputs = self.abstract_2_concrete_inputs_map[letter]

        if 'topic' in concrete_inputs[0]:
            current_topic = None
            for cv in concrete_inputs:
                if self.current_topic in cv:
                    current_topic = cv
                    break
            assert current_topic
            concrete_input = current_topic
        else:
            concrete_input = concrete_inputs[0]

        self.seq.append(concrete_input)

        out = self.rnn.step(concrete_input)

        concrete_out = self.int_2_output_dict[out]

        abstract_output = None
        for ao in self.abstract_outputs:
            if concrete_out.startswith(ao):
                abstract_output = ao
                break
        assert abstract_output

        return abstract_output
