from aalpy.base import SUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle

from TrainAndExtract import train_RNN_on_tomita_grammar


class InternalSUL(SUL):

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def pre(self):
        self.rnn.state = self.rnn.rnn.initial_state()

    def post(self):
        self.rnn.renew()

    def step(self, letter):
        if letter is None:
            return self.rnn.step(None)
        return self.rnn.step_internal(letter)


rnn = train_RNN_on_tomita_grammar(tomita_grammar=3, train=False)

sul = InternalSUL(rnn)
alphabet = ['0', '1']

eq_oracle = StatePrefixEqOracle(alphabet, sul)

model = run_Lstar(alphabet, sul, eq_oracle, automaton_type='mealy')

print(model)
