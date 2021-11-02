import random
from math import log, ceil
from random import randint, choice, shuffle

from aalpy.base import Oracle, SUL


class PacOracle(Oracle):
    """
    Equivalence oracle where queries are of random length in a predefined range.
    """

    def __init__(self, alphabet: list, sul: SUL, epsilon=0.01, delta=0.01, min_walk_len=10, max_walk_len=100):

        super().__init__(alphabet, sul)
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.epsilon = epsilon
        self.delta = delta
        self.round = 0

    def find_cex(self, hypothesis):
        self.round += 1
        num_test_cases = 1 / self.epsilon * (log(1 / self.delta) + self.round * log(2))

        for i in range(ceil(num_test_cases)):
            inputs = []
            self.reset_hyp_and_sul(hypothesis)

            num_steps = randint(self.min_walk_len, self.max_walk_len)

            for _ in range(num_steps):
                inputs.append(choice(self.alphabet))

                out_sul = self.sul.step(inputs[-1])
                out_hyp = hypothesis.step(inputs[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    print(f'Num tests: {i+1}')
                    self.sul.post()
                    print(''.join(inputs))
                    return inputs

        return None


class TransitionFocusOraclePrime(Oracle):
    """
    Only differance to the one provided in AALpy is that this one prints number of test cases needed to find a cex.

    This equivalence oracle focuses either on the same state transitions or transitions that lead to the different
    states. This equivalence oracle should be used on grammars like balanced parentheses. In such grammars,
    all interesting behavior occurs on the transitions between states and potential bugs can be found only by
    focusing on transitions.
    """

    def __init__(self, alphabet, sul: SUL, num_random_walks=1000, walk_len=20, same_state_prob=0.2):
        """
        Args:
            alphabet: input alphabet
            sul: system under learning
            num_random_walks: number of walks
            walk_len: length of each walk
            same_state_prob: probability that the next input will lead to same state transition
        """

        super().__init__(alphabet, sul)
        self.num_walks = num_random_walks
        self.steps_per_walk = walk_len
        self.same_state_prob = same_state_prob

    def find_cex(self, hypothesis):

        for i in range(self.num_walks):
            self.reset_hyp_and_sul(hypothesis)

            curr_state = hypothesis.initial_state
            inputs = []
            for _ in range(self.steps_per_walk):
                if random.random() <= self.same_state_prob:
                    possible_inputs = curr_state.get_same_state_transitions()
                else:
                    possible_inputs = curr_state.get_diff_state_transitions()

                act = random.choice(possible_inputs) if possible_inputs else random.choice(self.alphabet)
                inputs.append(act)

                out_sul = self.sul.step(inputs[-1])
                out_hyp = hypothesis.step(inputs[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    print(f'Num tests: {i+1}')
                    self.sul.post()
                    return inputs

        return None

class RandomWMethodEqOraclePrime(Oracle):
    """
    Only differance to the one provided in AALpy is that this one prints number of test cases needed to find a cex.
    Randomized version of the W-Method equivalence oracle.
    Random walks stem from fixed prefix (path to the state). At the end of the random
    walk an element from the characterization set is added to the test case.
    """

    def __init__(self, alphabet: list, sul: SUL, walks_per_state=10, walk_len=20):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state: number of random walks that should start from each state

            walk_len: length of random walk
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.random_walk_len = walk_len
        self.freq_dict = dict()

    def find_cex(self, hypothesis):

        num_test_cases = 0
        states_to_cover = []
        for state in hypothesis.states:
            if state.prefix not in self.freq_dict.keys():
                self.freq_dict[state.prefix] = 0

            states_to_cover.extend([state] * (self.walks_per_state - self.freq_dict[state.prefix]))

        shuffle(states_to_cover)

        for state in states_to_cover:
            self.freq_dict[state.prefix] = self.freq_dict[state.prefix] + 1

            self.reset_hyp_and_sul(hypothesis)

            prefix = state.prefix
            random_walk = tuple(choice(self.alphabet) for _ in range(randint(1, self.random_walk_len)))

            test_case = prefix + random_walk + choice(hypothesis.characterization_set)

            num_test_cases += 1
            for ind, i in enumerate(test_case):
                output_hyp = hypothesis.step(i)
                output_sul = self.sul.step(i)
                self.num_steps += 1

                if output_sul != output_hyp:
                    print(f'Num tests: {num_test_cases}')
                    self.sul.post()
                    return test_case[:ind + 1]

        return None

