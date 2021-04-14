from random import randint, choice

from aalpy.base import Oracle, SUL


class LongCexEqOracle(Oracle):
    """
    Equivalence oracle where queries are of random length in a predefined range.
    """

    def __init__(self, alphabet: list, sul: SUL, num_walks=100, min_walk_len=1, max_walk_len=50,
                 reset_after_cex=True):
        """
        Args:
            alphabet: input alphabet
            sul: system under learning
            num_walks: number of walks to perform during search for cex
            min_walk_len: minimum length of each walk
            max_walk_len: maximum length of each walk
            reset_after_cex: if True, num_walks will be preformed after every counter example, else the total number
                or walks will equal to num_walks
        """

        super().__init__(alphabet, sul)
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.reset_after_cex = reset_after_cex
        self.num_walks_done = 0

    def find_cex(self, hypothesis):

        while self.num_walks_done < self.num_walks:
            self.sul.post()
            self.sul.pre()
            hypothesis.reset_to_initial()
            self.num_queries += 1
            self.num_walks_done += 1
            cex_found = False

            num_steps = randint(self.min_walk_len, self.max_walk_len)

            inputs = [choice(self.alphabet) for _ in range(num_steps)]

            for i in inputs:
                out_sul = self.sul.step(i)
                out_hyp = hypothesis.step(i)
                self.num_steps += 1

                if out_sul != out_hyp:
                    cex_found = True
                    break

            if cex_found:
                if self.reset_after_cex:
                    self.num_walks_done = 0

                return inputs

        return None
