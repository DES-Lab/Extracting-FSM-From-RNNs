import dynet as dy
import numpy as np

nn_type_options = ["LSTM", "GRU"]


class RNNClassifier:
    def __init__(self, alphabet, num_layers, hidden_dim, x_train, y_train, x_test=None, y_test=None,
                 batch_size=32, nn_type="LSTM"):
        assert nn_type in nn_type_options
        self.vocab_size = len(alphabet) + 1
        input_dim = self.vocab_size
        num_of_classes = len(set(y_train)) if not y_test else len(set(y_test).union(set(y_train)))

        self.state = None

        self.state_size = hidden_dim
        self.token_dict = dict((c, i) for i, c in enumerate(alphabet))

        self.pc = dy.ParameterCollection()
        self.input_lookup = self.pc.add_lookup_parameters((self.vocab_size, input_dim))  # TODO DOUBLE-CHECK
        self.W = self.pc.add_parameters((input_dim, hidden_dim))  # TODO DOUBLE-CHECK
        nn_fun = dy.LSTMBuilder if nn_type == "LSTM" else dy.GRUBuilder
        self.rnn = nn_fun(num_layers, input_dim, hidden_dim, self.pc)

        self.x_train, self.y_train = self._to_batch(x_train, y_train, batch_size)

        self.x_test, self.y_test = None, None
        if x_test:
            self.x_test, self.y_test = self._to_batch(x_test, y_test, batch_size)
            # self.x_test = list(map(self._pad_batch, b_x))
            # self.y_test = b_y

    def _to_batch(self, x, y, batch_size):
        data = list(zip(*sorted(zip(x, y), key=lambda k: len(k[0]))))
        batched_X = []
        batched_Y = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            batched_X.append(data[0][i * batch_size:(i + 1) * batch_size])
            batched_Y.append(data[1][i * batch_size:(i + 1) * batch_size])

        # to prevent bug in validate
        if len(batched_X[-1]) == 1:
            batched_X.pop()
            batched_Y.pop()

        padded_x = []
        for batch in batched_X:
            max_len = len(max(batch, key=len))
            tmp = []
            for x in batch:
                tmp.append([self.vocab_size - 1] * (max_len - len(x)) + x)
            padded_x.append(tmp)

        return padded_x, batched_Y

    def _get_probabilities_over_batch(self, batch):
        dy.renew_cg()
        # The I iteration embed all the i-th items in all batches
        embedded = [dy.lookup_batch(self.input_lookup, chars) for chars in zip(*batch)]
        state = self.rnn.initial_state()
        output_vec = state.transduce(embedded)[-1]
        w = self.W.expr(update=False)
        return w * output_vec

    # either define stop loss, or stop acc and for how many epochs acc must not fall lower than it
    def train(self, epochs=10000, stop_acc=0.99, stop_epochs=3, stop_loss=0.0005, max_epochs=1000, verbose=True):
        assert 0 < stop_acc <= 1
        print('Starting train')
        trainer = dy.AdamTrainer(self.pc)
        avg_loss = []
        num_epos_above_threshold = 0
        for i in range(epochs):
            loss_values = []
            for sequence, label in zip(self.x_train, self.y_train):
                probabilities = self._get_probabilities_over_batch(sequence)
                loss = dy.sum_batches(dy.pickneglogsoftmax_batch(probabilities, label))
                loss_values.append(loss.value())
                loss.backward()
                trainer.update()
            avg_loss.append(np.mean(loss_values))
            acc_train, acc_test = self.validate()
            if verbose:
                print(f'Epoch {i}: Accuracy {acc_train.round(5)}, Avg. Loss {avg_loss[-1].round(5)} '
                      f'Validation Accuracy {acc_test.round(5)}')

            if acc_train >= stop_acc and acc_test >= stop_acc:
                num_epos_above_threshold += 1
                if num_epos_above_threshold == stop_epochs:
                    break
            if num_epos_above_threshold > 0 and acc_train < stop_acc or acc_test < stop_acc:
                num_epos_above_threshold = 0

            if stop_loss >= avg_loss[-1] > 0:
                break
        print('Done training!')

    def validate(self):
        acc_train, acc_test = [], []
        for X, Y in zip(self.x_train, self.y_train):
            probabilities = self._get_probabilities_over_batch(X).npvalue()
            for i in range(len(probabilities[0])):
                prediction = np.argmax(probabilities[:, i])
                label = Y[i]
                if prediction == label:
                    acc_train.append(1)
                else:
                    acc_train.append(0)
        if self.x_test:
            for X, Y in zip(self.x_test, self.y_test):
                probabilities = self._get_probabilities_over_batch(X).npvalue()
                for i in range(len(probabilities[0])):
                    prediction = np.argmax(probabilities[:, i])
                    label = Y[i]
                    if prediction == label:
                        acc_test.append(1)
                    else:
                        acc_test.append(0)
        return np.mean(acc_train), np.mean(acc_test) if self.x_test else np.mean(0.0)

    def predict(self, string: str):
        w = self.W.expr(update=False)
        str_2_int = [self.token_dict[i] for i in string]
        # The I iteration embed all the i-th items in all batches
        embedded = [self.input_lookup[i] for i in str_2_int] if str_2_int else [
            self.input_lookup[self.vocab_size - 1]]
        state = self.rnn.initial_state()
        output_vec = state.transduce(embedded)[-1]
        return np.argmax((w * output_vec).npvalue())

    def step(self, inp):
        w = self.W.expr(update=False)
        str_2_int = self.token_dict[inp] if inp else max(self.token_dict.values()) + 1
        embedded = self.input_lookup[str_2_int]
        out = self.state.add_input(embedded)
        self.state = out
        return np.argmax((w * out.output()).npvalue())

    def renew(self):
        dy.renew_cg()

    def save(self, path):
        self.pc.save(path)

    def load(self, path):
        self.pc.populate(path)
