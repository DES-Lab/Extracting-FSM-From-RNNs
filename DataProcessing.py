from random import choice, shuffle

from aalpy.utils import load_automaton_from_file


def parse_data(path: str):
    x, y = [], []
    file = open(path, 'r')
    lines = file.readlines()
    for l in lines:
        key_val_pair = l.rstrip().split(':')
        x.append(list(key_val_pair[0]))
        y.append(key_val_pair[1])
    file.close()
    return x, y


def preprocess_binary_classification_data(x, y, alphabet):
    char_dict = tokenized_dict(alphabet)
    tokenized_x = [seq_to_tokens(word, char_dict) for word in x]
    tokenized_y = [1 if i == 'True' else 0 for i in y]

    x_train, y_train, x_test, y_test = split_train_validation(tokenized_x, tokenized_y)

    return x_train, y_train, x_test, y_test


def split_train_validation(x_values, y_values, ratio=0.8, uniform=False):
    num_cat = len(set(y_values))
    x_train, y_train, x_test, y_test = None, None, None, None
    target_cat = 0
    # both train and test should have all labels
    while True:
        c = list(zip(x_values, y_values))
        shuffle(c)
        x_values, y_values = zip(*c)
        cutoff = int((len(x_values) + 1) * ratio)
        x_train, x_test = x_values[:cutoff], x_values[cutoff:]
        y_train, y_test = y_values[:cutoff], y_values[cutoff:]
        if not uniform and len(set(y_train)) == num_cat:  # and len(set(y_test)) == num_cat:
            break
        elif len(set(y_train)) == num_cat and len(set(y_test)) == num_cat:
            break

    return x_train, y_train, x_test, y_test


def tokenized_dict(alphabet):
    return dict((c, i) for i, c in enumerate(alphabet))


def seq_to_tokens(word, lookup_dict: dict):
    return [lookup_dict[letter] for letter in word] if isinstance(word, (list, tuple)) else lookup_dict[
        word] if word else []


def get_mqtt_mealy():
    return load_automaton_from_file('TrainingDataAndAutomata/MQTT.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def get_coffee_machine():
    return load_automaton_from_file('TrainingDataAndAutomata/Coffee_machine.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def get_tcp():
    return load_automaton_from_file('TrainingDataAndAutomata/TCP_Linux_Client.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def get_ssh():
    return load_automaton_from_file('TrainingDataAndAutomata/OpenSSH.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def generate_data_from_mealy(mealy_machine, input_al, num_examples, lens=(1, 2, 4, 6, 10, 15, 20)):
    output_al = {output for state in mealy_machine.states for output in state.output_fun.values()}
    ex_per_len = (num_examples // len(lens)) + 1

    sum_lens = sum(lens)
    # key is length, value is number of examples for said length
    ex_per_len = {}
    for l in lens:
        ex_per_len[l] = int(num_examples * (l / sum_lens)) + 1

    train_seq = []
    train_labels = []
    for l in ex_per_len.keys():
        for _ in range(ex_per_len[l]):
            seq = [choice(input_al) for _ in range(l)]

            mealy_machine.reset_to_initial()
            out = None
            for inp in seq:
                out = mealy_machine.step(inp)

            train_seq.append(seq)
            train_labels.append(out)

    # map to integers
    input_dict = tokenized_dict(input_al)
    out_dict = tokenized_dict(output_al)

    train_seq = [seq_to_tokens(word, input_dict) for word in train_seq]
    train_labels = [seq_to_tokens(word, out_dict) for word in train_labels]
    return train_seq, train_labels


def tokenize(seq, alphabet):
    dictionary = tokenized_dict(alphabet)
    return [seq_to_tokens(word, dictionary) for word in seq]


def generate_data_based_on_characterization_set(automaton, automaton_type='mealy'):
    from aalpy.SULs import MealySUL, DfaSUL
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.learning_algs import run_Lstar

    # automaton = load_automaton_from_file(path_to_automaton, automaton_type)
    alphabet = automaton.get_input_alphabet()
    eq_oracle = RandomWalkEqOracle(alphabet, automaton, num_steps=5000, reset_prob=0.09, reset_after_cex=True)

    sul = DfaSUL(automaton) if automaton_type == 'dfa' else MealySUL(automaton)

    automaton, data = run_Lstar(alphabet, sul, eq_oracle, automaton_type=automaton_type,
                                print_level=0, return_data=True, suffix_closedness=True)

    characterization_set = data['characterization set']
    prefixes = [state.prefix for state in automaton.states]

    sequences = [p + e for e in characterization_set for p in prefixes]

    sequences.extend([p + tuple([i]) + e for p in prefixes for i in automaton.get_input_alphabet()
                      for e in characterization_set])
    #sequences.extend([p + e for p in sequences for e in characterization_set])
    for _ in range(1):
        sequences.extend([p + tuple([i]) + e for p in sequences for i in automaton.get_input_alphabet()
                          for e in characterization_set])
    for _ in range(3):
        sequences.extend(sequences)

    labels = [sul.query(s)[-1] for s in sequences]

    sequences = [list(s) for s in sequences]

    input_al = automaton.get_input_alphabet()
    output_al = {output for state in automaton.states for output in state.output_fun.values()}

    input_dict = tokenized_dict(input_al)
    out_dict = tokenized_dict(output_al)

    train_seq = [seq_to_tokens(word, input_dict) for word in sequences]
    train_labels = [seq_to_tokens(word, out_dict) for word in labels]

    return train_seq, train_labels
