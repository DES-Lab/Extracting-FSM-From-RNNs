import random
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


def generate_concrete_data_MQTT(num_examples, num_rand_topics=5, lens=(1, 2, 4, 6, 10), uniform_concretion=False):
    mealy_machine = load_automaton_from_file('TrainingDataAndAutomata/MQTT.dot', automaton_type='mealy')
    input_al = mealy_machine.get_input_alphabet()

    sum_lens = sum(lens)

    if uniform_concretion:
        num_examples = num_examples // num_rand_topics

    # key is length, value is number of examples for said length
    ex_per_len = {}
    for l in lens:
        # if l == 1 or l == 2:
        #     ex_per_len[l] = pow(len(input_al), l + 2)
        #     sum_lens -= l
        #     continue

        ex_per_len[l] = int(num_examples * (l / sum_lens)) + 1

    abstract_train_seq = []
    train_labels = []

    for l in ex_per_len.keys():
        for i in range(ex_per_len[l]):
            seq = [choice(input_al) for _ in range(l)]
            if i == 0 and l != 1 and random.random() >= 0.15:
                seq[0] = 'connect'

            mealy_machine.reset_to_initial()
            out = None
            for inp in seq:
                out = mealy_machine.step(inp)

            abstract_train_seq.append(seq)
            train_labels.append(out)

    random_topics = [gen_random_str() for _ in range(num_rand_topics)]
    concrete_train_seq = []
    concrete_input_set = set()
    concrete_labels = []

    for ind, seq in enumerate(abstract_train_seq):

        topics = [choice(random_topics)] if not uniform_concretion else random_topics
        for t in topics:

            topic = t
            concrete_seq = []

            for abstract_input in seq:
                if abstract_input == 'connect':
                    concrete_seq.append('connect_Client1_ping=False')
                elif abstract_input == 'disconnect':
                    concrete_seq.append(f'disconnect_Client1_var')
                elif abstract_input == 'subscribe':
                    concrete_seq.append(f'subscribe_Client1_topic="{topic}"_retain=False')
                elif abstract_input == 'unsubscribe':
                    concrete_seq.append(f'unsubscribe_Client1_topic="{topic}"')
                elif abstract_input == 'publish':
                    concrete_seq.append(f'publish_Client1_global_topic="{topic}"')
                elif abstract_input == 'invalid':
                    concrete_seq.append(f'invalid=Client1_opt=NULL')
                else:
                    assert False

            if train_labels[ind] == 'CONNACK' or train_labels[ind] == 'CONCLOSED':
                concrete_labels.append(train_labels[ind] + f'_User1')
            else:
                concrete_labels.append(train_labels[ind] + f'_User1_topic:{topic}')

            #concrete_labels.append(train_labels[ind])

            concrete_train_seq.append(concrete_seq)

            concrete_input_set.update(concrete_seq)

    concrete_input_set = list(concrete_input_set)
    output_al = list(set(concrete_labels))

    # map to integers
    input_dict = tokenized_dict(concrete_input_set)
    out_dict = tokenized_dict(output_al)

    train_seq = [seq_to_tokens(word, input_dict) for word in concrete_train_seq]
    train_labels = [seq_to_tokens(word, out_dict) for word in concrete_labels]

    return train_seq, train_labels, concrete_input_set, output_al


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
    # sequences.extend([p + e for p in sequences for e in characterization_set])
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


def gen_random_str():
    from random import choice, randint
    import string

    str_len = randint(2, 10)
    letters = string.ascii_lowercase
    return ''.join(choice(letters) for i in range(10))
