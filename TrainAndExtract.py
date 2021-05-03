import string

from aalpy.automata import MealyMachine
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle, TransitionFocusOracle
from aalpy.utils import save_automaton_to_file, visualize_automaton

from DataProcessing import parse_data, preprocess_binary_classification_data, generate_data_from_automaton, \
    split_train_validation, tokenized_dict, get_coffee_machine, get_mqtt_mealy, get_ssh, get_tcp
from RNNClassifier import RNNClassifier
from RNN_SULs import RnnBinarySUL, RnnMealySUL

tomita_dict = {tomita_grammar: f'TrainingDataAndAutomata/tomita{tomita_grammar}.txt'
               for tomita_grammar in [-3, 1, 2, 3, 4, 5, 6, 7]}


def train_RNN_on_tomita_grammar(tomita_grammar, acc_stop=1., loss_stop=0.005, train=True):
    assert tomita_grammar in [-3, 1, 2, 3, 4, 5, 6, 7]

    path = tomita_dict[tomita_grammar]

    tomita_alphabet = ["0", "1"]
    x, y = parse_data(path)
    x_train, y_train, x_test, y_test = preprocess_binary_classification_data(x, y, tomita_alphabet)

    # CHANGE PARAMETERS OF THE RNN if you want
    rnn = RNNClassifier(tomita_alphabet, output_dim=2, num_layers=2, hidden_dim=50, batch_size=18,
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, nn_type="LSTM")

    if train:
        rnn.train(stop_acc=acc_stop, stop_loss=loss_stop)
        rnn.save(f'RNN_Models/tomita{tomita_grammar}.rnn')

    return rnn


def train_and_extract_tomita(tomita_grammar, acc_stop=1., loss_stop=0.005, load=False):
    tomita_alphabet = ["0", "1"]

    if not load:
        rnn = train_RNN_on_tomita_grammar(tomita_grammar, acc_stop=acc_stop, loss_stop=loss_stop)
    else:
        rnn = train_RNN_on_tomita_grammar(tomita_grammar, train=False)
        rnn.load(f"RNN_Models/tomita_{tomita_grammar}.model")

    sul = RnnBinarySUL(rnn)
    alphabet = tomita_alphabet

    state_eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=1000, walk_len=5)

    dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=state_eq_oracle, automaton_type='dfa',
                    cache_and_non_det_check=True)

    save_automaton_to_file(dfa, f'LearnedAutomata/learned_tomita{tomita_grammar}')
    visualize_automaton(dfa)


def train_and_extract_bp(path="TrainingDataAndAutomata/balanced()_1.txt", load=False):
    bp_alphabet = list(string.ascii_lowercase + "()")

    x, y = parse_data(path)
    x_train, y_train, x_test, y_test = preprocess_binary_classification_data(x, y, bp_alphabet)

    # CHANGE PARAMETERS OF THE RNN if you want
    rnn = RNNClassifier(bp_alphabet, output_dim=2, num_layers=2, hidden_dim=50, x_train=x_train,
                        y_train=y_train, x_test=x_test, y_test=y_test, batch_size=18, nn_type="GRU")

    data_index = path[-5]
    if not load:
        rnn.train(stop_acc=1., stop_epochs=3, verbose=True)
        rnn.save(f"RNN_Models/balanced_parentheses{data_index}.rnn")
    else:
        rnn.load(f"RNN_Models/balanced_parentheses{data_index}.rnn")

    sul = RnnBinarySUL(rnn)
    alphabet = bp_alphabet

    state_eq_oracle = TransitionFocusOracle(alphabet, sul, num_random_walks=500, walk_len=30,
                                            same_state_prob=0.3)

    dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=state_eq_oracle, automaton_type='dfa',
                    cache_and_non_det_check=False, max_learning_rounds=5)

    save_automaton_to_file(dfa, f'LearnedAutomata/balanced_parentheses{data_index}')
    return dfa


def train_RNN_on_mealy_machine(mealy_machine: MealyMachine, ex_name, num_hidden_dim=2, hidden_dim_size=50,
                               nn_type='GRU',
                               batch_size=32, lens=(2, 8, 10, 12, 15), stopping_acc=1.0, num_train_samples=15000,
                               load=False):
    assert nn_type in ['GRU', 'LSTM']

    input_al = mealy_machine.get_input_alphabet()
    output_al = {output for state in mealy_machine.states for output in state.output_fun.values()}

    train_seq, train_labels = generate_data_from_automaton(mealy_machine, input_al,
                                                           num_examples=num_train_samples, lens=lens)

    x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

    # train_seq, train_labels = generate_data_based_on_characterization_set(mealy_machine)
    # x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

    rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=num_hidden_dim, hidden_dim=hidden_dim_size,
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        batch_size=batch_size, nn_type=nn_type)

    if not load:
        rnn.train(epochs=150, stop_acc=stopping_acc, stop_epochs=3)
        rnn.save(f'RNN_Models/{ex_name}.rnn')
    else:
        rnn.load(f'RNN_Models/{ex_name}.rnn')

    return rnn


def train_RNN_on_mealy_data(mealy_machine, data, ex_name, num_hidden_dim=2, hidden_dim_size=50,
                            nn_type='GRU', batch_size=32, stopping_acc=1.0, load=False):
    assert nn_type in ['GRU', 'LSTM']

    input_al = mealy_machine.get_input_alphabet()
    output_al = {output for state in mealy_machine.states for output in state.output_fun.values()}

    train_seq, train_labels = data[0], data[1]

    x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

    # train_seq, train_labels = generate_data_based_on_characterization_set(mealy_machine)
    # x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

    rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=num_hidden_dim, hidden_dim=hidden_dim_size,
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        batch_size=batch_size, nn_type=nn_type)

    if not load:
        rnn.train(epochs=150, stop_acc=stopping_acc, stop_epochs=3)
        rnn.save(f'RNN_Models/{ex_name}.rnn')
    else:
        rnn.load(f'RNN_Models/{ex_name}.rnn')

    return rnn


def extract_finite_state_transducer(rnn, input_alphabet, output_al, max_learning_rounds=10, formalism='mealy', print_level=2):
    assert formalism in ['mealy', 'moore']

    outputs_2_ints = {integer: output for output, integer in tokenized_dict(output_al).items()}

    sul = RnnMealySUL(rnn, outputs_2_ints)

    eq_oracle = StatePrefixEqOracle(input_alphabet, sul, walks_per_state=150, walk_len=25)

    learned_automaton = run_Lstar(alphabet=input_alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type=formalism,
                                  cache_and_non_det_check=False, max_learning_rounds=max_learning_rounds,
                                  suffix_closedness=False, print_level=print_level)

    return learned_automaton


def train_RNN_and_extract_FSM(ex_name, automaton_type = 'mealy'):
    mealy_dict = {'coffee': get_coffee_machine(), 'mqtt': get_mqtt_mealy(),
                  'tcp': get_tcp(), 'ssh': get_ssh()}

    rnn = train_RNN_on_mealy_machine(mealy_machine=mealy_dict[ex_name], ex_name=ex_name)

    input_al = mealy_dict[ex_name].get_input_alphabet()
    output_al = {output for state in mealy_dict[ex_name].states for output in state.output_fun.values()}

    learned_automaton = extract_finite_state_transducer(rnn, input_al, output_al, formalism=automaton_type)

    return learned_automaton


if __name__ == '__main__':
    # Train on a any of the 7 Tomita grammars and extract the DFA from the trained RNN
    tomita_dfa = train_and_extract_tomita(3, load=False)

    # Train on a Balanced Parentheses Dataset and extract the DFA
    bp_dfa = train_and_extract_bp()

    # Extract Mealy and Moore machine from RNN trained on the coffee machine FMS
    coffee_machine_Mealy_model = train_RNN_and_extract_FSM('coffee')
    coffee_machine_Moore_model = train_RNN_and_extract_FSM('coffee', automaton_type='moore')


