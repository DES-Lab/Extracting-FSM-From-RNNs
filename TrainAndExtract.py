import string

from aalpy.SULs import DfaSUL, MealySUL
from aalpy.automata import MealyMachine
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle, TransitionFocusOracle, RandomWalkEqOracle
from aalpy.utils import save_automaton_to_file, visualize_automaton, load_automaton_from_file

from DataProcessing import parse_data, preprocess_binary_classification_data, generate_data_from_mealy, \
    split_train_validation, get_mqtt_mealy, get_coffee_machine, tokenized_dict, \
    generate_data_based_on_characterization_set
from LearningBasedTesting import learning_based_testing_against_correct_model, training_data_from_cex_set
from RNNClassifier import RNNClassifier
from RNN_SULs import RnnBinarySUL, RnnMealySUL

tomita_dict = {l: f'TrainingDataAndAutomata/tomita{l}.txt' for l in [-3, 1, 2, 3, 4, 5, 6, 7]}


def train_RNN_on_tomita_grammar(tomita_grammar, acc_stop=1., loss_stop=0.005, train=True):
    assert tomita_grammar in [-3, 1, 2, 3, 4, 5, 6, 7]

    path = tomita_dict[tomita_grammar]

    tomita_alphabet = ["0", "1"]
    x, y = parse_data(path)
    x_train, y_train, x_test, y_test = preprocess_binary_classification_data(x, y, tomita_alphabet)

    # TODO CHANGE PARAMETERS OF THE RNN if you want
    rnn = RNNClassifier(tomita_alphabet, num_layers=2, hidden_dim=50, batch_size=18,
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
        rnn.load(f"RNN_Models/tomita{tomita_grammar}.model")

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

    # TODO CHANGE PARAMETERS OF THE RNN if you want
    rnn = RNNClassifier(bp_alphabet, num_layers=4, hidden_dim=50, x_train=x_train,
                        y_train=y_train,
                        x_test=x_test, y_test=y_test, batch_size=18, nn_type="GRU")

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
                    cache_and_non_det_check=False, max_learning_rounds=4)

    save_automaton_to_file(dfa, f'LearnedAutomata/balanced_parentheses{data_index}')
    visualize_automaton(dfa)
    return dfa


def train_and_extract_mealy(mealy_machine: MealyMachine, ex_name,
                            lens=(2,8,10,12,15),
                            num_train_samples=100000, load=False, formalism='mealy', extract_automaton = False):
    assert formalism in ['mealy', 'moore']

    input_al = mealy_machine.get_input_alphabet()

    train_seq, train_labels = generate_data_from_mealy(mealy_machine, input_al,
                                                       num_examples=num_train_samples, lens=lens)

    #train_seq, train_labels = generate_data_based_on_characterization_set(mealy_machine)

    x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8)
    #x_train, y_train, x_test, y_test = train_seq, train_labels, None, None

    rnn = RNNClassifier(input_al, num_layers=4, hidden_dim=50, x_train=x_train,
                        y_train=y_train, x_test=x_test, y_test=y_test, batch_size=32, nn_type="GRU")

    if not load:
        rnn.train(epochs=1000, stop_acc=1.0, stop_epochs=3)
        rnn.save(f'RNN_Models/{ex_name}.rnn')
    else:
        rnn.load(f'RNN_Models/{ex_name}.rnn')

    if not extract_automaton:
        return None

    output_al = {output for state in mealy_machine.states for output in state.output_fun.values()}
    outputs_2_ints = {integer: output for output, integer in tokenized_dict(output_al).items()}

    # TODO Change automata learning setting if you want
    sul = RnnMealySUL(rnn, outputs_2_ints)

    eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=100, walk_len=20)

    print('start learning')
    mealy = run_Lstar(alphabet=input_al, sul=sul, eq_oracle=eq_oracle, automaton_type=formalism,
                      cache_and_non_det_check=True, max_learning_rounds=10)

    save_automaton_to_file(mealy, f'LearnedAutomata/learned_{ex_name}')
    visualize_automaton(mealy)
    return mealy_machine


if __name__ == '__main__':
    # train_and_extract_tomita(tomita_grammar=3, load=False)

    # learning based testing with coffee machine, RNN models saved to coffee_1, and coffee_2
    # coffee_1 -> lens=(3,5,7,10,12), num_train_samples=50000
    # coffee_2 -> lens=(2,8,10,12,15), num_train_samples=50000

    tcp = load_automaton_from_file('TrainingDataAndAutomata/TCP_Linux_Client.dot', 'mealy')
    train_and_extract_mealy(tcp, ex_name='tcp', lens=(1, 3, 5, 8, 10, 12, 15, 20), load=False, extract_automaton = False)

    # cex_set = learning_based_testing_against_correct_model('TrainingDataAndAutomata/Coffee_machine.dot', 'LearnedAutomata/learned_coffee_2.dot', cex_rounds=10)
    # print(cex_set)
    # training_x, training_y = training_data_from_cex_set(cex_set, 'TrainingDataAndAutomata/Coffee_machine.dot')
    # print(training_x, training_y)