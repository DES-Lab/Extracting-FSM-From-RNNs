from aalpy.SULs import DfaSUL, MealySUL
from aalpy.automata import MealyMachine
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle, RandomWalkEqOracle
from aalpy.utils import load_automaton_from_file, save_automaton_to_file

from DataProcessing import generate_data_from_mealy, split_train_validation, tokenized_dict, tokenize
from RNNClassifier import RNNClassifier
from RNN_SULs import RnnMealySUL


def learning_based_testing_against_correct_model(model_1_path, model_2_path, cex_rounds=10,
                                                 automaton_type='mealy'):
    model_1 = load_automaton_from_file(model_1_path, compute_prefixes=True, automaton_type=automaton_type)
    model_2 = load_automaton_from_file(model_2_path, compute_prefixes=True, automaton_type=automaton_type)

    alphabet = model_1.get_input_alphabet()
    model_1_sul = DfaSUL(model_1) if automaton_type == 'dfa' else MealySUL(model_1)

    eq_oracle = StatePrefixEqOracle(alphabet, model_1_sul, walk_len=200, walks_per_state=20)
    eq_oracle = RandomWalkEqOracle(alphabet, model_1_sul, num_steps=5000, reset_prob=0.09, reset_after_cex=True)

    cex_set = set()
    for i in range(cex_rounds):
        cex = eq_oracle.find_cex(model_2)
        if cex:
            cex_set.add(tuple(cex))
        else:
            break

    return cex_set


def training_data_from_cex_set(cex_set, path_2_correct_hyp, automaton_type='mealy'):
    model_1 = load_automaton_from_file(path_2_correct_hyp, compute_prefixes=True, automaton_type=automaton_type)

    model_1_sul = DfaSUL(model_1) if automaton_type == 'dfa' else MealySUL(model_1)

    training_sequences, training_labels = [], []
    for cex in cex_set:
        output = model_1_sul.query(cex)
        training_sequences.append(cex)
        training_labels.append(output[-1])

    return training_sequences, training_labels


def train_extract_retrain(path_mealy_machine, ex_name,
                          lens=(2, 8, 10, 12, 15),
                          num_train_samples=50000, load=False, formalism='mealy', extract_automaton=False):
    assert formalism in ['mealy', 'moore']

    mealy_machine = load_automaton_from_file(path_mealy_machine, automaton_type='mealy')
    input_al = mealy_machine.get_input_alphabet()

    train_seq, train_labels = generate_data_from_mealy(mealy_machine, input_al,
                                                       num_examples=num_train_samples, lens=lens)
    x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8)

    rnn = RNNClassifier(input_al, num_layers=4, hidden_dim=50, x_train=x_train,
                        y_train=y_train, x_test=x_test, y_test=y_test, batch_size=32, nn_type="GRU")

    if not load:
        rnn.train(epochs=100, stop_acc=1.0, stop_epochs=3)
        rnn.save(f'RNN_Models/{ex_name}.rnn')
    else:
        rnn.load(f'RNN_Models/{ex_name}.rnn')

    output_al = {output for state in mealy_machine.states for output in state.output_fun.values()}
    outputs_2_ints = {integer: output for output, integer in tokenized_dict(output_al).items()}

    # TODO Change automata learning setting if you want
    sul = RnnMealySUL(rnn, outputs_2_ints)

    eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=100, walk_len=20)

    mealy = run_Lstar(alphabet=input_al, sul=sul, eq_oracle=eq_oracle, automaton_type=formalism,
                      cache_and_non_det_check=True, max_learning_rounds=10)

    save_automaton_to_file(mealy, f'LearnedAutomata/learned_{ex_name}')

    while True:
        cex_set = learning_based_testing_against_correct_model(path_mealy_machine,
                                                               f'LearnedAutomata/learned_{ex_name}.dot',
                                                               cex_rounds=100)
        if not cex_set:
            print('No counterexamples found.')
            break

        new_x, new_y = training_data_from_cex_set(cex_set, path_mealy_machine)

        new_x = tokenize(new_x, input_al)
        new_y = tokenize(new_y, output_al)

        for i in range(len(new_x)):
            x_train += tuple(new_x[i])
            y_train += tuple([new_y[i]])

        print('Counterexamples between correct model and learned model found via learning-based testing.')
        print(f"Starting retraining with {len(x_train)} new samples.")
        rnn.train(epochs=100, stop_acc=1.0, stop_epochs=3)

        sul = RnnMealySUL(rnn, outputs_2_ints)

        eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=100, walk_len=20)

        mealy = run_Lstar(alphabet=input_al, sul=sul, eq_oracle=eq_oracle, automaton_type=formalism,
                          cache_and_non_det_check=True, max_learning_rounds=10)

        save_automaton_to_file(mealy, f'LearnedAutomata/learned_{ex_name}')

if __name__ == '__main__':
    train_extract_retrain('TrainingDataAndAutomata/Coffee_machine.dot', ex_name='coffee_lbt', lens=(8, 10, 12), load=False)