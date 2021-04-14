from aalpy.SULs import DfaSUL, MealySUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle, RandomWalkEqOracle
from aalpy.utils import load_automaton_from_file, save_automaton_to_file

from DataProcessing import generate_data_from_mealy, split_train_validation, tokenized_dict, tokenize, \
    get_coffee_machine, get_tcp, get_ssh, get_mqtt_mealy
from LongCexEqOracle import LongCexEqOracle
from RNNClassifier import RNNClassifier
from RNN_SULs import RnnMealySUL
from TrainAndExtract import train_RNN_on_mealy_machine, extract_mealy_machine


def learning_based_testing_against_correct_model(model_correct_path, model_learned_path, cex_rounds=10,
                                                 automaton_type='mealy'):
    model_correct = load_automaton_from_file(model_correct_path, compute_prefixes=True, automaton_type=automaton_type)
    model_learned = load_automaton_from_file(model_learned_path, compute_prefixes=True, automaton_type=automaton_type)

    alphabet = model_correct.get_input_alphabet()
    model_1_sul = DfaSUL(model_learned) if automaton_type == 'dfa' else MealySUL(model_learned)

    eq_oracle = RandomWalkEqOracle(alphabet, model_1_sul, num_steps=5000, reset_prob=0.09, reset_after_cex=True)
    eq_oracle = LongCexEqOracle(alphabet, model_1_sul, num_walks=500, min_walk_len=1, max_walk_len=20,
                                reset_after_cex=True)
    eq_oracle = StatePrefixEqOracle(alphabet, model_1_sul, walks_per_state=100, walk_len=20)

    cex_set = set()
    for i in range(cex_rounds):
        cex = eq_oracle.find_cex(model_correct)
        if cex:
            cex_set.add(tuple(cex))

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
                          num_train_samples=50000, load=False, formalism='mealy'):
    assert formalism in ['mealy', 'moore']

    mealy_machine = load_automaton_from_file(path_mealy_machine, automaton_type='mealy')
    input_al = mealy_machine.get_input_alphabet()
    output_al = {output for state in mealy_machine.states for output in state.output_fun.values()}

    train_seq, train_labels = generate_data_from_mealy(mealy_machine, input_al,
                                                       num_examples=num_train_samples, lens=lens)
    x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

    rnn = RNNClassifier(input_al, num_layers=4, output_dim=len(output_al), hidden_dim=50, x_train=x_train,
                        y_train=y_train, x_test=x_test, y_test=y_test, batch_size=32, nn_type="GRU")

    if not load:
        rnn.train(epochs=100, stop_acc=0.99, stop_epochs=3)
        rnn.save(f'RNN_Models/{ex_name}.rnn')
    else:
        rnn.load(f'RNN_Models/{ex_name}.rnn')

    outputs_2_ints = {integer: output for output, integer in tokenized_dict(output_al).items()}

    # TODO Change automata learning setting if you want
    sul = RnnMealySUL(rnn, outputs_2_ints)

    eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=250, walk_len=15)

    mealy = run_Lstar(alphabet=input_al, sul=sul, eq_oracle=eq_oracle, automaton_type=formalism,
                      cache_and_non_det_check=False, max_learning_rounds=10, suffix_closedness=False)

    save_automaton_to_file(mealy, f'LearnedAutomata/learned_{ex_name}')

    while True:
        cex_set = learning_based_testing_against_correct_model(path_mealy_machine,
                                                               f'LearnedAutomata/learned_{ex_name}.dot',
                                                               cex_rounds=1000)
        if not cex_set:
            print('No counterexamples found.')
            break

        new_x, new_y = training_data_from_cex_set(cex_set, path_mealy_machine)

        new_x = tokenize(new_x, input_al)
        new_y = tokenize(new_y, output_al)

        train_seq.extend(new_x)
        train_labels.extend(new_y)
        # split new data in train and validation
        x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=False)

        # # add new data next to old training/validation data
        # for i in range(len(new_x_train)): # TODO REFACTOR ADDING
        #     x_train += tuple([new_x_train[i]])
        #     y_train += tuple([[new_y_train[i]]])
        # for i in range(len(new_x_test)):
        #     x_test += tuple([new_x_test[i]])
        #     y_test += tuple([[new_y_test[i]]])

        rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=4, hidden_dim=50, x_train=x_train,
                            y_train=y_train, x_test=x_test, y_test=y_test, batch_size=32, nn_type="GRU")
        # rnn.load(f'RNN_Models/{ex_name}.rnn')

        print('Counterexamples between correct model and learned model found via learning-based testing.')
        print(f"Starting retraining with {len(cex_set)} new samples.")
        print(f'Total number of samples: {len(x_train) + len(x_test)}')
        rnn.train(epochs=200, stop_acc=1.0, stop_epochs=3)

        rnn.save(f'RNN_Models/{ex_name}.rnn')

        sul = RnnMealySUL(rnn, outputs_2_ints)

        eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=250, walk_len=15)

        mealy = run_Lstar(alphabet=input_al, sul=sul, eq_oracle=eq_oracle, automaton_type=formalism,
                          cache_and_non_det_check=False, max_learning_rounds=5, suffix_closedness=False)

        save_automaton_to_file(mealy, f'LearnedAutomata/learned_{ex_name}')


def lbt_of_2_same_trainings():
    mm, exp = get_mqtt_mealy(), 'mqtt'
    mm, exp = get_coffee_machine(), 'coffee'

    input_al = mm.get_input_alphabet()
    output_al = {output for state in mm.states for output in state.output_fun.values()}

    rnn_1 = train_RNN_on_mealy_machine(mm, ex_name=f'{exp}_1', num_train_samples=10000, lens=(5, 8, 10))
    rnn_2 = train_RNN_on_mealy_machine(mm, ex_name=f'{exp}_2', num_train_samples=5000, lens=(1, 3, 5, 8))

    learned_automaton_1 = extract_mealy_machine(rnn_1, input_al, output_al, max_learning_rounds=50)
    learned_automaton_2 = extract_mealy_machine(rnn_2, input_al, output_al, max_learning_rounds=50)

    sul = MealySUL(learned_automaton_1)

    eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=100, walk_len=20)
    eq_oracle = LongCexEqOracle(input_al, sul, num_walks=500, min_walk_len=1, max_walk_len=30,
                                reset_after_cex=True)

    cex_set = set()
    for i in range(200):
        cex = eq_oracle.find_cex(learned_automaton_2)
        if cex:
            if tuple(cex) not in cex_set:
                print('Cex Found: ', cex)
            cex_set.add(tuple(cex))

    return cex_set

if __name__ == '__main__':

    # a1 = load_automaton_from_file('lbt1.dot')
    # a2 = load_automaton_from_file('lbt2.dot')
    #
    lbt_of_2_same_trainings()

    #train_extract_retrain('TrainingDataAndAutomata/MQTT.dot', ex_name='mqtt_lbt', lens=(10, 12), load=False,
    #                      num_train_samples=5000)
