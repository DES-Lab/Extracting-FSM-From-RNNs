from aalpy.SULs import DfaSUL, MealySUL, MooreSUL
from aalpy.automata import Dfa, MealyMachine
from aalpy.oracles import StatePrefixEqOracle, RandomWalkEqOracle
from aalpy.utils import load_automaton_from_file, save_automaton_to_file

from DataProcessing import generate_data_from_automaton, split_train_validation, tokenize, \
    get_coffee_machine, get_mqtt_mealy, tokenized_dict, get_tomita
from LongCexEqOracle import LongCexEqOracle
from RNNClassifier import RNNClassifier
from RNN_SULs import RnnMealySUL
from TrainAndExtract import extract_mealy_machine, train_RNN_on_mealy_data


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


def training_data_from_cex_set(ground_truth_model, cex_set):

    sul = DfaSUL if isinstance(ground_truth_model, Dfa) else \
        MealySUL if isinstance(ground_truth_model, MealyMachine) else MooreSUL
    model_1_sul = sul(ground_truth_model)

    training_sequences, training_labels = [], []
    for cex in cex_set:
        output = model_1_sul.query(cex)
        training_sequences.append(cex)
        training_labels.append(output[-1])

    return training_sequences, training_labels


def lbt_of_2_same_trainings():
    mm, exp = get_coffee_machine(), 'coffee'
    mm, exp = get_mqtt_mealy(), 'mqtt'

    input_al = mm.get_input_alphabet()
    output_al = {output for state in mm.states for output in state.output_fun.values()}

    train_seq, train_labels = generate_data_from_automaton(mm, input_al,
                                                           num_examples=10000, lens=(2, 5, 8, 10))

    training_data = (train_seq, train_labels)

    rnn_1 = train_RNN_on_mealy_data(mm, data=training_data, ex_name=f'{exp}_1')
    rnn_2 = train_RNN_on_mealy_data(mm, data=training_data, ex_name=f'{exp}_2')

    learned_automaton_1 = extract_mealy_machine(rnn_1, input_al, output_al, max_learning_rounds=25)
    learned_automaton_2 = extract_mealy_machine(rnn_2, input_al, output_al, max_learning_rounds=25)

    sul = MealySUL(learned_automaton_1)
    sul2 = MealySUL(learned_automaton_2)

    eq_oracle = LongCexEqOracle(input_al, sul, num_walks=500, min_walk_len=1, max_walk_len=30,
                                reset_after_cex=True)
    eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=100, walk_len=20)

    cex_set = set()
    for i in range(200):
        cex = eq_oracle.find_cex(learned_automaton_2)
        if cex:
            if tuple(cex) not in cex_set:
                print('--------------------------------------------------------------------------')
                print('Case of Non-Conformance between Automata: ', cex)
                print('Model 1  : ', sul.query(cex))
                print('Model 2  : ', sul2.query(cex))
            cex_set.add(tuple(cex))

    return cex_set


def retraining_based_on_ground_truth(ground_truth_model=get_coffee_machine(), num_train_samples=5000,
                                     lens=(2, 8, 10, 12, 15)):
    input_al = ground_truth_model.get_input_alphabet()

    if isinstance(ground_truth_model, MealyMachine):
        output_al = {output for state in ground_truth_model.states for output in state.output_fun.values()}
    else:
        output_al = [False, True]

    # Create initial training data
    train_seq, train_labels = generate_data_from_automaton(ground_truth_model, input_al,
                                                           num_examples=num_train_samples, lens=lens)

    # While the input-output behaviour of all trained neural networks is different
    iter = 0
    while True:
        iter += 1

        x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

        # Train all neural networks with same parameters
        rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=2, hidden_dim=40,
                            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                            batch_size=32, nn_type='GRU')

        print(f"Starting training of the neural network for the {iter} time")
        rnn.train(epochs=150, stop_acc=1.0, stop_epochs=3,verbose=0)

        # Extract automaton from trained NN
        #print("Starting extraction of automaton from Neural Network")

        # Select one automaton as a basis for conformance-checking. You can also do conformance checking with all pairs
        # of learned automata.

        outputs_2_ints = {integer: output for output, integer in tokenized_dict(output_al).items()}

        sul = RnnMealySUL(rnn, outputs_2_ints)

        # Select the eq. oracle
        eq_oracle = LongCexEqOracle(input_al, sul, num_walks=500, min_walk_len=1, max_walk_len=30,
                                    reset_after_cex=True)
        eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=100, walk_len=50)

        cex_set = set()

        # Try to find cases of non-conformance between learned automatons.
        print('Searching for counterexample.')
        for i in range(200):
            cex = eq_oracle.find_cex(ground_truth_model)
            if cex:
                cex_set.add(tuple(cex))

        # if there were no counterexamples between any learned automata, we end the procedure
        if not cex_set:
            print('No counterexamples found between extracted automaton and neural network.')
            break

        # Ask ground truth model for correct labels
        new_x, new_y = training_data_from_cex_set(ground_truth_model, cex_set)

        print(f'Adding {len(cex_set)} new examples to training data.')
        new_x = tokenize(new_x, input_al)
        new_y = tokenize(new_y, output_al)

        train_seq.extend(new_x)
        train_labels.extend(new_y)
        print(f'Size of training data: {len(train_seq)}')


def retraining_based_on_non_conformance(ground_truth_model=get_coffee_machine(), num_rnns=2, num_training_samples=5000,
                                        samples_lens=(3, 6, 9, 12)):
    assert num_rnns >= 2 and num_training_samples > 0

    input_al = ground_truth_model.get_input_alphabet()

    if isinstance(ground_truth_model, MealyMachine):
        output_al = {output for state in ground_truth_model.states for output in state.output_fun.values()}
    else:
        output_al = [False, True]

    # Create initial training data
    train_seq, train_labels = generate_data_from_automaton(ground_truth_model, input_al,
                                                           num_examples=num_training_samples, lens=samples_lens)

    # While the input-output behaviour of all trained neural networks is different
    iter = 0
    while True:
        iter += 1
        print(f'Learning/extraction round: {iter}')

        trained_networks = []

        x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

        # Train all neural networks with same parameters
        for i in range(num_rnns):
            rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=2, hidden_dim=40,
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                batch_size=32, nn_type='GRU')
            print(f'Starting training of RNN {i}')
            rnn.train(epochs=150, stop_acc=1.0, stop_epochs=3, verbose=False)
            trained_networks.append(rnn)

        learned_automatons = []

        # Extract automaton for each neural network
        for i, rnn in enumerate(trained_networks):
            print(f'Starting extraction of the automaton from RNN {i}')
            learned_automaton = extract_mealy_machine(rnn, input_al, output_al, max_learning_rounds=25, print_level=1)
            learned_automatons.append(learned_automaton)

        learned_automatons.sort(key=lambda x: len(x.states), reverse=True)

        # Select one automaton as a basis for conformance-checking. You can also do conformance checking with all pairs
        # of learned automata.

        base_sul = MealySUL(learned_automatons[0])

        # Select the eq. oracle
        eq_oracle = LongCexEqOracle(input_al, base_sul, num_walks=500, min_walk_len=1, max_walk_len=30,
                                    reset_after_cex=True)
        eq_oracle = StatePrefixEqOracle(input_al, base_sul, walks_per_state=100, walk_len=50)

        cex_set = set()

        # Try to find cases of non-conformance between learned automatons.
        for la in learned_automatons[1:]:
            for i in range(200):
                cex = eq_oracle.find_cex(la)
                if cex:
                    cex_set.add(tuple(cex))

        # if there were no counterexamples between any learned automata, we end the procedure
        if not cex_set:
            print('No counterexamples between extracted automata found.')
            break

        # Ask ground truth model for correct labels
        new_x, new_y = training_data_from_cex_set(ground_truth_model, cex_set)

        print(f'Adding {len(cex_set)} new examples to training data.')
        new_x = tokenize(new_x, input_al)
        new_y = tokenize(new_y, output_al)

        train_seq.extend(new_x)
        train_labels.extend(new_y)
        print(f'Size of training data: {len(train_seq)}')


if __name__ == '__main__':
    # a1 = load_automaton_from_file('lbt1.dot')
    # a2 = load_automaton_from_file('lbt2.dot')
    #
    dfa = get_tomita(3)
    retraining_based_on_ground_truth(dfa)

    # train_extract_retrain('TrainingDataAndAutomata/MQTT.dot', ex_name='mqtt_lbt', lens=(10, 12), load=False,
    #                      num_train_samples=5000)


