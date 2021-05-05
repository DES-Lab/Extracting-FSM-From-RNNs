import pickle

from aalpy.SULs import DfaSUL, MealySUL, MooreSUL
from aalpy.automata import Dfa, MealyMachine
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import visualize_automaton

from DataProcessing import generate_data_from_automaton, split_train_validation, tokenize, \
    get_coffee_machine, get_mqtt_mealy, tokenized_dict, generate_concrete_data_MQTT
from LongCexEqOracle import LongCexEqOracle
from RNNClassifier import RNNClassifier
from RNN_SULs import RnnMealySUL, Abstract_Mapper_MQTT_RNN_SUL
from TrainAndExtract import extract_finite_state_transducer, train_RNN_on_mealy_data


def learn_with_mapper():
    train_seq, train_labels, input_al, output_al = generate_concrete_data_MQTT(num_examples=300000, num_rand_topics=2,
                                                                               lens=(1, 2, 3, 5, 8, 10, 12),
                                                                               uniform_concretion=True)

    x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

    # train_seq, train_labels = generate_data_based_on_characterization_set(mealy_machine)
    # x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

    rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=5, hidden_dim=40,
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        batch_size=32, nn_type='GRU')

    load = True
    ex_name = 'abstracted_mqtt'

    if not load:
        rnn.train(epochs=200, stop_acc=1.0, stop_epochs=3)
        rnn.save(f'RNN_Models/{ex_name}.rnn')
        with open(f'RNN_Models/{ex_name}.pickle', 'wb') as handle:
            pickle.dump((input_al, output_al), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        rnn.load(f'RNN_Models/{ex_name}.rnn')
        with open(f'RNN_Models/{ex_name}.pickle', 'rb') as handle:
            inp_out_tuple = pickle.load(handle)
        input_al, output_al = inp_out_tuple[0], inp_out_tuple[1]
        rnn.token_dict = dict((c, i) for i, c in enumerate(input_al))

    sul = Abstract_Mapper_MQTT_RNN_SUL(rnn, input_al, output_al)

    abstract_inputs = sul.abstract_inputs

    eq_oracle = StatePrefixEqOracle(abstract_inputs, sul, walks_per_state=100, walk_len=20)

    model = run_Lstar(abstract_inputs, sul, eq_oracle, automaton_type='mealy', cache_and_non_det_check=True,
                      suffix_closedness=False)

    visualize_automaton(model)
    return model


def label_sequences_with_correct_model(ground_truth_model, cex_set):
    """

    :param ground_truth_model: correct model
    :param cex_set: sequences to label
    :return: list of sequences and list of their labels
    """
    sul = DfaSUL if isinstance(ground_truth_model, Dfa) else \
        MealySUL if isinstance(ground_truth_model, MealyMachine) else MooreSUL
    model_1_sul = sul(ground_truth_model)

    training_sequences, training_labels = [], []
    for cex in cex_set:
        output = model_1_sul.query(cex)
        training_sequences.append(cex)
        training_labels.append(output[-1])

    return training_sequences, training_labels


def conformance_check_2_RNNs(experiment='coffee'):
    """
    Show how learning based testing can find differences between 2 trained RNNs.
    RNNs are have the same configuration, but it can be different.
    :param experiment: either coffee of mqtt
    :return: cases of non-conformance between trained RNNs
    """
    if experiment == 'coffee':
        mm, exp = get_coffee_machine(), experiment
    else:
        mm, exp = get_mqtt_mealy(), experiment

    input_al = mm.get_input_alphabet()
    output_al = {output for state in mm.states for output in state.output_fun.values()}

    train_seq, train_labels = generate_data_from_automaton(mm, input_al,
                                                           num_examples=10000, lens=(2, 5, 8, 10))

    training_data = (train_seq, train_labels)

    rnn_1 = train_RNN_on_mealy_data(mm, data=training_data, ex_name=f'{exp}_1')
    rnn_2 = train_RNN_on_mealy_data(mm, data=training_data, ex_name=f'{exp}_2')

    learned_automaton_1 = extract_finite_state_transducer(rnn_1, input_al, output_al, max_learning_rounds=25)
    learned_automaton_2 = extract_finite_state_transducer(rnn_2, input_al, output_al, max_learning_rounds=25)

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
                                     lens=(3, 8, 10, 12, 15)):
    """
    :param ground_truth_model: correct model used for data generation and confromance checking
    :param num_train_samples: num of training samples for the initial data generation
    :param lens: lengths of counterexample
    :return: trained RNN that conforms to the ground truth model
    """
    input_al = ground_truth_model.get_input_alphabet()

    if isinstance(ground_truth_model, MealyMachine):
        output_al = {output for state in ground_truth_model.states for output in state.output_fun.values()}
    else:
        output_al = [False, True]

    # Create initial training data
    train_seq, train_labels = generate_data_from_automaton(ground_truth_model, input_al,
                                                           num_examples=num_train_samples, lens=lens)

    # While the input-output behaviour of all trained neural networks is different
    iteration = 0
    while True:
        iteration += 1

        # split dataset into training and verification
        x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

        # Train all neural networks with same parameters, this can be configured to train with different parameters
        rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=2, hidden_dim=40,
                            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                            batch_size=32, nn_type='GRU')

        print(f"Starting training of the neural network for the {iteration} time")
        # Train the NN
        rnn.train(epochs=150, stop_acc=1.0, stop_epochs=3, verbose=0)

        # encode outputs
        outputs_2_ints = {integer: output for output, integer in tokenized_dict(output_al).items()}

        # use RNN as SUL
        sul = RnnMealySUL(rnn, outputs_2_ints)

        # Select the eq. oracle
        eq_oracle = LongCexEqOracle(input_al, sul, num_walks=500, min_walk_len=1, max_walk_len=30,
                                    reset_after_cex=True)
        eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=200, walk_len=20)

        cex_set = set()

        # Try to find cases of non-conformance between learned automatons.
        print('Searching for counterexample.')
        for i in range(200):
            # Conformance check ground truth model and trained RNN
            # Alternatively, one can extract automaton from RNN and then model check against GT
            cex = eq_oracle.find_cex(ground_truth_model)
            if cex:
                cex_set.add(tuple(cex))

        # if there were no counterexamples between any learned automata, we end the procedure
        if not cex_set:
            print('No counterexamples found between extracted automaton and neural network.')
            # Extract automaton from rnn and print it

            final_model = run_Lstar(input_al, sul, eq_oracle, automaton_type='mealy', max_learning_rounds=15)
            print(final_model)
            return rnn

        # Ask ground truth model for correct labels
        new_x, new_y = label_sequences_with_correct_model(ground_truth_model, cex_set)

        print(f'Adding {len(cex_set)} new examples to training data.')
        new_x = tokenize(new_x, input_al)
        new_y = tokenize(new_y, output_al)

        train_seq.extend(new_x)
        train_labels.extend(new_y)
        print(f'Size of training data: {len(train_seq)}')


def retraining_based_on_non_conformance(ground_truth_model=get_coffee_machine(), num_rnns=2, num_training_samples=5000,
                                        samples_lens=(3, 6, 9, 12)):
    """

    :param ground_truth_model: correct model used for labeling cases of non-conformance
    :param num_rnns: number of RNN to be trained and learned
    :param num_training_samples: initial number of training samples in the training data set
    :param samples_lens: lengths of initial training data set samples
    :return: one RNN obtained after active retraining
    """
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
    iteration = 0
    while True:
        iteration += 1
        print(f'Learning/extraction round: {iteration}')

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
            learned_automaton = extract_finite_state_transducer(rnn, input_al, output_al, max_learning_rounds=8 , print_level=0)
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

        # If there were no counterexamples between any learned automata, we end the procedure
        if not cex_set:
            for i, la in enumerate(learned_automatons):
                print(f'Size of automata {i}: {len(la.states)}')
            print(learned_automatons[-1])
            print('No counterexamples between extracted automata found.')
            break

        # Ask ground truth model for correct labels
        new_x, new_y = label_sequences_with_correct_model(ground_truth_model, cex_set)

        print(f'Adding {len(cex_set)} new examples to training data.')
        new_x = tokenize(new_x, input_al)
        new_y = tokenize(new_y, output_al)

        train_seq.extend(new_x)
        train_labels.extend(new_y)
        print(f'Size of training data: {len(train_seq)}')


if __name__ == '__main__':
    retraining_based_on_non_conformance(get_mqtt_mealy(), num_rnns=4, num_training_samples=1000, samples_lens=(3,6,9))
    exit()
    # Find differences between 2 trained RNNs
    conformance_check_2_RNNs()

    # Retrain RNN based on non-conformance between the trained RNN and the ground truth/correct model
    retraining_based_on_ground_truth()

    # Retrain RNN based on non-conformance between several trained RNN with minimal querying of the ground
    # truth/correct model
    retraining_based_on_non_conformance(get_mqtt_mealy(), num_rnns=4, num_training_samples=1000)

    # Learn the abstract model of the RNN trained on the concrete samples
    learn_with_mapper()
