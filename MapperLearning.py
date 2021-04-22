import pickle
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWalkEqOracle, StatePrefixEqOracle
from aalpy.utils import visualize_automaton

from DataProcessing import generate_concrete_data_MQTT, split_train_validation
from RNNClassifier import RNNClassifier
from RNN_SULs import AbstractMQTT_RNN_SUL

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

sul = AbstractMQTT_RNN_SUL(rnn, input_al, output_al)

abstract_inputs = sul.abstract_inputs

eq_oracle = StatePrefixEqOracle(abstract_inputs, sul, walks_per_state=100, walk_len=20)

model = run_Lstar(abstract_inputs, sul, eq_oracle, automaton_type='mealy', cache_and_non_det_check=True, suffix_closedness=False)

visualize_automaton(model)
