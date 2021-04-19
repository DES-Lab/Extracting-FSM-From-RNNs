from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWalkEqOracle
from aalpy.utils import visualize_automaton

from DataProcessing import generate_concrete_data_MQTT, split_train_validation, tokenized_dict
from RNNClassifier import RNNClassifier
from RNN_SULs import AbstractMQTT_RNN_SUL

train_seq, train_labels, input_al, output_al = generate_concrete_data_MQTT(num_examples=100000, num_rand_topics=2,
                                                                           lens=(1,2,3,5,8,), uniform_concretion=True)

x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

# train_seq, train_labels = generate_data_based_on_characterization_set(mealy_machine)
# x_train, y_train, x_test, y_test = split_train_validation(train_seq, train_labels, 0.8, uniform=True)

rnn = RNNClassifier(input_al, output_dim=len(output_al), num_layers=5, hidden_dim=40,
                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                    batch_size=32, nn_type='GRU')

load = True
ex_name = 'abstracted_mqtt'

if not load:
    rnn.train(epochs=100, stop_acc=1.0, stop_epochs=3)
    rnn.save(f'RNN_Models/{ex_name}.rnn')
else:
    rnn.load(f'RNN_Models/{ex_name}.rnn')

sul = AbstractMQTT_RNN_SUL(rnn, input_al, output_al)

abstract_inputs = sul.abstract_inputs

eq_oracle = RandomWalkEqOracle(abstract_inputs, sul, num_steps=10000, reset_after_cex=True)

model = run_Lstar(abstract_inputs, sul, eq_oracle, automaton_type='mealy', cache_and_non_det_check=True)

visualize_automaton(model)