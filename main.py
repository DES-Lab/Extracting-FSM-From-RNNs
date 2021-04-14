from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import save_automaton_to_file, visualize_automaton

from RNN_SULs import RnnBinarySUL
from TrainAndExtract import train_RNN_on_tomita_grammar, train_and_extract_bp, learn_and_extract_mealy

# learn and extract tomita 3 grammar.
# same can be achieved with train_and_extract_tomita function
rnn = train_RNN_on_tomita_grammar(tomita_grammar=3, acc_stop=1., loss_stop=0.005, train=True)

tomita_alphabet = ["0", "1"]

sul = RnnBinarySUL(rnn)
alphabet = tomita_alphabet

state_eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=200, walk_len=6)

dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=state_eq_oracle, automaton_type='dfa',
                cache_and_non_det_check=True)

save_automaton_to_file(dfa, f'RNN_Models/tomita{3}')
visualize_automaton(dfa)

# train and extract balanced parentheses
train_and_extract_bp(path='TrainingDataAndAutomata/balanced()_2.txt', load=False)

# train and learn mealy machine example
coffee_machine_automaton = learn_and_extract_mealy('coffee')
mqtt_automaton = learn_and_extract_mealy('mqtt')
