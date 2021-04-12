from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import visualize_automaton

from Weiss_et_al.Extraction import extract
from Weiss_et_al.GRU import GRUNetwork
from Weiss_et_al.RNNClassifier import RNNClassifier
from Weiss_et_al.Tomita_Grammars import tomita_3
from Weiss_et_al.Training_Functions import make_train_set_for_target, mixed_curriculum_train

from RNN_SULs import RNN_BinarySUL_for_Weiss_Framework

target = tomita_3
alphabet = "01"

train_set = make_train_set_for_target(target, alphabet, max_train_samples_per_length=300)

# alphabet = alphabet_bp
# train_set = get_balanced_parantheses_train_set(n=20000, short=1, longg=15)


rnn = RNNClassifier(alphabet, num_layers=2, hidden_dim=25, RNNClass=GRUNetwork)
mixed_curriculum_train(rnn, train_set, stop_threshold=0.0005)

# exit(1)
# initial examples
all_words = sorted(list(train_set.keys()), key=lambda x: len(x))
pos = next((w for w in all_words if rnn.classify_word(w) == True), None)
neg = next((w for w in all_words if rnn.classify_word(w) == False), None)
starting_examples = [w for w in [pos, neg] if not None == w]

rnn.renew()

dfa_weiss = extract(rnn, time_limit=500, initial_split_depth=10, starting_examples=starting_examples)

rnn.renew()

sul = RNN_BinarySUL_for_Weiss_Framework(rnn)

alphabet = list(alphabet)

state_eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=150, walk_len=15)

dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=state_eq_oracle, automaton_type='dfa')

visualize_automaton(dfa)
