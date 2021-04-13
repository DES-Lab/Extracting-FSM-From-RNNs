import string

from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import visualize_automaton

from RNN_SULs import RNN_BinarySUL_for_Weiss_Framework
from Weiss_et_al.Extraction import extract
from Weiss_et_al.GRU import GRUNetwork
from Weiss_et_al.RNNClassifier import RNNClassifier
from Weiss_et_al.Specific_Language_Generation import get_balanced_parantheses_train_set
from Weiss_et_al.Tomita_Grammars import tomita_dicts
from Weiss_et_al.Training_Functions import make_train_set_for_target, mixed_curriculum_train

examples = ['tomita_1', 'tomita_2', 'tomita_3', 'tomita_4', 'tomita_5', 'tomita_6', 'tomita_7',
            'balanced_parenthesis_1',
            'balanced_parenthesis_2']
models = {example: f'RNN_Models/Weiss_Framework_{example}.rnn' for example in examples}


def train_or_load_rnn(example, train=False):
    alphabet = "01" if 'tomita' in example else list(string.ascii_lowercase + "()")

    rnn = RNNClassifier(alphabet, num_layers=2, hidden_dim=25, RNNClass=GRUNetwork)
    if 'tomita' in example:
        target_fun = tomita_dicts[example]
        train_set = make_train_set_for_target(target_fun, alphabet, max_train_samples_per_length=300)
    else:
        train_set = get_balanced_parantheses_train_set(n=20000, short=1, longg=15)

    if train:
        mixed_curriculum_train(rnn, train_set, stop_threshold=0.0005)
        rnn.save(example)
    else:
        assert example in models.keys()
        rnn.load(models[example])

    return rnn, alphabet, train_set

# TODO Investigate: TOMITA 5 could not be trained, some error is tron
# TODO for tomita 3, she had 4 states, I had 5
rnn, alphabet, train_set = train_or_load_rnn('tomita_3', train=True)
#exit(1)
# initial examples for Weiss et Al
all_words = sorted(list(train_set.keys()), key=lambda x: len(x))
pos = next((w for w in all_words if rnn.classify_word(w) is True), None)
neg = next((w for w in all_words if rnn.classify_word(w) is False), None)
starting_examples = [w for w in [pos, neg] if None is not w]

rnn.renew()

dfa_weiss = extract(rnn, time_limit=500, initial_split_depth=10, starting_examples=starting_examples)

rnn.renew()

sul = RNN_BinarySUL_for_Weiss_Framework(rnn)

alphabet = list(alphabet)

state_eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=150, walk_len=8)

dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=state_eq_oracle, automaton_type='dfa')

visualize_automaton(dfa)
