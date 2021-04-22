import string

from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle, TransitionFocusOracle
from aalpy.utils import visualize_automaton

from RNN_SULs import RNN_BinarySUL_for_Weiss_Framework
from Weiss_et_al.Extraction import extract
from Weiss_et_al.GRU import GRUNetwork
from Weiss_et_al.LSTM import LSTMNetwork
from Weiss_et_al.RNNClassifier import RNNClassifier
from Weiss_et_al.Specific_Language_Generation import get_balanced_parantheses_train_set
from Weiss_et_al.Tomita_Grammars import tomita_dicts
from Weiss_et_al.Training_Functions import make_train_set_for_target, mixed_curriculum_train


def train_or_load_rnn(example, num_layers=2, hidden_dim=50, rnn_class=GRUNetwork, train=True):
    alphabet = "01" if 'tomita' in example else list(string.ascii_lowercase + "()")

    rnn = RNNClassifier(alphabet, num_layers=num_layers, hidden_dim=hidden_dim, RNNClass=rnn_class)

    nn_props = F'{"GRU" if rnn_class == GRUNetwork else "LSTM"}_layers_{num_layers}_dim_{hidden_dim}'
    if 'tomita' in example:
        target_fun = tomita_dicts[example]
        lengths = (1, 2, 3, 4, 5, 6, 7, 8) if example == 'tomita_6' or example == 'tomita_5' else None
        train_set = make_train_set_for_target(target_fun, alphabet, max_train_samples_per_length=500,
                                              search_size_per_length=2000, lengths=lengths)
    else:
        train_set = get_balanced_parantheses_train_set(n=20000, short=1, longg=15)

    if train:
        mixed_curriculum_train(rnn, train_set, stop_threshold=0.0005)
        rnn.save(f'RNN_Models/WeissComparisonModels/{example}_{nn_props}.rnn')
        print('saving to', f'RNN_Models/WeissComparisonModels/{example}_{nn_props}.rnn')
    else:
        # assert example in models.keys()
        rnn.load(f'RNN_Models/WeissComparisonModels/{example}_{nn_props}.rnn')

    return rnn, alphabet, train_set


def run_comparison(example, train=True, num_layers=2, hidden_dim=50, rnn_class=GRUNetwork):

    rnn, alphabet, train_set = train_or_load_rnn(example, num_layers=num_layers, hidden_dim=hidden_dim,
                                                 rnn_class=rnn_class, train=train)

    # initial examples for Weiss et Al
    all_words = sorted(list(train_set.keys()), key=lambda x: len(x))
    pos = next((w for w in all_words if rnn.classify_word(w) is True), None)
    neg = next((w for w in all_words if rnn.classify_word(w) is False), None)
    starting_examples = [w for w in [pos, neg] if None is not w]

    # Extract Automaton Using White-Box eq. query
    rnn.renew()
    print('---------------------------------WHITE BOX EXTRACTION--------------------------------------------------')
    dfa_weiss = extract(rnn, time_limit=500, initial_split_depth=10, starting_examples=starting_examples)

    # Make sure that internal states are back to initial
    rnn.renew()

    print('---------------------------------BLACK BOX EXTRACTION--------------------------------------------------')
    sul = RNN_BinarySUL_for_Weiss_Framework(rnn)

    alphabet = list(alphabet)

    eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=150, walk_len=8)
    if 'tomita' not in example:
        eq_oracle = TransitionFocusOracle(alphabet, sul, num_random_walks=200, walk_len=15)

    dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='dfa', max_learning_rounds=10,
                    print_level=1)

    if len(dfa.states) != len(dfa_weiss.Q):
        print('---------------------------------WHITE vs. BLACK BOX EXTRACTION----------------------------------------')
        nn_props = F'{"GRU" if rnn_class == GRUNetwork else "LSTM"}_layers_{num_layers}_dim_{hidden_dim}'
        print(f'Example       : {example}')
        print(f'Configuration : {nn_props}')
        print(f"Number of states\n  White-box extraction: {len(dfa_weiss.Q)}\n  Black-box extraction: {len(dfa.states)}")

if __name__ == '__main__':

    # Run extraction on all pre-trained tomita examles
    for tomita_ex in tomita_dicts.keys():
        for nn in [GRUNetwork, LSTMNetwork]:
            run_comparison(tomita_ex, rnn_class=nn, train=False)

    # Run extraction on all pre-trained bp examples
    for i in range(1,3):
        for nn in [GRUNetwork, LSTMNetwork]:
            run_comparison(f'bp_{i}', rnn_class=nn, train=False)