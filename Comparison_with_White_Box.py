import os
import string
import sys
import time

from aalpy.SULs import DfaSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import TransitionFocusOracle, RandomWMethodEqOracle, RandomWordEqOracle
from aalpy.utils import visualize_automaton, load_automaton_from_file

from RNN_SULs import RNN_BinarySUL_for_Weiss_Framework
from Refinement_based_extraction.DFA import DFA
from Refinement_based_extraction.Extraction import extract
from Refinement_based_extraction.GRU import GRUNetwork
from Refinement_based_extraction.LSTM import LSTMNetwork
from Refinement_based_extraction.RNNClassifier import RNNClassifier
from Refinement_based_extraction.Specific_Language_Generation import get_balanced_parantheses_train_set
from Refinement_based_extraction.Tomita_Grammars import tomita_dicts
from Refinement_based_extraction.Training_Functions import make_train_set_for_target, mixed_curriculum_train


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def Weiss_to_AALpy_DFA_format(dfa: DFA):
    """
    Transform the DFA fromat found in Weiss Framework to the AALpy DFA format
    """
    from aalpy.automata import Dfa, DfaState

    alphabet = dfa.alphabet
    states = []
    prefix_state_dict = {}
    for i, prefix in enumerate(dfa.Q):
        state = DfaState(f's{i}')
        state.prefix = prefix
        prefix_state_dict[prefix] = state
        states.append(state)

    initial_state = None
    for state in states:
        if state.prefix == dfa.q0:
            initial_state = state
            break

    assert initial_state

    for state in states:
        if state.prefix in dfa.F:
            state.is_accepting = True

    for prefix, val in dfa.delta.items():
        state = prefix_state_dict[prefix]
        for inp, out in val.items():
            state.transitions[inp] = prefix_state_dict[out]

    dfa = Dfa(initial_state, states)
    for state in states:
        state.prefix = dfa.get_shortest_path(initial_state, state)

    return dfa


#
def verify_cex(aalpy_model, white_box_model, rnn, cex_set):
    """
    Verify that counterexamples are not spurious and find which model classified correctly
    :param aalpy_model: model obtained by our approach
    :param white_box_model: modle obtained by refinement-based learning
    :param rnn: RNN that serves as system under learning
    :param cex_set: found cases of non-conformance between two models
    :return:
    """
    correct_model = None
    for cex in cex_set:
        sul1, sul2 = DfaSUL(aalpy_model), DfaSUL(white_box_model)
        output_black_box = sul1.query(cex)[-1]
        output_white_box = sul2.query(cex)[-1]

        rnn.renew()
        rnn_sul = RNN_BinarySUL_for_Weiss_Framework(rnn)
        rnn_output = rnn_sul.query(cex)[-1]

        if output_black_box == output_white_box:
            return False
        if output_black_box != rnn_output and output_white_box != rnn_output:
            return False
        if output_black_box == rnn_output:
            if correct_model and correct_model == 'White-Box':
                assert False
            correct_model = 'Black-Box'
        else:
            print(output_black_box)
            print(rnn_output)
            if correct_model and correct_model == 'Black-Box':
                assert False
            correct_model = 'White-Box'

    print(f'All examples were classified correctly by the {correct_model} model and misclassified by the other.')
    return True


# Trains RNN or Loads it from the file if it has been pretrained
def train_or_load_rnn(example, num_layers=2, hidden_dim=50, rnn_class=GRUNetwork, train=True):
    alphabet = "01" if 'tomita' in example else list(string.ascii_lowercase + "()")

    rnn = RNNClassifier(alphabet, num_layers=num_layers, hidden_dim=hidden_dim, RNNClass=rnn_class)

    nn_props = F'{"GRU" if rnn_class == GRUNetwork else "LSTM"}_layers_{num_layers}_dim_{hidden_dim}'
    if 'tomita' in example:
        target_fun = tomita_dicts[example]
        # more lenghts defined for tomita 5 and 6 as otherwise learning will fail
        lengths = (1, 2, 3, 4, 5, 6, 7, 8) if example == 'tomita_5' or example == 'tomita_6' else None
        train_set = make_train_set_for_target(target_fun, alphabet, max_train_samples_per_length=500,
                                              search_size_per_length=2000, lengths=lengths)
    else:
        # balanced parentheses train set
        train_set = get_balanced_parantheses_train_set(n=20000, short=1, longg=5)

    if train:
        # Train the RNN
        mixed_curriculum_train(rnn, train_set, stop_threshold=0.0005)
        # Save it to file
        rnn.save(f'RNN_Models/WeissComparisonModels/{example}_{nn_props}.rnn')
        print('saving to', f'RNN_Models/WeissComparisonModels/{example}_{nn_props}.rnn')
        #exit()
    else:
        # loads the neural network if it has been pretrained... will terminate the execution if weights file does
        # not exist
        rnn.load(f'RNN_Models/WeissComparisonModels/{example}_{nn_props}.rnn')

    return rnn, alphabet, train_set


def run_comparison(example, train=True, num_layers=2, hidden_dim=50, rnn_class=GRUNetwork,
                   insufficient_testing=False, verbose=False):
    rnn, alphabet, train_set = train_or_load_rnn(example, num_layers=num_layers, hidden_dim=hidden_dim,
                                                 rnn_class=rnn_class, train=train)

    # initial examples for Weiss et Al
    all_words = sorted(list(train_set.keys()), key=lambda x: len(x))
    pos = next((w for w in all_words if rnn.classify_word(w) is True), None)
    neg = next((w for w in all_words if rnn.classify_word(w) is False), None)
    starting_examples = [w for w in [pos, neg] if None is not w]

    # Extract Automaton Using White-Box eq. query
    rnn.renew()
    if verbose:
        print('---------------------------------WHITE BOX EXTRACTION--------------------------------------------------')
    else:
        blockPrint()
    start_white_box = time.time()
    dfa_weiss = extract(rnn, time_limit=500, initial_split_depth=10, starting_examples=starting_examples)
    time_white_box = time.time() - start_white_box
    # Make sure that internal states are back to initial
    rnn.renew()

    if verbose:
        print('---------------------------------BLACK BOX EXTRACTION--------------------------------------------------')
    sul = RNN_BinarySUL_for_Weiss_Framework(rnn)

    alphabet = list(alphabet)

    # define the equivalence oracle
    if insufficient_testing:
        eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=3, max_walk_len=12)
    else:
        eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=10, walk_len=5)
        if 'tomita' not in example:
            eq_oracle = TransitionFocusOracle(alphabet, sul, num_random_walks=1000, walk_len=20)

    start_black_box = time.time()
    aalpy_dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='dfa', max_learning_rounds=10,
                          print_level=1 if verbose else 0, cache_and_non_det_check=True, cex_processing='rs')
    time_black_box = time.time() - start_black_box

    enablePrint()
    if insufficient_testing:
        if len(aalpy_dfa.states) == len(dfa_weiss.Q):
            translated_weiss_2_aalpy = Weiss_to_AALpy_DFA_format(dfa_weiss)
            sul = DfaSUL(translated_weiss_2_aalpy)
            eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=1000, walk_len=10)

            cex = eq_oracle.find_cex(aalpy_dfa)
            if not cex:
                print(
                    '-------------------------WHITE-Box vs. BLACK-BOX WITH INSUFFICIENT TESTING -------------------------')
                print('White-box and Black-box technique extracted the same automaton.')
                print(f'White-box time: {round(time_white_box, 2)} seconds.')
                print(f'Black-box time: {round(time_black_box, 2)} seconds.')
            else:
                verify_cex(aalpy_dfa, translated_weiss_2_aalpy, rnn, [cex])
        return

    visualize_automaton(aalpy_dfa)
    print(aalpy_dfa)
    if len(aalpy_dfa.states) != len(dfa_weiss.Q):
        print('---------------------------------WHITE vs. BLACK BOX EXTRACTION----------------------------------------')
        nn_props = F'{"GRU" if rnn_class == GRUNetwork else "LSTM"}_layers_{num_layers}_dim_{hidden_dim}'
        print(f'Example       : {example}')
        print(f'Configuration : {nn_props}')
        print(f"Number of states\n  "
              f"White-box extraction: {len(dfa_weiss.Q)}\n  "
              f"Black-box extraction: {len(aalpy_dfa.states)}")

        translated_weiss_2_aalpy = Weiss_to_AALpy_DFA_format(dfa_weiss)

        sul = DfaSUL(translated_weiss_2_aalpy)
        eq_oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=10000, walk_len=20)
        if 'tomita' not in example:
            eq_oracle = TransitionFocusOracle(alphabet, sul)

        cex_set = []
        for _ in range(10):
            cex = eq_oracle.find_cex(aalpy_dfa)
            if cex and cex not in cex_set:
                cex_set.append(cex)

        cex_set.sort(key=len)
        # verify that the counterexamples are not spurios and find out which model is correct one
        real_cex = verify_cex(aalpy_dfa, translated_weiss_2_aalpy, rnn, cex_set)
        if not real_cex:
            print('Spurious CEX')
            assert False
        print('Few Counterexamples')
        print('  ', cex_set[:3])


def falsify_refinement_based_model():
    """
    Show how extensive coverage-based testing can be used to falsify model returned from refinement-based extraction
    approach.
    """
    rnn, alphabet, train_set = train_or_load_rnn('bp_1', num_layers=2, hidden_dim=50,
                                                 rnn_class=GRUNetwork, train=False)

    # initial examples for Weiss et Al
    all_words = sorted(list(train_set.keys()), key=lambda x: len(x))
    pos = next((w for w in all_words if rnn.classify_word(w) is True), None)
    neg = next((w for w in all_words if rnn.classify_word(w) is False), None)
    starting_examples = [w for w in [pos, neg] if None is not w]

    # Extract Automaton Using White-Box eq. query
    rnn.renew()

    start_white_box = time.time()
    dfa_weiss = extract(rnn, time_limit=500, initial_split_depth=10, starting_examples=starting_examples)
    time_white_box = time.time() - start_white_box
    # Make sure that internal states are back to initial
    rnn.renew()

    white_box_hyp = Weiss_to_AALpy_DFA_format(dfa_weiss)
    sul = RNN_BinarySUL_for_Weiss_Framework(rnn)
    eq_oracle = TransitionFocusOracle(alphabet, sul, num_random_walks=1000, walk_len=20)

    cex_set = set()
    for _ in range(10):
        start_time = time.time()
        cex = eq_oracle.find_cex(white_box_hyp)
        if tuple(cex) in cex_set:
            continue
        cex_set.add(tuple(cex))
        end_time = time.time() - start_time
        print(round(end_time, 2), "".join(cex))


def find_bp_cex():
    """
    This example shows how transition focus equivalence oracle can be used to efficiently find counterexamples.
    """
    rnn, alphabet, train_set = train_or_load_rnn('bp_2', num_layers=2, hidden_dim=50,
                                                 rnn_class=GRUNetwork, train=False)

    model = load_automaton_from_file('TrainingDataAndAutomata/bp_depth4.dot', automaton_type='dfa')
    sul = RNN_BinarySUL_for_Weiss_Framework(rnn)
    eq_oracle = TransitionFocusOracle(alphabet, sul, num_random_walks=1000, walk_len=20)

    cex_set = set()
    for _ in range(10):
        start_time = time.time()
        cex = eq_oracle.find_cex(model)
        if tuple(cex) in cex_set:
            continue
        cex_set.add(tuple(cex))
        end_time = time.time() - start_time
        print(round(end_time, 2), "".join(cex))


if __name__ == '__main__':

    falsify_refinement_based_model()
    exit()

    # Run extraction on all pre-trained tomita examples
    for tomita_ex in tomita_dicts.keys():
        for nn in [GRUNetwork, LSTMNetwork]:
            run_comparison(tomita_ex, rnn_class=nn, train=False, insufficient_testing=True)

    # Run extraction on all pre-trained bp examples
    for i in range(1, 3):
        for nn in [GRUNetwork, LSTMNetwork]:
            run_comparison(f'bp_{i}', rnn_class=nn, train=False)
