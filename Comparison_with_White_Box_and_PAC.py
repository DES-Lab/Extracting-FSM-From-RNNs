import os
import string
import sys
import time

from aalpy.SULs import DfaSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import TransitionFocusOracle, RandomWMethodEqOracle, RandomWordEqOracle, StatePrefixEqOracle
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
from Oracles import PacOracle, TransitionFocusOraclePrime, RandomWMethodEqOraclePrime


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
            if correct_model and correct_model == 'Model-guided based':
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
        # exit()
    else:
        # loads the neural network if it has been pretrained... will terminate the execution if weights file does
        # not exist
        rnn.load(f'RNN_Models/WeissComparisonModels/{example}_{nn_props}.rnn')

    return rnn, alphabet, train_set


rnn_classes = {'gru': GRUNetwork, 'lstm': LSTMNetwork}


def run_comparison(example, train=True, num_layers=2, hidden_dim=50, rnn_class='gru', verbose=True):
    print('Experiment comparing all learning processes. First white-box refinement based learning will be executed,'
          'then pac-based black-box learning, and finally model-guided learning.')
    assert rnn_class in rnn_classes.keys()
    rnn_class = rnn_classes[rnn_class]
    print('RNN is being trained or loaded.')
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
        print(
            '---------------------------------PAC-ORACLE (Bounded L*) EXTRACTION----------------------------------------')

    sul = RNN_BinarySUL_for_Weiss_Framework(rnn)

    alphabet = list(alphabet)

    pac_oracle = PacOracle(alphabet, sul, delta=0.01, epsilon=0.01, min_walk_len=3, max_walk_len=12)
    start_pac_time = time.time()
    pac_model = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=pac_oracle, automaton_type='dfa',
                          max_learning_rounds=10,
                          print_level=2, cache_and_non_det_check=True, cex_processing='rs')
    end_pac_time = time.time()

    if verbose:
        print(
            '---------------------------------COVERAGE-GUIDED EXTRACTION----------------------------------------')

    rnn.renew()
    sul = RNN_BinarySUL_for_Weiss_Framework(rnn)
    # define the equivalence oracle
    eq_oracle = RandomWMethodEqOraclePrime(alphabet, sul, walks_per_state=1000, walk_len=25)
    if 'tomita' not in example:
        eq_oracle = TransitionFocusOraclePrime(alphabet, sul, num_random_walks=1000, walk_len=20)
    start_black_box = time.time()
    aalpy_dfa = run_Lstar(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='dfa', max_learning_rounds=10,
                          print_level=2, cache_and_non_det_check=True, cex_processing='rs')
    time_black_box = time.time() - start_black_box

    enablePrint()

    if len(aalpy_dfa.states) != len(dfa_weiss.Q) or len(aalpy_dfa.states) != len(pac_model.states):
        print('---------------------------------COMPARISON OF EXTRACTIONS----------------------------------------')
        nn_props = F'{"GRU" if rnn_class == GRUNetwork else "LSTM"}_layers_{num_layers}_dim_{hidden_dim}'
        print(f'Example       : {example}')
        print(f'Configuration : {nn_props}')
        print(f"Number of states\n  "
              f"White-box extraction       : {len(dfa_weiss.Q)}\n  "
              f"PAC-Based Oracle           : {len(pac_model.states)}\n  "
              f"Coverage-guided extraction : {len(aalpy_dfa.states)}")

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
        # print('Few Counterexamples')
        # print('  ', cex_set[:3])
    else:
        print('Size of all extracted models (refinement-based, pac-based, coverage-based): ', len(aalpy_dfa.states))


def falsify_refinement_based_model(exp_name='bp_1'):
    """
    Show how extensive coverage-based testing can be used to falsify model returned from refinement-based extraction
    approach.
    """
    print('Experiment in which we show how model-guided testing can falsify the model learned with refinement-based extraction.')

    rnn, alphabet, train_set = train_or_load_rnn(exp_name, num_layers=2, hidden_dim=50,
                                                 rnn_class=GRUNetwork, train=False)

    # initial examples for Weiss et Al
    all_words = sorted(list(train_set.keys()), key=lambda x: len(x))
    pos = next((w for w in all_words if rnn.classify_word(w) is True), None)
    neg = next((w for w in all_words if rnn.classify_word(w) is False), None)
    starting_examples = [w for w in [pos, neg] if None is not w]

    # Extract Automaton Using White-Box eq. query
    rnn.renew()
    print("Refinement extraction started.")
    start_white_box = time.time()
    dfa_weiss = extract(rnn, time_limit=500, initial_split_depth=10, starting_examples=starting_examples)
    time_white_box = time.time() - start_white_box
    # Make sure that internal states are back to initial
    rnn.renew()

    white_box_hyp = Weiss_to_AALpy_DFA_format(dfa_weiss)
    sul = RNN_BinarySUL_for_Weiss_Framework(rnn)
    eq_oracle = TransitionFocusOraclePrime(alphabet, sul, num_random_walks=1000, walk_len=20)
    eq_oracle = RandomWMethodEqOraclePrime(alphabet, sul, walks_per_state=1500, walk_len=20)

    cex_set = set()
    print(f'Refinement-based learning learned {len(white_box_hyp.states)}-state automaton. Following 10 pairs of '
          f'outputs are time and number of tests needed to falsify refinement-based model.')
    for _ in range(10):
        start_time = time.time()
        cex = eq_oracle.find_cex(white_box_hyp)
        if not cex or tuple(cex) in cex_set:
            continue
        cex_set.add(tuple(cex))
        end_time = time.time() - start_time
        print(round(end_time, 2), "".join(cex))


def falsify_pac_based_model(exp_name='bp_1'):
    """
    Show how extensive coverage-based testing can be used to falsify model returned from bounden-L* extraction
    approach.
    """
    print('Experiment in which we show how model-guided testing can falsify the model learned with PAC-based approach.')
    rnn, alphabet, train_set = train_or_load_rnn(exp_name, num_layers=2, hidden_dim=50,
                                                 rnn_class=GRUNetwork, train=False)

    # Extract Automaton Using White-Box eq. query
    rnn.renew()

    pac_sul = RNN_BinarySUL_for_Weiss_Framework(rnn)
    pac_oracle = PacOracle(alphabet, pac_sul, epsilon=0.0001, delta=0.01, min_walk_len=4, max_walk_len=20)

    print('PAC-based Learning Started. Following outputs are outputs of the eq. oracle (number of tests needed to '
          'find a counterexample nad counterexample itself).')
    pac_model = run_Lstar(alphabet, pac_sul, pac_oracle, 'dfa', print_level=1)

    rnn.renew()
    coverage_sul = RNN_BinarySUL_for_Weiss_Framework(rnn)

    trans_focus = TransitionFocusOraclePrime(alphabet, coverage_sul, num_random_walks=1000, walk_len=20)
    random_w = RandomWMethodEqOraclePrime(alphabet, coverage_sul, walks_per_state=1500, walk_len=20)

    cex_set = set()
    print(f'PAC-based learning learned {len(pac_model.states)}-state automaton. Following 20 pairs of '
          f'outputs are time and number of tests needed to falsify PAC-based model.')
    for oracle in [trans_focus, random_w]:
        print(type(oracle).__name__)
        for _ in range(10):
            start_time = time.time()
            cex = oracle.find_cex(pac_model)
            if not cex or tuple(cex) in cex_set:
                continue
            cex_set.add(tuple(cex))
            end_time = time.time() - start_time
            print("".join(cex), round(end_time, 2), )


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
    print("Print Format: Time Counterexample")
    for _ in range(10):
        start_time = time.time()
        cex = eq_oracle.find_cex(model)
        if tuple(cex) in cex_set:
            continue
        cex_set.add(tuple(cex))
        end_time = time.time() - start_time
        print(round(end_time, 2), "".join(cex))


def comparison_of_learning_process(exp_name='tomita_3'):
    print('Comparison of PAC-based learning and model-guided learning.')
    rnn, alphabet, train_set = train_or_load_rnn(exp_name, num_layers=2, hidden_dim=50,
                                                 rnn_class=GRUNetwork, train=False)

    # Extract Automaton Using White-Box eq. query
    rnn.renew()
    print('STARTING PAC-BASED LEARNING')
    pac_sul = RNN_BinarySUL_for_Weiss_Framework(rnn)
    pac_oracle = PacOracle(alphabet, pac_sul, epsilon=0.01, delta=0.01, min_walk_len=4, max_walk_len=20)

    pac_model = run_Lstar(alphabet, pac_sul, pac_oracle, 'dfa', print_level=2, max_learning_rounds=10)

    rnn.renew()
    print('STARTING MODEL-GUIDED LEARNING')
    coverage_sul = RNN_BinarySUL_for_Weiss_Framework(rnn)

    # trans_focus = TransitionFocusOraclePrime(alphabet, coverage_sul, num_random_walks=1000, walk_len=20)
    random_w = RandomWMethodEqOraclePrime(alphabet, coverage_sul, walks_per_state=1000, walk_len=20)
    pac_model = run_Lstar(alphabet, coverage_sul, random_w, 'dfa', print_level=2, max_learning_rounds=10)


if __name__ == '__main__':
    # Two experiments showing falsification of PAC-based learned model
    falsify_pac_based_model(exp_name='bp_1')
    falsify_pac_based_model('tomita_3')
    # Two experiments showing falsification of refinement-based learned model
    falsify_refinement_based_model('bp_1')
    falsify_refinement_based_model('tomita_3')
    # Comparison of the whole learning process with the PAC-based learning
    comparison_of_learning_process(exp_name='tomita_3')
    comparison_of_learning_process(exp_name='tomita_7')
    # This example shows how transition focus equivalence oracle can be used to efficiently find counterexamples.
    find_bp_cex()

    print('--------------------------------------------------')
    print('Learning process comparison will be preformed on all examples with both supported RNN types.')
    # Run extraction on all pre-trained tomita examples
    for tomita_ex in tomita_dicts.keys():
        for nn in ['gru', 'lstm']:
            run_comparison(tomita_ex, train=False, rnn_class=nn)

    # Run extraction on all pre-trained bp examples
    for i in range(1, 3):
        for nn in ['gru', 'lstm']:
            run_comparison(f'bp_{i}', train=False, rnn_class=nn)

    # Run extraction on all pre-trained tomita examples
    for tomita_ex in tomita_dicts.keys():
        for nn in ['gru', 'lstm']:
            run_comparison(tomita_ex, train=False, rnn_class=nn)

    # Run extraction on all pre-trained bp examples
    for i in range(1, 3):
        for nn in ['gru', 'lstm']:
            run_comparison(f'bp_{i}', train=False, rnn_class=nn)