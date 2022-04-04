# Extracting FSMs from RNNs

This artifact contains the Python source code for performing experiments on extracting finite state machines 
from RNNs trained on regular languages. The experiments compare three different approaches based on active automata learning:
1. Active automata learning combined with model-guided conformance testing
2. Active automata learning combined with refinement-based white-box analysis of hypothesis automata 
3. Active automata learning combined with random sampling with PAC guarantees
Each combination uses Angluin's L* for learning, but a different equivalence-query
implementation to check hypothesis automata.

# Setup

## Installing Python 3.6 and Graphviz
The TACAS VM comes with Python 3.8 already preinstalled. However, this version deprecated and removed some methods that the refinement-based analysis uses. To run the experiments, it is, therefore, necessary to install Python 3.6.
The following console commands will install this version of Python as well as the virtual environment support that we will use.
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6-dev
sudo apt-get install python3.6-venv
```
Please enter the VM password when required.


In addition to Python, we use Graphviz to visualize FSMs, which needs to be installed using
```
sudo apt-get install graphviz
```

## Setting up the Experiments
To set up the environment for the experiments, perform the following steps:
1. Extract the artifact contents. We will refer to the directory containing the source code as
"/path/to/code".
2. Open the bash in "/path/to/code"
3. Create a virtual environment "env" via ```python3.6 -m venv env``` and activate it via ```source env/bin/activate```
4. After activating the virtual environment, upgrade pip using:
```
pip install --upgrade pip
```
5. Install the Python dependencies using pip through ```pip install -r requirements.txt```

After the successful completion of these steps, the experiments can be performed. Please note that the virtual environment created above needs to be active. That is, when you close bash after the setup you should activate the virtual environment again by navigating to "/path/to/code" and issuing the command ```source env/bin/activate```.

# Running the Experiments

## Checking the Setup

To check whether the setup was successful, run the of "/path/to/code/quick_start.py" using
```
python3.6 quick_start.py
```
This will train an RNN on the Tomita 3 grammar and extract the automaton from the trained RNN using model-guided learning.

## Reproducing the Experiments from the Paper

### Command Line Setup for Experiment Reproduction
To reproduce the experiments, run the script "/path/to/code/evaluation.py" with the
appropriate command line parameters.
To see available parameter values, run "/path/to/code/evaluation.py -h".
A few examples:
- python3.6 evaluation.py compare_all tomita_3 (corresponding to Table 1.)
- python3.6 evaluation.py falsify_refinement tomita_3 (corresponding to Table 2.)
- python3.6 evaluation.py falsify_pac bp_1 (corresponding to Table 3.)
- python3.6 evaluation.py compare_pac bp_1 (corresponding to Table 4.)

The first parameter specifies which kind of experiment to perform and the second specifies
which case study subject to use. The subject is one of seven Tomita grammar or the balanced parentheses grammar.
Due to the randomness involved in multiple processes, results may vary (not substantially) between runs.
The general trends of the presented experimental results are reproducible.
We will now explain what each command-line option does and the relation between the produced
console output and the results presented in the tables of the paper.

### ```compare_all``` -- Table 1
The command-line option ```compare_all``` runs active automata learning three times,
once for each type of equivalence-query implementation. It will print some output
on the automata-learning process first and show a direct comparison between each of the three methods at the end.
The comparison header is "COMPARISON OF EXTRACTIONS" and first shows the subject
under evaluation (e.g., tomita_3) and the RNN configuration. Then, it shows the numbers of states of each of the three learned automata as well as the number of counterexamples during the learning runs.
An example output is:
```
---------------------------------COMPARISON OF EXTRACTIONS----------------------------------------
Example       : tomita_3
Configuration : GRU_layers_2_dim_50
Number of states
  White-box extraction       : 5
  PAC-Based Oracle           : 5
  Coverage-guided extraction : 1001
Number of counterexamples found
  White-box extraction       : 3
  PAC-Based Oracle           : 2
  Coverage-guided extraction : 11
```
That is, using the refinement-based and the PAC random-sampling oracle resulted in learning 5-state automata, whereas using the coverage-guided oracle resulted in learning a more accurate 1001-state automaton. This can be explained by considering
that the coverage-guided oracle found more counterexamples.

The row corresponding to the Tomita 2 grammar in Table 1 can be reproduced using
the command ```python3.6 compare_all tomita_2```. All other rows can be reproduced
similarly.
For efficiency reasons, we limit the number of counterexamples to at most 11, which translates
to the number of rounds of active automata learning. As a result, not all results
can be reproduced exactly (e.g., coverage-guided testing detected 115 counterexamples
for Tomita 5) but the general trend of the measurement is reproducible.
This limit can be changed by changing lines 185 and 200 in the Python file
"Comparison_with_White_Box_and_PAC.py". The optional parameter "max_learning_rounds"
of the function "run_Lstar" encodes this limit. Setting the parameter to "None"
completely removes the limit.

### ```falsify_refinement``` -- Table 2
The command-line option ```falsify_refinement``` runs active automata learning first
with the refinement-based equivalence oracle to learn an initial automaton. Then,
this learned automaton is used for model-guided conformance testing, which tries to falsify
the automaton. Table 2 in the paper presents results from such experiments.
Running the script with "falsify_refinement" will first print debug output from active automata learning with the refinement-based oracle. After initial learning
is finished, the script prints lines of the form:
```
# tests needed to find a counterexample: 15
0.12 (())
```
The first line shows how many tests from model-guided conformance testing are required
to falsify the initially learned automaton. The second line contains the required runtime
and the counterexample.
Column 6 of Table 2 can be reproduced using the command
```
python3.6 falsify_refinement bp_1
```
The values from the second column of Table 2 can be read from the debug output
printed by the initial learning. This output contains lines of the format, where
the first value is a counterexample found by the refinement-based oracle and the
second value is the required time:
```
('))',0.4124270000000001
```


### ```falsify_pac``` -- Table 3
The command-line option ```falsify_pac``` runs active automata learning first
with the random-sampling oracle with PAC guarantees to learn an initial automaton. Then,
this learned automaton is used for model-guided conformance testing, which tries to falsify
the automaton. Table 3 in the paper presents results from such experiments.
Similarly for "falsify_refinement", the script will print lines of the form:
```
# tests needed to find a counterexample: 4
(aa(weako)(
```
Compared to "falsify_refinement", the initial debug output is different, because
we use the active-learning implementation by Weiss et al. for their refinement-based
oracle, whereas we use our library AALpy otherwise. For "falsify_pac" we do not provide the runtime required for falsification, which is in line with the paper.

Column 4 of Table 3 can be reproduced using the command
```
python3.6 falsify_pac bp_1
```

### ```compare_pac``` -- Table 4
The command-line option ```compare_pac``` runs active automata learning two times,
once with random sampling with PAC guarantees and once with model-guided conformance testing.
After starting PAC sampling-based learning, the script prints "STARTING PAC-BASED LEARNING".
This is followed by console output of the form:
```
Hypothesis i: n states.
Num tests: m
<counterexample>
```
The number n denotes the size of the i-th hypothesis automaton and the number m is the number of tests (randomly sampled at first) required to falsify the i-th hypothesis.
Since the last hypothesis is not falsified during learning, the number of tests is given for all but the last hypotheses.
The experiments presented in Table 4 can be reproduced via:
```
python3.6 compare_pac bp_1
```
Row 2 of Table 4 contains the hypothesis sizes, i.e., the numbers n printed to the console. Row 3 of Table 4 contains the tests required to falsify hypotheses, i.e.,
the numbers m printed to the console.

After PAC-based learning is finished, some statistics on the learning run are printed to the console.
Next, learning with model-guided conformance testing is started, which is signaled by printing "STARTING MODEL-GUIDED LEARNING".
For this type of learning, the same output format as above is used, where the
concrete results correspond to Row 4 and Row 5 of Table 4.

These experiments may take a long time to complete.

### Detailed Experimentation
For more detailed experimentation the python script "Comparison_with_White_Box_and_PAC.py"
can be used. It also allows changing some settings.
Running
```
python3.6 Comparison_with_White_Box_and_PAC.py
```
will execute all experiments found in the paper and will print some information on the learning process on the console. As noted above, results may vary between runs due to the randomness involved.

# Availability
This artifact is available at [figshare](https://figshare.com/s/2ab12fec60db3087b5a3). This is currently a private link. After acceptance, we will publish the artifact via figshare, thus assigning a DOI to it and making it permanently publicly available.

The source code is also available at [GitHub](https://github.com/DES-Lab/Extracting-FSM-From-RNNs).

# Further Information
The [notebooks folder](https://github.com/DES-Lab/Extracting-FSM-From-RNNs/tree/master/notebooks) of the Github repository
contains the text output and interactive examples for some experiments.
