# Extracting FSMs from RNNs

This artifact contains the Python source code for performing experiments on extracting finite state machines 
from RNNs trained on regular languages. The experiments compare three different approaches based on active automata learning:
1. Active automata learning combined with model-guided conformance testing
2. Active automata learning combined with refinement-based white-box analysis of hypothesis automata 
3. Active automata learning combined with random sampling with PAC guarantees

# Setup

## Installing Python 3.6 and Graphviz
The TACAS VM comes with Python 3.8 already preinstalled. However, this version deprecated and removed some methods that the refinement-based analysis uses. To run the experiments, it is therefore necessary to install Python 3.6.
The following console commands will install this version of Python as well as the virtual environment support that we will use.
```
sudo add-apt-repository ppa:deadsnakes/ppa 
sudo apt-get update 
sudo apt-get install python3.6-dev
sudo apt-get install python3.6-venv
```
Please enter the VM password when required.
After installing Python 3.6, upgrade pip using:
```
pip install --upgrade pip
``` 

In addition to Python, we use Graphviz to visualize FSMs, which needs to be installed using 
```
sudo apt-get install graphviz
```

## Setting up the Experiments
To setup the environment for the experiments, perform the following steps:
1. Extract the artifact contents. We will refer to the directory containing the source code as 
"/path/to/code". 
2. Open the bash in "/path/to/code"
3. Create a virtual environment "env" via ```python3.6 -m venv env``` and activate it via ```source env/bin/activate```
4. Install the Python dependencies using pip through ```pip install -r requirements.txt```

After the successful completion of these steps, the experiments can be performed. Please note that the virtual environments created above needs to be active. That is, when you close bash after the setup you should activate the virtual environment again by navigating to "/path/to/code" and issuing the command ```source env/bin/activate```.

# Running the Experiments

## Checking the Setup

To check whether the setup was successful, run the main method of "/path/to/code/quick_start.py" using
```
python3.6 quick_start.py 
```
This will train a RNN on the Tomita 3 grammar and extract the automaton from trained RNN using model-guided learning.

## Reproducing the Experiments from the Paper

### Command line setup for Experiment Reproduction
To reproduce the experiments, run the methods defined in of "/path/to/code/evaluation.py".
To see available methods, run "/path/to/code/evaluation.py -h".
Few examples:
- python evaluation.py compare_all tomita_3 (corresponding to Table 1.)
- python evaluation.py falsify_refinement tomita_3 (corresponding to Table 2.)
- python evaluation.py falsify_pac bp_1 (corresponding to Table 3.)
- python evaluation.py compare_pac bp_1 (corresponding to Table 4.)

Due to the randomness involved in multiple processes, results may vary (not substantially) between runs.

### For more detailed experimentation (if reviewer wants to change some settings)
```
python3.6 Comparison_with_White_Box_and_PAC.py 
```
This Python script will execute all experiments found in the in the paper and will print some information on the learning process on the console. Due to the randomness involved in multiple processes, results may vary (not substantially) between runs.

# Availability
This artifact is available at [figshare](figshare.com/todo). After acceptance, we will publish it via figshare, thus assigning a DOI to it and making it permanently publicly available.  

The source code is also available at [GitHub](https://github.com/DES-Lab/Extracting-FSM-From-RNNs).

# Further Information
The [notebooks folder](https://github.com/DES-Lab/Extracting-FSM-From-RNNs/tree/master/notebooks) of the Github repository
contains the text output and interactive examples for some experiments.
