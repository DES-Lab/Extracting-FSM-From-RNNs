# Extraction of RNN's Input-Output Behaviour via Automata Learning and Testing-based Eq. Oracle

## Structure of the project:
- `LearnedAutomata/` - automata representing behaviour of RNN's inferred via automata learning
- `TrainingDataAndModels/` - data and models used for RNN training and verification 
- `Weiss_et_al/` - code from paper [Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples ](https://github.com/tech-srl/lstar_extraction)
###
- `Comparison_Weiss.py` - comparison of our approach to [Weiss et al.](https://github.com/tech-srl/lstar_extraction)
###
- `DataProcessing.py` - collection of helper functions used to prepare data for learning
- `RNNClassifier.py` - RNN class, either LSTM or GRU
###
- `RNN_SULs` - system under learning. Implements [AALpy](https://github.com/DES-Lab/AALpy)'s SUL class
- `TrainAndExtract.py` - train RNNs and extract their behaviour via automata learning

## How to Run

Simply call a function from `TrainAndExtract.py` with appropriate parameters.
Examples can be found in `main.py`.

## How to Install and Dependencies

To run extraction, only dependecy are AALpy and Dynet.
However, to run comparisson with Weiss et al, further dependencies need to be defined.

To install, clone this repo, (suggestion: create a python virtual environment) and call

``
pip install -r  requirements.txt
``