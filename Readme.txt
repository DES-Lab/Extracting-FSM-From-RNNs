Online Source Code:
https://github.com/DES-Lab/Extracting-FSM-From-RNNs

Install

To run extraction, only dependencies are AALpy and Dynet. However, to run a comparison with the refinement-based
approach proposed by Weiss et al., further dependencies are required.

To install, clone this repo (suggestion: create a python virtual environment) and call:

call pip install -r requirements.txt

Additional, but not necessarily needed requirement (for model visualization):
Graphviz - installed to the system and added to the path

Run

TrainAndExtract.py, Comparison_with_White_Box_and_PAC.py, and Applications.py all have main function defined at
the bottom of the file. quick_start.py has the simple minimal example that shows how to train RNNS and extract
multiple models. (to visualize models, ensure that Graphviz is installed to the system and added to the path)
Notebooks folder of the Github repository (https://github.com/DES-Lab/Extracting-FSM-From-RNNs/tree/master/notebooks)
contains the text output/interactive examples for some examples.
Furthermore, interactive notebooks can be found in the notebooks folder.

To reproduce experiments, simply run the __main__ method in the Comparison_with_White_Box_and_PAC.py.
Due to the randomness involved in multiple processes, results may vary (not substantially) between runs.