Near Fault Tolerant GSEE algorithms
===========================================================

References
----------

[Heaviside filter](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010318)
[Gaussian filter](https://arxiv.org/abs/2209.06811)

Authors: Chong Sun <sunchong137@gmail.com>
         Peter Johnson <peter@zapatacomputing.com>

Installation
------------
* create a new conda environment (optional)

        conda create --name <env_name> --file requirements.txt

        conda activate <env_name>

* install using setup.py

        pip install -e .

Key features
------------
* Approximate cumulate distribution function 

    acdf.py

* One qubit circuit example

    one_qubit_circ.py

* Binary search based on majority voting

    binary_search.py

Running the code
----------------
* Follow the scripts in ./examples and ./tests
