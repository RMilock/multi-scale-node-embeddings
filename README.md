# Multi-Scale-Node-Embeddings
## Description:
This package contains a set of methods to calculate:

1) Renormalizable Vector Embeddings, if the full adjacency matrix is known.
See "Multi-Scale Node Embeddings for Graph Modeling and Generation" [arXiv](https://arxiv.org/abs/2412.04354);

2) Global or local scalar embeddings, if only marginal quantities are available, such as the number of links or the node degrees. See "Renormalizable graph embeddings for multi-scale network reconstruction" (stay tuned)

Feel free to reach out at [riccardo.milocco@imtlucca.it](mailto:riccardo.milocco@imtlucca.it) or [riccardo.milocco@gmail.com](mailto:riccardo.milocco@gmail.com), for code refinement, feedback, or collaborations.

## Installation
------------
Install using:

python3 -m venv .venv
pip3 install --editable . --config-settings editable_mode=compat


## Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:


    git merge --no-ff myfeature

To build a development environment run (Linux):


    python3 -m venv .venv 
    source .venv/bin/activate 
    pip3 install --editable . --config-settings editable_mode=compat

Have fun! 🚀