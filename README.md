# Neural density functional theory of liquid-gas phase coexistence

This repository contains code, datasets and models corresponding to the following publication:

**Neural density functional theory of liquid-gas phase coexistence**  
*Florian Sammüller, Matthias Schmidt, and Robert Evans.*

### Setup

Working in a virtual environment is recommended.
Set one up with `python -m venv .venv`, activate it with `source .venv/bin/activate` and install the required packages with `pip install -r requirements.txt`.
To use a GPU with Tensorflow/Keras, refer to the corresponding section in the installation guide at [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).

### Instructions

Simulation data can be found in `data` and trained models are located in `models`.
A sample script for thermal training of a neural functional from scratch is given in `learn.py`.
Utilities for making predictions with trained neural functionals are given in `utils.py`, see also `predict.py` for how to calculate self-consistent density profiles.

### Further information

The reference data has been generated with grand canonical Monte Carlo simulations using [MBD](https://gitlab.uni-bayreuth.de/bt306964/mbd).
