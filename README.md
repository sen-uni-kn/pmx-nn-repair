# pharmacometrics-nn-repair
TODO:

## Downloading this Repository
To download all files in this repository, you may either download
the files as a `.zip` archive using the webinterface, or you can
clone the repository using git. 

Cloning using git requires you to install [git-lfs](https://git-lfs.com/).
After cloning the project, you have to pull the submodules and the dataset
and neural network files that are handled by git-lfs.
```bash
git clone REPOSITORY-URL-TODO
git submodule update --init
git lfs install && git lfs pull
```

## Installation
This project requires Ubuntu 22.04 as operating system.
It uses Python 3.8. 
To run on windows, install the project inside a virtual machine.

To install the project in Ubuntu 22.04, open a terminal and navigate
to the directory where this README file is placed.
Make sure to update the submodules first, as described in the previous
section.
Then, run:
```bash
cd specrepair
./setup.sh
cd ..  # navigate back to the main directory for running the experiments
```
This script installs system dependencies and creates a new virtual environment
for this project.
The script will query you for sudo privileges.
Running the script may take 10-20 minutes.

This project also requires a Gurobi license for running the ERAN verifier.
Gurobi provides [free academic licenses](https://www.gurobi.com/academia/academic-program-and-licenses/).
Follow the instructions in `specrepair/README.md` on obtaining a
Gurobi license.
Your Gurobi license needs to be placed at `$HOME/gurobi.lic` 
(the default installation location).

## Running the Experiments
Once you have finished the installation, you can
activate the new virtual enviroment and run the experiments.
```bash
source specrepair/env-nn-repair/bin/activate  # activate the environment
# train networks on log-normally and uniformly distributed 1 CMP PO datasets
./train_1_cmp_po.sh
# repair some of the trained networks
./repair_1_cmp_po.sh
# train networks on log-normally and uniformly distributed IDR datasets
./train_idr.sh
```

## Tensorboard
The experiments allow visualizing datasets and training progress using
[tensorboard](https://www.tensorflow.org/tensorboard).
Pass the `--show_plots` command line option to an experiment to turn on
showing these plots.
For example:
```bash
./train_1_cmp_po.sh --show_plots
```
You can then view the plots in tensorboard by running
(while having the virtual environment activated)
```bash
tensorboard --logdir .tensorboard/
# Tensorboard starts and shows you a link at which you can
# view all plots.
```
Then, you can open tensorboard in your browser, 
usually at `http://localhost:6006/`.
