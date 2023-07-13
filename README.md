# PMX NN Repair 
This repository contains the source code for running the experiments
in the paper "Can we trust neural networks in pharmacometrics? 
Yes, we can, but we need guarantees!"
on verifying and repairing learned pharmacometric models
by David Boetius, Dominic Stefan Bräm, Marc Pfister, Stefan Leue, 
and Gilbert Koch.

You can download our datasets and trained and repaired neural networks
on [Zenondo](https://doi.org/10.5281/zenodo.8143636). 
The Zenodo upload also contains a virtual machine for running the
experiments in this repository.
You can also run the experiments on your own Ubuntu machine.
For this, follow the installation instructions below.

In any case, this project also requires a Gurobi license for running 
the ERAN verifier.
Gurobi provides [free academic licenses](https://www.gurobi.com/academia/academic-program-and-licenses/).
Follow the instructions in `specrepair/README.md` on obtaining a
Gurobi license.
Your Gurobi license needs to be placed at `$HOME/gurobi.lic` 
(the default installation location).
If you are using our virtual machine, the Gurobi license needs to
be placed *in* the virtual machine.

## Downloading this Repository
To download all files in this repository, you may either download
the files as a `.zip` archive using the webinterface, or you can
clone the repository using git. 

When cloning the project, you also have to pull the submodules.
```bash
git clone https://github.com/sen-uni-kn/pmx-nn-repair
git submodule update --init
```

## Running the Experiments
Once you have started the virtual machine or finished the installation, 
you can activate the SpecRepair virtual environment and run the experiments.
Navigate to the root folder of this repository. Then run:
```bash
source specrepair/env-nn-repair/bin/activate  # activate the environment
# train networks on log-normally and uniformly distributed 1 CMP PO datasets
./train_1_cmp_po.sh
# repair some of the trained networks
./repair_1_cmp_po.sh
# train networks on log-normally and uniformly distributed IDR datasets
./train_idr.sh
```

## Installation
This project requires Ubuntu 22.04 as operating system.
It uses Python 3.8. 
To run it on Microsoft Windows or other operating systems, 
use our virtual machine or install the project inside 
your own virtual machine.

To install the project in Ubuntu 22.04, open a terminal and navigate
to the directory where this README file is placed.
Make sure to update the submodules first, as described in the
"Downloading this Repository" section.
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

