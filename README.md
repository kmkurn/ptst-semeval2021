# Parsimonious Transfer for Sequence Tagging

This repository contains the code for the University of Melbourne's submission (PTST-UoM) in
SemEval 2021 Task 10: Source-Free Domain Adaptation for Semantic Processing, subtask time
expression recognition.

## Installing requirements

We recommend you to use conda package manager. Then, create a virtual environment with all
the required dependencies with:

```
conda env create -n [env] -f environment.yml
```

Replace `[env]` with your desired environment. Once created, activate the environment. The
command above also installs the CPU version of PyTorch. If you need the GPU version, follow
the corresponding PyTorch installation docs afterwards. If you're using other package manager
(e.g., pip), you can look at the environment.yml file to see what the requirements are.

## Get the data

Follow the instructions on the task's CodaLab site: https://competitions.codalab.org/competitions/26152#learn_the_details-getting-started-time

## Running PTST

Assuming you have the practice data in the current working directory, run:

```
./run_ptst.py with best gold_path=practice_data corpus.path=practice_text
```

This command will train a model on the given practice text. To see other available options, run:

```
./run_ptst.py print_config
```

## Sacred: an experiment manager

Script `run_ptst.py` uses [Sacred](https://github.com/IDSIA/sacred/) so that you can store all
about an experiment run in a MongoDB database. Simply set environment variables `SACRED_MONGO_URL`
to point to a MongoDB instance and `SACRED_DB_NAME` to a database name to activate it. Also,
invoke the help command to print its usage, i.e. `./run_ptst.py help`.
