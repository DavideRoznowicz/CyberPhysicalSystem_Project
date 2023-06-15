# Dashpot - Cyber Physical System

## Author: Davide Roznowicz
-----------------------------------------------------------------------------------------


## Installation
Since we use a python package called [Moonlight](https://github.com/MoonLightSuite/MoonLight) to help in the trajectory control for verification of formal requirements, you must make sure to have `jdk17` installed.

After that, the only action you have to perform is install the conda environment via the file `env.yaml`:

```bash
conda env create -f environment.yaml
```


## Usage
The structure of the package is the following:
```bash
myProject
    main.ipynb
    __init__.py
    env.yaml
    measures.py
    Pendolum_Model.py
    pid_model.py
    README.md
```


* In [main.ipynb](/main.ipynb) we can find the notebook going step by step through the project
and giving some explanations while showing the relevant implementation detials.




