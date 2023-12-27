# Continuous-Time Dynamic Graphs
This is a framework to easlity experiment with Graph Neural Networks (GNNs) in the *Continuous-Time Dynamic Graph* domain, i.e., the graph is observed as a stream of events. The framework allows to run robust model selection and assessment leveraging same experimental seeds and data splits, for *reproducible* and *robust* evaluation within the tasks of link prediction.


Also refer to this [repository](https://github.com/gravins/CTDG-learning-framework) for a more comprehensive experimental framework for deep learning on Continuous-Time Dynamic Graphs.

---

## Requirements
_Note: we assume Miniconda/Anaconda is installed, otherwise see this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for correct installation. The proper Python version is installed during the first step of the following procedure._

Install the required packages and create the environment with create_env script
``` 
./create_env.sh 
```

or create the environment from the yml file
``` 
conda env create -f environment.yml
conda activate ctdg 
```

---

## How to reproduce our results
To reproduce our results please run:
```
python3 run.py
```

For more details about distrubuted execution, please refer to our main framework [repository](https://github.com/gravins/CTDG-learning-framework).
