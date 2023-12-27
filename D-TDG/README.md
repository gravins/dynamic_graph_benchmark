# Discrete-Time Dynamic Graphs
This is the repository to reproduce our experiments in the *Discrete-Time Dynamic Graph* domain, i.e., the graph is observed as a stream of snapshots. Our code allows to run robust model selection and assessment leveraging same experimental seeds and data splits, for *reproducible* and *robust* evaluation.


---

## Requirements
_Note: we assume Miniconda/Anaconda is installed, otherwise see this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for correct installation. The proper Python version is installed during the first step of the following procedure._

Create the environment from the yml file
``` 
conda env create -f environment.yml
conda activate dtdg 

python -m pip install kaggle
conda install jemalloc -c conda-forge
```

---

## How to reproduce our results
To reproduce our results:

1) download and untar the DATA folder:
```
wget https:\\
tar -xzvf DATA.tar.gz
```

2) build the dataset and data splits
```
export data_config=DATA_CONFIGS/spatio_temporal/config_montevideo.yml
pydgn-dataset --config-file $data_config
```

3) run the training
```
# Jemalloc init
export MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000
export LD_PRELOAD=~/miniconda3/envs/dtdg/lib/libjemalloc.so

export model_config=MODEL_CONFIGS/SpatioTemporal/config_a3tgcn_montevideo.yml
pydgn-train --config-file $model_config
```

Please set ``data_config`` and ``model_config`` according to the data and model that you want to experiment.

Also, refer to [PyDGN](https://github.com/diningphil/PyDGN) for more details about the experimental framework.
