#!/bin/bash
conda create -n ctdg python=3.9
conda activate ctdg

#conda install gpustat -c conda-forge

python3 -m pip install torch==1.13.1+cpu
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
python3 -m pip install torch_geometric==2.3.1 -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
python3 -m pip install ray==2.4.0
python3 -m pip install pandas==1.5.3
python3 -m pip install tqdm==4.65.0
python3 -m pip install wandb==0.15.0
python3 -m pip install scikit-learn==1.2.2
python3 -m pip install matplotlib==3.7.1