# Decomposed-Hybrid-Precoding

This repository provides the implementation of the methods presented in our paper for both **narrowband** and **wideband** MIMO communication scenarios.

Repository address:

```text
https://github.com/Freshznt0213/Decomposed-Hybrid-Precoding
```

## Overview

The repository contains two main parts:

* Narrowband MIMO channel experiments
* Wideband MIMO channel experiments

The narrowband implementation includes channel dataset generation, GNN model training, and MATLAB-based numerical optimization. For the wideband setting, we provide a pretrained model that can be directly evaluated under a large-scale Urban Macro-cell channel scenario.

## Narrowband Scenario

For the narrowband scenario, please follow the steps below.

### 1. Generate the Channel Dataset

Use `gen_dataset.py` to generate channel realizations according to the desired system and channel configurations.

Before running the script, please modify the relevant parameters in `gen_dataset.py` to match your experimental settings, such as the number of antennas, number of users.

### 2. Train the GNN Model

After generating the channel dataset, run `MIMOHPC_main.py` to train the GNN model proposed in the paper.

Please ensure that the dataset path and system parameters in `MIMOHPC_main.py` are consistent with those used during dataset generation.

### 3. Compute the Numerical Optimum

The MATLAB script

```text
MIMOprecoding.m
```

can be used to compute the numerically optimized precoding solution for the corresponding channel scenario.

The numerical results obtained from this script can be used as a performance benchmark for evaluating the proposed learning-based model.

## Wideband Scenario

For the wideband scenario, we provide a pretrained model trained under a large-scale **Urban Macrocell (UMa)** channel configuration.

The pretrained model can be directly evaluated without retraining to reproduce the wideband performance reported in the paper.

Please configure the test parameters and pretrained-model path according to the provided testing scripts before running the evaluation.

## Recommended Workflow

For narrowband experiments:

```text
Generate channel data with gen_dataset.py
                ↓
Train the GNN model with MIMOHPC_main.py
                ↓
Compute the numerical benchmark with MIMOprecoding.m
                ↓
Compare the model performance with the numerical optimum
```

For wideband experiments:

```text
Load the provided pretrained UMa model
                ↓
Run the corresponding testing script
                ↓
Evaluate the performance reported in the paper
```

## Requirements

The narrowband learning-based implementation requires Python and a compatible deep-learning environment. The numerical optimization code requires MATLAB.

Please refer to the source files for detailed model parameters, channel configurations, and experimental settings.

## Repository

The complete source code is available at:

```text
https://github.com/freshznt/hyb.com
```
