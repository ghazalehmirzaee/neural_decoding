# Neural Decoding from M1: Hybrid CNN-BiLSTM for Bilateral Forelimb Movement Prediction

This repository implements the models and methods from the paper "Decoding Skilled Bilateral Forelimb Movements from Unilateral M1 Neuronal Networks: Integrating Hybrid Attention-Based Model with In Vivo Two-Photon Calcium Imaging".

## Overview

This paper implements advanced deep learning architectures to decode complex forelimb movements (footsteps) from neuronal activity recorded from the primary motor cortex (M1) using two-photon calcium imaging. The key finding is that unilateral M1 neuronal networks contain sufficient information to decode movements of both ipsilateral and contralateral forelimbs.

## Paper Summary

The research explores whether complex motor movements in both forelimbs of mice can be decoded from the activity of excitatory neuronal ensembles in unilateral M1. The study:

1. Uses in vivo two-photon calcium imaging to record neuronal activity in M1
2. Collects data from mice performing a specialized grid-walking task requiring skilled movement
3. Implements two neural decoding approaches:
   - LSTM with fully connected layers (baseline)
   - Hybrid CNN-BiLSTM with attention mechanisms (advanced model)
4. Demonstrates accurate decoding of both ipsilateral and contralateral forelimb movements

The hybrid model showed significant performance improvements by incorporating both spatial features (through CNNs) and temporal dependencies (through BiLSTM) of neuronal activity.

## Repository Structure
```bash
.
├── config/                # Configuration files (Hydra)
│   ├── dataset/          # Dataset configurations
│   ├── model/            # Model architectures
│   └── training/         # Training parameters
├── scripts/              # Execution scripts
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── src/                  # Source code
│   ├── data/             # Dataset handling
│   ├── models/           # Model implementations
│   ├── training/         # Training utilities
│   └── utils/            # Metrics and visualization
└── requirements.txt      # Dependencies
```

## Installation
```bash
# Clone the repository
git clone https://github.com/ghazalehmirzaee/neural_decoding.git
cd neural_decoding

# Create and activate a new virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation
The code expects calcium imaging data with the following structure:

- CSV files containing calcium transients from neurons and behavioral labels
- Each row represents a time frame, with columns for:
  - Frame index
  - Neural activity for each neuron
  - Behavioral label (0: no footstep, 1: contralateral footstep, 2: ipsilateral footstep)

Place your data files in the data/ directory.

## Usage
## Training

To train the LSTM model:
```bash
python scripts/train.py model=lstm
```

To train the hybrid CNN-BiLSTM model:
```bash
python scripts/train.py model=hybrid
```

Additional configuration options:
```bash
# Use specific dataset
python scripts/train.py model=hybrid dataset=neural_data

# Modify training parameters
python scripts/train.py model=hybrid training.batch_size=16 training.learning_rate=0.0005

# Enable W&B logging
python scripts/train.py wandb.mode=online wandb.project=my-neural-decoding
```

## Evaluation
To evaluate a trained model:
```bash
python scripts/evaluate.py model=lstm
```
or:
```bash
python scripts/evaluate.py model=hybrid
```
