# Neural Decoding from M1 Neuronal Networks

This repository contains the code for the paper "Decoding Skilled Bilateral Forelimb Movements from Unilateral M1 Neuronal Networks: Integrating Hybrid Attention-Based CNN-BiLSTM Model with In Vivo Two-Photon Calcium Imaging".

## Overview

In this project, we develop and compare two approaches for decoding complex forelimb movements from neuronal activity recorded from unilateral M1 (primary motor cortex) using in vivo two-photon calcium imaging:

1. **LSTM Model**: A standard Long Short-Term Memory network that focuses solely on temporal dependencies in neuronal activity.
2. **Hybrid CNN-BiLSTM Model**: An advanced architecture that combines Convolutional Neural Networks (CNNs) for spatial feature extraction, Bidirectional LSTM for temporal processing, and attention mechanisms for focusing on the most relevant neuronal patterns.

Our study demonstrates that:
1. Excitatory neuronal ensembles in unilateral M1 can be leveraged to predict complex movement in bilateral forelimbs using deep learning models
2. The hybrid deep learning model substantially enhances decoding accuracy by capturing both spatial and temporal features of neuronal networks

## Key Contributions

- **Novel Architecture Design**: Our enhanced CNN-BiLSTM model integrates spatial and temporal processing in a unique way specifically optimized for neural decoding
- **Advanced Feature Integration**: Hierarchical skip connections preserve fine-grained neuronal patterns across processing stages
- **Adaptive Attention Mechanism**: Position-sensitive attention with dynamic scaling factors helps focus on the most behaviorally-relevant neuronal activity
- **Dynamic Normalization Strategy**: Combines batch, layer, and group normalization with learnable weights to handle the high variability in neuronal data
- **Multitask Learning Framework**: Simultaneous prediction of different behavioral aspects (multiclass, ipsilateral, contralateral) with neural activity regression

## Features

- **State-of-the-art Deep Learning Models**:
  - Enhanced CNN-BiLSTM with sophisticated attention mechanisms
  - Dynamic normalization techniques for handling neuronal data variability
  - Hierarchical skip connections for preserving fine-grained neuronal patterns
  - Adaptive focal loss for handling class imbalance
  - Multitask learning with uncertainty-based task weighting

- **Comprehensive Evaluation Framework**:
  - Detailed performance metrics (accuracy, precision, recall, F1-score)
  - Confusion matrices with percentages
  - ROC curves for all classification tasks
  - Neural activity visualization with behavior predictions

- **Experiment Management**:
  - Configuration management using Hydra
  - Experiment tracking with Weights & Biases
  - Efficient checkpointing and result logging
  - Reproducible experiment setup

## Dataset

The neural data consists of calcium imaging recordings from excitatory neuronal ensembles in the unilateral primary motor cortex (M1) of mice performing a grid-walking task. The dataset includes:

- Neuronal activity signals from hundreds of neurons (typically 400-600 neurons per recording)
- Behavioral labels for footsteps (no footstep, contralateral, ipsilateral)
- Deconvoluted calcium signals to improve temporal resolution of neural events
- Temporal sequences capturing the dynamics of neuronal activity during movement

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/username/neural-decoding.git
cd neural-decoding
```

2. Create and activate a conda environment:
```bash
conda create -n neural-decoding python=3.8
conda activate neural-decoding
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up Weights & Biases for experiment tracking:
```bash
wandb login
```

## Usage

### Data Preparation

Place your neural data in the `data/` directory. The dataset should be in CSV format with:
- First column: Frame indices
- Middle columns: Neuronal activity for each neuron
- Last column: Behavioral labels (0: no footstep, 1: contralateral, 2: ipsilateral)

### Training

To train the LSTM model:
```bash
python scripts/train.py model=lstm
```

To train the Hybrid CNN-BiLSTM model:
```bash
python scripts/train.py model=hybrid
```

### Evaluation

To evaluate the trained LSTM model:
```bash
python scripts/evaluate.py model=lstm
```

To evaluate the trained Hybrid CNN-BiLSTM model:
```bash
python scripts/evaluate.py model=hybrid
```

### Hyperparameter Tuning

Hydra enables easy hyperparameter sweeps:
```bash
python scripts/train.py --multirun model=hybrid model.hidden_size=64,128,256 training.learning_rate=0.001,0.0001
```

### Experiment Tracking

Monitor your training progress in real-time with W&B:
```bash
wandb sweep config/sweeps/hybrid_sweep.yaml
```

## Results

The hybrid CNN-BiLSTM model significantly outperforms the LSTM model in decoding accuracy, particularly for multiclass classification where it achieves ~15 percentage points improvement:

| Model | Task | Accuracy | F1-Score |
|-------|------|----------|----------|
| LSTM | Multiclass | 71.49% | 69.71% |
| LSTM | Contralateral | 88.64% | 89.35% |
| LSTM | Ipsilateral | 87.31% | 87.51% |
| Hybrid | Multiclass | 84.85% | 84.84% |
| Hybrid | Contralateral | 93.10% | 78.76% |
| Hybrid | Ipsilateral | 90.57% | 84.09% |

The results demonstrate that:

1. Unilateral M1 neuronal networks contain sufficient information to decode bilateral forelimb movements
2. Integrating spatial and temporal processing significantly enhances decoding performance
3. Our hybrid architecture with attention mechanisms can effectively capture complex spatiotemporal patterns in neuronal activity

## Visualization Examples

The repository includes tools for visualizing model performance:

- **Confusion Matrices**: Show the distribution of true vs. predicted classes with percentages
- **ROC Curves**: Illustrate the discriminative ability of the models for each class
- **Neural Activity Plots**: Display the relationship between neuronal activity and behavior predictions

## Model Architecture Details

### LSTM Model
- Input: Sequence of neuronal activity (time × neurons)
- Processing: Single-directional LSTM layers capture temporal dependencies
- Output: Behavioral class prediction

### Hybrid CNN-BiLSTM Model
- Input: Sequence of neuronal activity (time × neurons)
- Spatial Processing: CNN layers with skip connections extract spatial patterns
- Temporal Processing: BiLSTM captures forward and backward temporal dependencies
- Attention: Multi-head attention with adaptive scaling focuses on relevant patterns
- Output: Parallel prediction of behavioral classes and neural activity

## Project Structure

```
neural-decoding/
├── config/               # Hydra configuration files
│   ├── config.yaml       # Main configuration
│   ├── model/            # Model configurations
│   ├── dataset/          # Dataset configurations
│   └── training/         # Training configurations
├── data/                 # Data directory
├── notebooks/            # Analysis notebooks
├── outputs/              # Output directory for results
│   ├── checkpoints/      # Model checkpoints
│   ├── logs/             # Training logs
│   ├── results/          # Evaluation results
│   └── visualizations/   # Generated visualizations
├── scripts/              # Training and evaluation scripts
├── src/                  # Source code
│   ├── data/             # Data loading and processing
│   ├── models/           # Model implementations
│   ├── training/         # Training utilities
│   └── utils/            # Utility functions
└── README.md             # Project documentation
```

## Extending the Code

### Adding New Models
To implement a new model, create a new file in `src/models/` and add a corresponding configuration in `config/model/`.

### Custom Datasets
To use your own dataset, ensure it follows the required format or create a custom dataset class in `src/data/`.

### New Visualization Types
Add new visualization functions in `src/utils/visualization.py` and call them in the evaluation script.

## Paper Citation


## License


## Acknowledgments

# neural_decoding
# neural_decoding
# neural_decoding
