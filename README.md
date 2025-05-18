# ETBiDecSD: Ensemble Transitive Bidirectional Decoupled Self-Distillation for Time Series Classification

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.3%2B-orange)](https://pytorch.org/)

This repository provides the implementation of ETBiDecSD ("Ensemble Transitive Bidirectional Decoupled Self-Distillation for Time Series Classification"), a novel approach for time series classification that combines ensemble learning with bidirectional knowledge distillation.

📄 **Paper Status**: Submitted to IEEE Transactions on Systems, Man, and Cybernetics: Systems Journal


## 📥 Dataset Preparation

### UCR Archive Datasets
1. Download datasets from [UCR Time Series Classification Archive](http://timeseriesclassification.com/dataset.php)
2. Unzip the downloaded datasets
3. Place them in the `UCRData` folder with the following structure:

## 🛠 Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/ETBiDecSD.git
cd ETBiDecSD
```

2. Install required packages:

```
pip install -r requirements.txt
```

## 🚀 Quick Start

python main.py

## 📂 Code Implementation Repository Structure
```
ETBiDecSD/
├── UCRData/ # Dataset directory
├── network/
│ ├── network.py # Model architecture
│ └── decouple_loss.py # Custom loss functions
├── processing/
│ ├── config.py # Configuration
│ ├── data.py # Data processing
│ └── augmentation.py # Data augmentation
├── SaveModel/  # Save Model 
│ ├── Beef.pkl # Saved Beef weights (demo)
├── main.py # Main implementation script (e.g., Train and Test)
├── requirements.txt # Dependencies
└── README.md # This file
```

## ⚙️ Configuration Parameters

All model configurations are set in `processing/config.py`. Key parameters include:

### Dataset Configuration
```python
data_lib_url = "http://timeseriesclassification.com/dataset.php"  # UCR dataset source
univarite_dimension = 1                                          # Input dimension (1 for univariate)
store_local_lib = "UCRData/"                                     # Local dataset storage path
dataset_name = "Beef"                                            # Current dataset name
```
### Hardware Setup
```
GPU_number = 0  # GPU index to use (0 for first GPU)
```
### Loss Function Parameters
```
loss_coefficient = 0.1  # Weight for distillation loss
temperate = 1.0         # Temperature for distillation
alpha = 1.0             # Weight for target class loss
beta = 1.0              # Weight for non-target class loss
```
### Training Parameters
```
learning_rate = 0.0001  # Initial learning rate
weight_decay = 0.0005   # L2 regularization weight
epoch = 500             # Total training epochs
```







## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## Cite
```
{
Authors="Zhiwen Xiao, Huanlai Xing, Rong Qu, Hui Li, Li Feng, Bowen Zhao, and Qian Wan",
Title = "Ensemble Transitive Bidirectional Decoupled Self-Distillation for Time Series Classification,"
Journals = "IEEE Transactions on Systems, Man, and Cybernetics: Systems (Under Review)",
}
```
